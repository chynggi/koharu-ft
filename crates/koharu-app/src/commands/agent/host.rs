use std::{future::Future, pin::Pin, sync::OnceLock};

use anyhow::{Context as _, Result, anyhow, bail};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use koharu_agent::{Control, Host, Invocation, Tool, ToolCall};
use koharu_desktop::{Desktop, Frame};
use koharu_pipeline::{Committer, Operation, RunStatus, Scope, Stage, StageOutput, StopToken};
use koharu_scene::{Commit, EntityId, Snapshot};
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::{Value, json};
use tauri::{AppHandle, Manager as _};
use tauri_runtime_cef::CefRuntime;

use crate::commands::{
    canvas::{CanvasChannel, Point},
    editing::{GeometryUpdate, TypographyUpdate},
    output,
    preferences::Preferences,
    processing::{JobId, Processing},
    project::{CurrentProject, Project, Typography},
};

#[derive(Clone)]
pub(super) struct KoharuHost {
    handle: AppHandle<CefRuntime>,
}

impl KoharuHost {
    pub(super) fn new(handle: AppHandle<CefRuntime>) -> Self {
        Self { handle }
    }

    async fn pages(&self) -> Result<(Value, Value)> {
        let (project, pages) = {
            let current = self.handle.state::<CurrentProject>();
            let current = current.project.lock().await;
            let project = current.as_ref().context("no project is open")?;
            let snapshot = project.snapshot();
            let pages = Project::pages(&snapshot)?
                .into_iter()
                .map(|page| Project::page(&snapshot, page.id))
                .collect::<Result<Vec<_>>>()?;
            (project.info(), pages)
        };
        Ok((serde_json::to_value(project)?, serde_json::to_value(pages)?))
    }

    /// Details only the active page; other pages are summarized and read on
    /// demand through `inspect_page`, so the context stays bounded by one page
    /// instead of growing with the project.
    async fn project_context(&self) -> Result<Value> {
        let (project, pages) = self.pages().await?;
        let pages = pages.as_array().map(Vec::as_slice).unwrap_or_default();
        let active_page = pages
            .iter()
            .find(|page| page["id"] == project["active_page"])
            .or_else(|| pages.first())
            .cloned()
            .map(compact);
        let pages = pages.iter().map(page_summary).collect::<Vec<_>>();
        let preferences = Preferences::load()?;
        let fonts = self
            .handle
            .state::<Desktop>()
            .renderer()
            .available_fonts()
            .await?;
        let providers = preferences
            .providers
            .entries
            .iter()
            .map(|provider| {
                json!({
                    "name": provider.name,
                    "config": provider.config,
                    "credential_configured": provider
                        .credential
                        .as_ref()
                        .is_some_and(|credential| credential.configured),
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "project": project,
            "pages": pages,
            "active_page": active_page,
            "configuration": {
                "pipeline": preferences.pipeline,
                "providers": providers,
                "typesetting": preferences.typesetting,
                "available_fonts": fonts.into_iter().map(|font| font.name).collect::<Vec<_>>(),
            },
        }))
    }

    async fn mutate<T>(
        &self,
        mutation: impl for<'project> FnOnce(
            &'project mut Project,
        ) -> Pin<
            Box<dyn Future<Output = Result<(Commit, T)>> + Send + 'project>,
        >,
    ) -> Result<Invocation>
    where
        T: serde::Serialize + Send,
    {
        let (commit, value, page, project) = {
            let current = self.handle.state::<CurrentProject>();
            let mut current = current.project.lock().await;
            let project = current.as_mut().context("no project is open")?;
            let (commit, value) = mutation(project).await?;
            project.record_commit(&commit);
            project.reconcile_page();
            (commit, value, project.active_page(), project.info())
        };
        let revision = commit.revision;
        self.synchronize(commit, page).await?;
        Invocation::changed(json!({
            "revision": revision,
            "project": project,
            "result": value,
        }))
    }

    async fn synchronize(&self, commit: Commit, page: Option<EntityId>) -> Result<()> {
        let desktop = self.handle.state::<Desktop>();
        desktop.synchronize(&commit.snapshot, page, &commit).await?;
        let canvas = desktop.canvas_state();
        self.handle.state::<CanvasChannel>().publish(canvas);
        Ok(())
    }

    async fn run_pipeline(&self, arguments: RunPipeline, control: &Control) -> Result<Invocation> {
        let scope = arguments.scope()?;
        let operation = arguments.operation.into();
        let snapshot = self
            .handle
            .state::<CurrentProject>()
            .project
            .lock()
            .await
            .as_ref()
            .context("no project is open")?
            .snapshot();
        let job = JobId::new();
        let stop = StopToken::default();
        {
            let processing = self.handle.state::<Processing>();
            let mut stops = processing.stops.lock();
            if !stops.is_empty() {
                bail!("another pipeline process is already running");
            }
            stops.insert(job, stop.clone());
        }

        let watcher = tauri::async_runtime::spawn({
            let control = control.clone();
            let stop = stop.clone();
            async move {
                control.cancelled().await;
                stop.stop();
            }
        });
        let mut committer = AgentCommitter { host: self.clone() };
        let request = koharu_pipeline::Request {
            operation,
            scope,
            stop,
            progress: None,
            inpainting_mask: None,
            text_box: None,
        };
        let result = self
            .handle
            .state::<koharu_pipeline::Pipeline>()
            .execute(snapshot, request, &mut committer)
            .await;
        watcher.abort();
        self.handle.state::<Processing>().stops.lock().remove(&job);
        let report = result.map_err(|error| anyhow!(error))?;
        if report.status == RunStatus::Stopped {
            bail!("pipeline processing was cancelled");
        }
        Invocation::changed(json!({
            "base_revision": report.base,
            "final_revision": report.final_revision,
            "completed": report.completed,
            "total": report.total,
            "elapsed_ms": report.elapsed.as_millis(),
        }))
    }
}

#[async_trait]
impl Host for KoharuHost {
    async fn context(&self) -> Result<Value> {
        self.project_context().await
    }

    fn tools(&self) -> Vec<Tool> {
        static TOOLS: OnceLock<Vec<Tool>> = OnceLock::new();
        TOOLS
            .get_or_init(|| {
                vec![
                    definition::<InspectProject>(
                        "inspect_project",
                        "Read the latest project state after edits: a summary of every page and the active page in detail. This does not include page images. Geometry is given as rounded page-space bounds [left, top, right, bottom].",
                    ),
                    definition::<InspectPage>(
                        "inspect_page",
                        "Read one page's layers, text, and analysis regions. The project state details only the active page.",
                    ),
                    definition::<InspectGeometry>(
                        "inspect_geometry",
                        "Return the exact page-space polygon of each listed element. Call this only when an edit needs more than the bounds in the project state.",
                    ),
                    definition::<ViewPage>(
                        "view_page",
                        "Render and inspect one page image. Call this only for pages whose visual appearance matters to the request.",
                    ),
                    definition::<RenamePage>("rename_page", "Rename a project page."),
                    definition::<MovePage>("move_page", "Move a page to a zero-based project index."),
                    definition::<DeletePages>("delete_pages", "Delete pages and their contents."),
                    definition::<AddTextBox>("add_text_box", "Add a paragraph text box to a page."),
                    definition::<SetText>("set_source_text", "Replace an element's source text."),
                    definition::<SetTranslation>(
                        "set_translation",
                        "Set an element's translation, or pass null to remove it.",
                    ),
                    definition::<SetTypography>(
                        "set_typography",
                        "Replace an element's typography settings. Font is a family name from available_fonts.",
                    ),
                    definition::<SetGeometry>(
                        "set_geometry",
                        "Replace an element's page-space polygon geometry.",
                    ),
                    definition::<SetVisibility>(
                        "set_visibility",
                        "Change visibility and/or opacity for elements.",
                    ),
                    definition::<DeleteElements>("delete_elements", "Delete elements and descendants."),
                    definition::<MoveElement>(
                        "move_element",
                        "Move an element under a page or element at a zero-based index.",
                    ),
                    definition::<RunPipeline>(
                        "run_pipeline",
                        "Run Koharu's configured processing pipeline for the whole project, selected pages, or selected text elements.",
                    ),
                ]
            })
            .clone()
    }

    async fn invoke(&self, call: ToolCall, control: &Control) -> Result<Invocation> {
        match call.name.as_str() {
            "inspect_project" => {
                let _: InspectProject = arguments(&call)?;
                Invocation::read(self.project_context().await?)
            }
            "inspect_page" => {
                let arguments: InspectPage = arguments(&call)?;
                let (_, pages) = self.pages().await?;
                let page = pages
                    .as_array()
                    .and_then(|pages| pages.iter().find(|page| page["id"] == arguments.page))
                    .with_context(|| format!("page {} does not exist", arguments.page))?;
                Invocation::read(compact(page.clone()))
            }
            "inspect_geometry" => {
                let arguments: InspectGeometry = arguments(&call)?;
                let (_, pages) = self.pages().await?;
                let mut found = serde_json::Map::new();
                collect_geometry(&pages, &arguments.elements, &mut found);
                if let Some(missing) = arguments
                    .elements
                    .iter()
                    .find(|element| !found.contains_key(*element))
                {
                    bail!("element {missing} has no geometry");
                }
                Invocation::read(found)
            }
            "view_page" => {
                let arguments: ViewPage = arguments(&call)?;
                let page = entity(&arguments.page)?;
                let (label, snapshot) = {
                    let current = self.handle.state::<CurrentProject>();
                    let current = current.project.lock().await;
                    let project = current.as_ref().context("no project is open")?;
                    let snapshot = project.snapshot();
                    let label = snapshot.page(page)?.page()?.label;
                    (label, snapshot)
                };
                let desktop = self.handle.state::<Desktop>();
                let renderer = desktop.renderer();
                let rasterizer = desktop.rasterizer().await?;
                let bytes =
                    output::rendered_preview(&renderer, rasterizer, &snapshot, page).await?;
                Ok(
                    Invocation::read(json!({ "page": page, "label": label }))?.with_image(
                        format!("Rendered page {label} ({page})"),
                        format!("data:image/webp;base64,{}", STANDARD.encode(bytes)),
                    ),
                )
            }
            "rename_page" => {
                let arguments: RenamePage = arguments(&call)?;
                let page = entity(&arguments.page)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.rename_page(page, arguments.label).await?,
                            json!({ "page": page }),
                        ))
                    })
                })
                .await
            }
            "move_page" => {
                let arguments: MovePage = arguments(&call)?;
                let page = entity(&arguments.page)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.move_page(page, arguments.index).await?,
                            json!({ "page": page }),
                        ))
                    })
                })
                .await
            }
            "delete_pages" => {
                let arguments: DeletePages = arguments(&call)?;
                let pages = entities(&arguments.pages)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.delete_pages(pages.clone()).await?,
                            json!({ "pages": pages }),
                        ))
                    })
                })
                .await
            }
            "add_text_box" => {
                let arguments: AddTextBox = arguments(&call)?;
                let page = entity(&arguments.page)?;
                let frame = Frame {
                    x: arguments.x,
                    y: arguments.y,
                    width: arguments.width,
                    height: arguments.height,
                    angle_degrees: arguments.angle_degrees,
                };
                self.mutate(|project| {
                    Box::pin(async move {
                        let (commit, element) = project.add_text_box(page, frame).await?;
                        Ok((commit, json!({ "page": page, "element": element })))
                    })
                })
                .await
            }
            "set_source_text" => {
                let arguments: SetText = arguments(&call)?;
                let element = entity(&arguments.element)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.set_source_text(element, arguments.text).await?,
                            json!({ "element": element }),
                        ))
                    })
                })
                .await
            }
            "set_translation" => {
                let arguments: SetTranslation = arguments(&call)?;
                let element = entity(&arguments.element)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.set_translation(element, arguments.text).await?,
                            json!({ "element": element }),
                        ))
                    })
                })
                .await
            }
            "set_typography" => {
                let arguments: SetTypography = arguments(&call)?;
                let element = entity(&arguments.element)?;
                let typography = Typography {
                    preferred_font: arguments.preferred_font,
                    font_weight: arguments.font_weight,
                    font_style: arguments.font_style.map(Into::into),
                    size: arguments.size,
                    auto_fit: arguments.auto_fit,
                    color: arguments.color,
                    stroke_color: arguments.stroke_color,
                    stroke_width: arguments.stroke_width,
                    stroke_method: arguments.stroke_method.map(|value| match value {
                        AgentStrokeMethod::Expansion => koharu_scene::StrokeMethod::Expansion,
                        AgentStrokeMethod::Contour => koharu_scene::StrokeMethod::Contour,
                    }),
                    stroke_position: arguments.stroke_position.map(|value| match value {
                        AgentStrokePosition::Outside => koharu_scene::StrokePosition::Outside,
                        AgentStrokePosition::Center => koharu_scene::StrokePosition::Center,
                        AgentStrokePosition::Inside => koharu_scene::StrokePosition::Inside,
                    }),
                    stroke_join: arguments.stroke_join.map(|value| match value {
                        AgentStrokeJoin::Round => koharu_scene::LineJoin::Round,
                        AgentStrokeJoin::Miter => koharu_scene::LineJoin::Miter,
                        AgentStrokeJoin::Bevel => koharu_scene::LineJoin::Bevel,
                    }),
                    outer_stroke_color: arguments.outer_stroke_color,
                    outer_stroke_width: arguments.outer_stroke_width,
                    alignment: arguments.alignment.map(Into::into),
                    writing_mode: arguments.writing_mode.map(Into::into),
                };
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project
                                .set_typography(vec![TypographyUpdate {
                                    layer: element,
                                    typography,
                                }])
                                .await?,
                            json!({ "element": element }),
                        ))
                    })
                })
                .await
            }
            "set_geometry" => {
                let arguments: SetGeometry = arguments(&call)?;
                let element = entity(&arguments.element)?;
                let points = arguments
                    .points
                    .into_iter()
                    .map(|point| Point {
                        x: point.x,
                        y: point.y,
                    })
                    .collect();
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project
                                .set_geometry(vec![GeometryUpdate {
                                    layer: element,
                                    points: Some(points),
                                }])
                                .await?,
                            json!({ "element": element }),
                        ))
                    })
                })
                .await
            }
            "set_visibility" => {
                let arguments: SetVisibility = arguments(&call)?;
                let elements = entities(&arguments.elements)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project
                                .set_visibility(
                                    elements.clone(),
                                    arguments.visible,
                                    arguments.opacity,
                                )
                                .await?,
                            json!({ "elements": elements }),
                        ))
                    })
                })
                .await
            }
            "delete_elements" => {
                let arguments: DeleteElements = arguments(&call)?;
                let elements = entities(&arguments.elements)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.delete_layers(elements.clone()).await?,
                            json!({ "elements": elements }),
                        ))
                    })
                })
                .await
            }
            "move_element" => {
                let arguments: MoveElement = arguments(&call)?;
                let element = entity(&arguments.element)?;
                let parent = entity(&arguments.parent)?;
                self.mutate(|project| {
                    Box::pin(async move {
                        Ok((
                            project.move_layer(element, parent, arguments.index).await?,
                            json!({ "element": element, "parent": parent }),
                        ))
                    })
                })
                .await
            }
            "run_pipeline" => self.run_pipeline(arguments(&call)?, control).await,
            name => bail!("unknown Koharu tool {name}"),
        }
    }
}

struct AgentCommitter {
    host: KoharuHost,
}

#[async_trait]
impl Committer for AgentCommitter {
    async fn commit(&mut self, output: StageOutput) -> Result<Snapshot> {
        let (commit, page) = {
            let current = self.host.handle.state::<CurrentProject>();
            let mut current = current.project.lock().await;
            let project = current.as_mut().context("no project is open")?;
            let Some(commit) = project.commit_rebased(output.patch).await? else {
                return Ok(project.snapshot());
            };
            project.record_commit(&commit);
            (commit, project.active_page())
        };
        let snapshot = commit.snapshot.clone();
        self.host.synchronize(commit, page).await?;
        Ok(snapshot)
    }
}

fn definition<T: JsonSchema>(name: &str, description: &str) -> Tool {
    Tool::new(
        name,
        description,
        serde_json::to_value(schema_for!(T)).expect("tool argument schema must serialize"),
    )
}

fn arguments<T: for<'de> Deserialize<'de>>(call: &ToolCall) -> Result<T> {
    serde_json::from_str(&call.arguments)
        .with_context(|| format!("invalid arguments for {}", call.name))
}

fn entity(value: &str) -> Result<EntityId> {
    serde_json::from_value(Value::String(value.to_owned()))
        .with_context(|| format!("invalid entity ID {value}"))
}

fn entities(values: &[String]) -> Result<Vec<EntityId>> {
    values.iter().map(|value| entity(value)).collect()
}

fn page_summary(page: &Value) -> Value {
    let text = || {
        page["layers"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|layer| layer["type"] == "text")
    };
    json!({
        "id": page["id"],
        "label": page["label"],
        "size": page["size"],
        "text_layers": text().count(),
        "translated": text()
            .filter(|layer| !layer["content"]["translation"].is_null())
            .count(),
    })
}

/// Shapes a page for the model: rounded bounds instead of polygons, and no
/// null fields, which carry no information but cost tokens on every element.
fn compact(mut page: Value) -> Value {
    compact_geometry(&mut page);
    strip_nulls(&mut page);
    page
}

fn strip_nulls(value: &mut Value) {
    match value {
        Value::Object(object) => {
            object.retain(|_, value| !value.is_null());
            object.values_mut().for_each(strip_nulls);
        }
        Value::Array(items) => items.iter_mut().for_each(strip_nulls),
        _ => {}
    }
}

/// Detection traces each bubble mask as a dense polygon, so exact geometry for
/// every page would dominate the model's context on every turn. The context
/// carries rounded bounds instead, and `inspect_geometry` returns exact
/// polygons on demand.
fn compact_geometry(value: &mut Value) {
    match value {
        Value::Object(object) => {
            if let Some(geometry) = object.get_mut("geometry")
                && let Some(bounds) = bounds(geometry)
            {
                *geometry = bounds;
            }
            object.values_mut().for_each(compact_geometry);
        }
        Value::Array(items) => items.iter_mut().for_each(compact_geometry),
        _ => {}
    }
}

fn bounds(geometry: &Value) -> Option<Value> {
    let points = geometry.get("points")?.as_array()?;
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for point in points {
        let (x, y) = (point.get("x")?.as_f64()?, point.get("y")?.as_f64()?);
        bounds = [
            bounds[0].min(x),
            bounds[1].min(y),
            bounds[2].max(x),
            bounds[3].max(y),
        ];
    }
    let [left, top, right, bottom] = bounds;
    (!points.is_empty()).then(|| {
        json!([
            left.floor() as i64,
            top.floor() as i64,
            right.ceil() as i64,
            bottom.ceil() as i64
        ])
    })
}

fn collect_geometry(
    value: &Value,
    elements: &[String],
    found: &mut serde_json::Map<String, Value>,
) {
    match value {
        Value::Object(object) => {
            if let (Some(Value::String(id)), Some(geometry)) =
                (object.get("id"), object.get("geometry"))
                && !geometry.is_null()
                && elements.contains(id)
            {
                found.insert(id.clone(), geometry.clone());
            }
            for value in object.values() {
                collect_geometry(value, elements, found);
            }
        }
        Value::Array(items) => {
            for value in items {
                collect_geometry(value, elements, found);
            }
        }
        _ => {}
    }
}

#[derive(Deserialize, JsonSchema)]
struct InspectProject {}

#[derive(Deserialize, JsonSchema)]
struct InspectPage {
    page: String,
}

#[derive(Deserialize, JsonSchema)]
struct InspectGeometry {
    elements: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
struct ViewPage {
    page: String,
}

#[derive(Deserialize, JsonSchema)]
struct RenamePage {
    page: String,
    label: String,
}

#[derive(Deserialize, JsonSchema)]
struct MovePage {
    page: String,
    index: usize,
}

#[derive(Deserialize, JsonSchema)]
struct DeletePages {
    pages: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
struct AddTextBox {
    page: String,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    #[serde(default)]
    angle_degrees: f32,
}

#[derive(Deserialize, JsonSchema)]
struct SetText {
    element: String,
    text: String,
}

#[derive(Deserialize, JsonSchema)]
struct SetTranslation {
    element: String,
    text: Option<String>,
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum AgentTextAlignment {
    Start,
    Center,
    End,
    Justify,
}

impl From<AgentTextAlignment> for koharu_scene::TextAlignment {
    fn from(value: AgentTextAlignment) -> Self {
        match value {
            AgentTextAlignment::Start => Self::Start,
            AgentTextAlignment::Center => Self::Center,
            AgentTextAlignment::End => Self::End,
            AgentTextAlignment::Justify => Self::Justify,
        }
    }
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum AgentWritingMode {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum AgentFontStyle {
    Normal,
    Italic,
    Oblique,
}

impl From<AgentFontStyle> for koharu_scene::FontStyle {
    fn from(value: AgentFontStyle) -> Self {
        match value {
            AgentFontStyle::Normal => Self::Normal,
            AgentFontStyle::Italic => Self::Italic,
            AgentFontStyle::Oblique => Self::Oblique,
        }
    }
}

impl From<AgentWritingMode> for koharu_scene::WritingMode {
    fn from(value: AgentWritingMode) -> Self {
        match value {
            AgentWritingMode::Horizontal => Self::Horizontal,
            AgentWritingMode::Vertical => Self::Vertical,
        }
    }
}

#[derive(Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum AgentStrokeMethod {
    Expansion,
    Contour,
}

#[derive(Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum AgentStrokePosition {
    Outside,
    Center,
    Inside,
}

#[derive(Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum AgentStrokeJoin {
    Round,
    Miter,
    Bevel,
}

#[derive(Deserialize, JsonSchema)]
struct SetTypography {
    element: String,
    preferred_font: Option<String>,
    font_weight: Option<u16>,
    font_style: Option<AgentFontStyle>,
    size: Option<f32>,
    auto_fit: bool,
    color: Option<[u8; 4]>,
    stroke_color: Option<[u8; 4]>,
    stroke_width: Option<f32>,
    stroke_method: Option<AgentStrokeMethod>,
    stroke_position: Option<AgentStrokePosition>,
    stroke_join: Option<AgentStrokeJoin>,
    /// Color of a second outline outside the first.
    outer_stroke_color: Option<[u8; 4]>,
    /// Width of the second outline in pixels; omit or 0 for none.
    outer_stroke_width: Option<f32>,
    alignment: Option<AgentTextAlignment>,
    writing_mode: Option<AgentWritingMode>,
}

#[derive(Deserialize, JsonSchema)]
struct SetGeometry {
    element: String,
    points: Vec<AgentPoint>,
}

#[derive(Deserialize, JsonSchema)]
struct AgentPoint {
    x: f64,
    y: f64,
}

#[derive(Deserialize, JsonSchema)]
struct SetVisibility {
    elements: Vec<String>,
    visible: Option<bool>,
    opacity: Option<f32>,
}

#[derive(Deserialize, JsonSchema)]
struct DeleteElements {
    elements: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
struct MoveElement {
    element: String,
    parent: String,
    index: usize,
}

#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum AgentPipelineOperation {
    Full,
    Detection,
    Ocr,
    Translation,
    Inpainting,
}

impl From<AgentPipelineOperation> for Operation {
    fn from(value: AgentPipelineOperation) -> Self {
        match value {
            AgentPipelineOperation::Full => Self::Full,
            AgentPipelineOperation::Detection => Self::Only {
                stage: Stage::Detection,
            },
            AgentPipelineOperation::Ocr => Self::Only { stage: Stage::Ocr },
            AgentPipelineOperation::Translation => Self::Only {
                stage: Stage::Translation,
            },
            AgentPipelineOperation::Inpainting => Self::Only {
                stage: Stage::Inpainting,
            },
        }
    }
}

#[derive(Deserialize, JsonSchema)]
struct RunPipeline {
    operation: AgentPipelineOperation,
    #[serde(default)]
    pages: Vec<String>,
    #[serde(default)]
    elements: Vec<String>,
}

impl RunPipeline {
    fn scope(&self) -> Result<Scope> {
        match (self.pages.is_empty(), self.elements.is_empty()) {
            (true, true) => Ok(Scope::Project),
            (false, true) => Ok(Scope::Pages(entities(&self.pages)?)),
            (true, false) => Ok(Scope::Entities(entities(&self.elements)?)),
            (false, false) => bail!("pipeline scope cannot contain both pages and elements"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_carries_bounds_and_inspection_returns_polygons() {
        let polygon = json!({ "points": [
            { "x": 10.4, "y": 20.6 },
            { "x": 30.2, "y": 21.0 },
            { "x": 29.9, "y": 40.1 },
        ] });
        let pages = json!([{
            "id": "page",
            "layers": [{ "id": "text", "geometry": polygon.clone() }, { "id": "group", "geometry": null }],
            "regions": [{ "id": "bubble", "geometry": polygon.clone() }],
        }]);

        let mut context = pages.clone();
        compact_geometry(&mut context);
        assert_eq!(context[0]["layers"][0]["geometry"], json!([10, 20, 31, 41]));
        assert_eq!(context[0]["layers"][1]["geometry"], Value::Null);
        assert_eq!(
            context[0]["regions"][0]["geometry"],
            json!([10, 20, 31, 41])
        );

        let page = compact(pages[0].clone());
        assert!(page["layers"][1].get("geometry").is_none());
        assert_eq!(
            page_summary(&pages[0]),
            json!({ "id": "page", "label": null, "size": null, "text_layers": 0, "translated": 0 })
        );

        let mut found = serde_json::Map::new();
        collect_geometry(&pages, &["bubble".to_owned()], &mut found);
        assert_eq!(found.get("bubble"), Some(&polygon));
        assert_eq!(found.len(), 1);
    }
}
