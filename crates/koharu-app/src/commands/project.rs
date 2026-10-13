use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    path::PathBuf,
    sync::Arc,
};

use anyhow::{Context as _, Result, anyhow, bail};
use image::{DynamicImage, ImageFormat, RgbaImage};
use koharu_desktop::Frame;
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, Authored, Commit, EntityId, EntityOrigin, FitsTo,
    Geometry as SceneGeometry, Group as SceneGroup, Origin, PageDraft, Point as ScenePoint,
    Presents, RasterLayer as SceneRasterLayer, RasterLayerKind, RecognizedFrom,
    Region as SceneRegion, RegionSpec, RemovePolicy, Revision, Session, ShapeGeometry,
    ShapeLayer as SceneShapeLayer, ShapeStyle, Snapshot, SourceText as SceneSourceText,
    TextGroup as SceneTextGroup, TextLayout as SceneTextLayout, TextLayoutKind, TextRegion,
    TextRole, Translation as SceneTranslation, Typography as SceneTypography,
    Visibility as SceneVisibility,
};
use serde::{Deserialize, Serialize};
use specta::Type;
use tokio::sync::Mutex;

use super::{
    canvas::Point,
    editing::{GeometryUpdate, TypographyUpdate},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RasterStrokeMode {
    Paint,
    Erase,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct ProjectInfo {
    pub name: String,
    pub revision: Revision,
    pub active_page: Option<EntityId>,
    pub can_undo: bool,
    pub can_redo: bool,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct ProjectSummary {
    pub name: String,
}

#[derive(Clone, Copy, Debug, Serialize, Type)]
pub struct PageSize {
    pub width: f64,
    pub height: f64,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct PageSummary {
    pub id: EntityId,
    pub label: String,
    pub size: PageSize,
    pub source_asset: Option<String>,
    pub chapter_start: bool,
    #[specta(type = f64)]
    pub layer_count: usize,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct Page {
    pub id: EntityId,
    pub label: String,
    pub size: PageSize,
    pub layers: Vec<Layer>,
    pub regions: Vec<AnalysisRegion>,
}

#[derive(Clone, Debug, Serialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Layer {
    Group {
        id: EntityId,
        parent: Option<EntityId>,
        visibility: LayerVisibility,
        name: String,
        role: Option<GroupRole>,
    },
    Text {
        id: EntityId,
        parent: Option<EntityId>,
        geometry: Option<Geometry>,
        angle_degrees: Option<f32>,
        visibility: LayerVisibility,
        content: Box<TextContent>,
        typography: Option<Typography>,
        layout: TextLayoutKind,
        automatic_region: Option<EntityId>,
    },
    Raster {
        id: EntityId,
        parent: Option<EntityId>,
        visibility: LayerVisibility,
        image: Option<String>,
        name: String,
        kind: RasterLayerKind,
    },
    Shape {
        id: EntityId,
        parent: Option<EntityId>,
        geometry: Option<Geometry>,
        angle_degrees: Option<f32>,
        visibility: LayerVisibility,
        shape: ShapeGeometry,
        style: ShapeStyle,
        rasterized: bool,
    },
    Image {
        id: EntityId,
        parent: Option<EntityId>,
        geometry: Geometry,
        visibility: LayerVisibility,
        image: String,
    },
    Artwork {
        id: EntityId,
        parent: Option<EntityId>,
        geometry: Geometry,
        visibility: LayerVisibility,
        image: String,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum GroupRole {
    Text,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct TextContent {
    pub id: EntityId,
    pub source: Option<SourceText>,
    pub translation: Option<Translation>,
    pub role: Option<String>,
    pub source_region: Option<EntityId>,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct AnalysisRegion {
    pub id: EntityId,
    pub parent: Option<EntityId>,
    pub geometry: Geometry,
    pub kind: String,
    pub label: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Type)]
pub struct Geometry {
    pub points: Vec<Point>,
}

impl Geometry {
    /// The transport view carries only points; the scene component also carries
    /// authorship, which every caller of this promotes to `User` anyway.
    pub fn into_scene(self) -> SceneGeometry {
        SceneGeometry {
            origin: Origin::User,
            points: self
                .points
                .into_iter()
                .map(|point| ScenePoint {
                    x: point.x,
                    y: point.y,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Type)]
pub struct LayerVisibility {
    pub visible: bool,
    pub opacity: f32,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct SourceText {
    pub text: String,
    pub language: Option<String>,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct Translation {
    pub text: String,
    pub language: Option<String>,
}

#[derive(Clone, Debug, serde::Deserialize, Serialize, Type)]
pub struct Typography {
    pub preferred_font: Option<String>,
    pub font_weight: Option<u16>,
    pub font_style: Option<koharu_scene::FontStyle>,
    pub size: Option<f32>,
    pub auto_fit: bool,
    pub color: Option<[u8; 4]>,
    pub stroke_color: Option<[u8; 4]>,
    pub stroke_width: Option<f32>,
    pub alignment: Option<koharu_scene::TextAlignment>,
    pub writing_mode: Option<koharu_scene::WritingMode>,
}

pub struct CurrentProject {
    pub project: Mutex<Option<Project>>,
}

#[derive(Clone)]
pub struct ProjectLibrary {
    root: PathBuf,
}

impl ProjectLibrary {
    pub fn new() -> Result<Self> {
        let root = dirs::document_dir()
            .context("the Documents directory is unavailable")?
            .join("Koharu");
        std::fs::create_dir_all(&root)
            .with_context(|| format!("failed to create {}", root.display()))?;
        Ok(Self { root })
    }

    pub fn list(&self) -> Result<Vec<ProjectSummary>> {
        let mut projects = std::fs::read_dir(&self.root)
            .with_context(|| format!("failed to read {}", self.root.display()))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
            .filter_map(|entry| {
                let path = entry.path();
                let is_project_directory = path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("khrproj"))
                    && (path.join("state-a.khr").is_file() || path.join("state-b.khr").is_file());
                if !is_project_directory {
                    return None;
                }
                let last_used = ["state-a.khr", "state-b.khr"]
                    .into_iter()
                    .filter_map(|file| std::fs::metadata(path.join(file)).ok()?.modified().ok())
                    .max()
                    .unwrap_or(std::time::UNIX_EPOCH);
                Some((
                    last_used,
                    ProjectSummary {
                        name: path.file_stem()?.to_str()?.to_owned(),
                    },
                ))
            })
            .collect::<Vec<_>>();
        projects.sort_unstable_by(|(left_used, left), (right_used, right)| {
            right_used
                .cmp(left_used)
                .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
        });
        Ok(projects.into_iter().map(|(_, project)| project).collect())
    }

    pub async fn create(&self, name: &str) -> Result<Project> {
        let (name, path) = self.resolve(name)?;
        Project::create(name, path).await
    }

    pub async fn open(&self, name: &str) -> Result<Project> {
        let (name, path) = self.resolve(name)?;
        Project::open(name, path).await
    }

    pub fn delete(&self, name: &str) -> Result<()> {
        let (_, path) = self.resolve(name)?;
        if !path.is_dir() {
            bail!("project {name:?} does not exist");
        }
        std::fs::remove_dir_all(&path)
            .with_context(|| format!("failed to delete {}", path.display()))
    }

    fn resolve(&self, name: &str) -> Result<(String, PathBuf)> {
        let name = validate_project_name(name)?;
        Ok((name.clone(), self.root.join(format!("{name}.khrproj"))))
    }
}

pub struct Project {
    pub session: Session,
    pub name: String,
    pub active_page: Option<EntityId>,
    pub undo: Vec<Vec<Revision>>,
    pub redo: Vec<Vec<Revision>>,
}

impl Project {
    pub async fn create(name: String, path: PathBuf) -> Result<Self> {
        let session = Session::create(&path)
            .await
            .with_context(|| format!("failed to create {}", path.display()))?;
        Ok(Self::new(session, name))
    }

    pub async fn open(name: String, path: PathBuf) -> Result<Self> {
        let session = Session::open(&path)
            .await
            .with_context(|| format!("failed to open {}", path.display()))?;
        Ok(Self::new(session, name))
    }

    fn new(session: Session, name: String) -> Self {
        let active_page = session.snapshot().pages().next().map(|page| page.id());
        Self {
            session,
            name,
            active_page,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        self.session.snapshot()
    }

    pub fn revision(&self) -> Revision {
        self.snapshot().revision()
    }

    pub fn active_page(&self) -> Option<EntityId> {
        self.active_page
    }

    pub fn select_page(&mut self, page: EntityId) -> Result<()> {
        self.snapshot().page(page)?;
        self.active_page = Some(page);
        Ok(())
    }

    pub fn reconcile_page(&mut self) {
        let snapshot = self.snapshot();
        if self
            .active_page
            .is_none_or(|page| snapshot.page(page).is_err())
        {
            self.active_page = snapshot.pages().next().map(|page| page.id());
        }
    }

    pub fn info(&self) -> ProjectInfo {
        ProjectInfo {
            name: self.name.clone(),
            revision: self.revision(),
            active_page: self.active_page,
            can_undo: !self.undo.is_empty(),
            can_redo: !self.redo.is_empty(),
        }
    }

    pub fn pages(snapshot: &Snapshot) -> Result<Vec<PageSummary>> {
        snapshot
            .pages()
            .map(|page| {
                let value = page.page()?;
                let source_asset = Self::asset_id(snapshot, page.id(), "source")?;
                let layer_count = snapshot
                    .descendants(page.id())?
                    .map(|entity| Self::is_content_layer(snapshot, entity.id()))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .filter(|present| *present)
                    .count()
                    + usize::from(source_asset.is_some());
                Ok(PageSummary {
                    id: page.id(),
                    label: value.label,
                    size: PageSize {
                        width: value.width,
                        height: value.height,
                    },
                    source_asset,
                    chapter_start: value.chapter_start,
                    layer_count,
                })
            })
            .collect()
    }

    pub async fn rename_page(&mut self, page: EntityId, label: String) -> Result<Commit> {
        let snapshot = self.snapshot();
        let current = snapshot.page(page)?.page()?;
        let patch = snapshot.patch(|edit| {
            edit.set_page(
                page,
                PageDraft::new(label, current.width, current.height)
                    .with_chapter_start(current.chapter_start),
            )
        })?;
        self.commit(patch).await
    }

    pub(crate) async fn set_page_chapter_start(
        &mut self,
        page: EntityId,
        chapter_start: bool,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let current = snapshot.page(page)?.page()?;
        let patch = snapshot.patch(|edit| {
            edit.set_page(
                page,
                PageDraft::new(current.label, current.width, current.height)
                    .with_chapter_start(chapter_start),
            )
        })?;
        self.commit(patch).await
    }

    pub async fn delete_pages(&mut self, pages: Vec<EntityId>) -> Result<Commit> {
        let snapshot = self.snapshot();
        let pages = Self::unique_roots(&snapshot, pages)?;
        let patch = snapshot.patch(|edit| {
            for page in pages {
                edit.remove_entity(page, RemovePolicy::Cascade)?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub async fn move_page(&mut self, page: EntityId, index: usize) -> Result<Commit> {
        let snapshot = self.snapshot();
        let siblings = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
        let at = Self::placement(&siblings, page, index);
        let patch = snapshot.patch(|edit| edit.move_entity(page, None, at))?;
        self.commit(patch).await
    }

    /// Combines an adjacent run of pages into one wide spread page. Source and
    /// cleanup pixels are laid out left to right, while every child layer, text
    /// group, translation, and analysis region is re-parented under the merged
    /// page with the right-hand pages translated by the accumulated width of the
    /// pages to their left. Relations are document scoped, so moving the entities
    /// keeps `Presents`, `RecognizedFrom`, `FitsTo`, `FlowsIn`, and `Inside`
    /// intact rather than rebuilding text content from scratch.
    pub(crate) async fn merge_pages(&mut self, pages: Vec<EntityId>) -> Result<(Commit, EntityId)> {
        let snapshot = self.snapshot();
        let order = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
        let pages = Self::contiguous_run(&order, pages)?;
        let source_role = AssetRole::new("source")?;

        // Read every page-sized image before building the patch, because blob
        // reads are asynchronous while the scene edit itself is synchronous.
        let mut offsets = Vec::with_capacity(pages.len());
        let mut source_images = Vec::with_capacity(pages.len());
        let mut total_width = 0u32;
        let mut total_height = 0u32;
        for &page in &pages {
            let value = snapshot.page(page)?.page()?;
            let width = value.width.round().max(1.0) as u32;
            let height = value.height.round().max(1.0) as u32;
            offsets.push(total_width);
            total_width = total_width
                .checked_add(width)
                .context("the merged page is too wide")?;
            total_height = total_height.max(height);
            let asset = snapshot
                .asset(page, &source_role)?
                .with_context(|| format!("page {page} has no source image to merge"))?;
            source_images.push(snapshot.read_blob(asset.blob).await?);
        }
        if total_width > MAX_MERGE_DIMENSION || total_height > MAX_MERGE_DIMENSION {
            bail!("the merged page is larger than Koharu supports");
        }

        // Cleanup and paint layers are full-page pixel layers, so the renderer
        // stretches them across the spread. Re-pad each to the merged width so its
        // pixels stay in the correct half.
        let mut raster_blobs = Vec::new();
        for (index, &page) in pages.iter().enumerate() {
            for child in snapshot.children(page)? {
                if snapshot.component::<SceneRasterLayer>(child)?.is_none() {
                    continue;
                }
                if let Some(asset) = snapshot.asset(child, &source_role)? {
                    raster_blobs.push((
                        child,
                        offsets[index],
                        snapshot.read_blob(asset.blob).await?,
                    ));
                }
            }
        }

        let compose_offsets = offsets.clone();
        let (merged_source, raster_assets) =
            tokio::task::spawn_blocking(move || -> Result<MergedSpread> {
                let decoded = source_images
                    .iter()
                    .map(|bytes| Ok(image::load_from_memory(bytes)?.to_rgba8()))
                    .collect::<Result<Vec<_>>>()?;
                let merged = compose_row(
                    decoded
                        .iter()
                        .zip(&compose_offsets)
                        .map(|(image, offset)| (i64::from(*offset), image)),
                    total_width,
                    total_height,
                );
                let merged_source = encode_png(merged)?;
                let mut rasters = HashMap::new();
                for (layer, offset, bytes) in raster_blobs {
                    let image = image::load_from_memory(&bytes)?.to_rgba8();
                    let padded = compose_row(
                        std::iter::once((i64::from(offset), &image)),
                        total_width,
                        total_height,
                    );
                    rasters.insert(layer, encode_png(padded)?);
                }
                Ok((merged_source, rasters))
            })
            .await??;

        let first = pages[0];
        let first_page = snapshot.page(first)?.page()?;
        let label = first_page.label;
        let chapter_start = first_page.chapter_start;
        let mut target_group = None;
        for &page in &pages {
            if let Some(group) = snapshot.page(page)?.text_group()? {
                target_group = Some(group.id());
                break;
            }
        }

        let width = f64::from(total_width);
        let height = f64::from(total_height);
        let mut merged_page = None;
        let patch = snapshot.patch(|edit| {
            let merged = edit.add_page(
                PageDraft::new(label.clone(), width, height).with_chapter_start(chapter_start),
                At::Before(first),
            )?;
            merged_page = Some(merged);
            edit.set_asset(
                merged,
                &source_role,
                AssetInput::new(
                    Arc::clone(&merged_source),
                    "image/png",
                    AssetMetadata {
                        width: Some(total_width),
                        height: Some(total_height),
                        attributes: Default::default(),
                    },
                ),
            )?;

            // Adopt one source page's text group as the merged page's single text
            // group; a page may only own one, so the others are emptied instead.
            if let Some(group) = target_group {
                edit.move_entity(group, Some(merged), At::End)?;
            }

            let mut roots = Vec::new();
            for (&page, &offset) in pages.iter().zip(&offsets) {
                for child in snapshot.children(page)? {
                    if Some(child) == target_group {
                        continue;
                    }
                    if snapshot.component::<SceneTextGroup>(child)?.is_some() {
                        let group = target_group
                            .expect("a merged text group exists when text layers are present");
                        for layer in snapshot.children(child)? {
                            roots.push((layer, group, f64::from(offset)));
                        }
                    } else {
                        roots.push((child, merged, f64::from(offset)));
                    }
                }
            }
            for (root, parent, offset) in roots {
                edit.move_entity(root, Some(parent), At::End)?;
                for entity in snapshot.subtree(root)? {
                    let id = entity.id();
                    if let Some(asset) = raster_assets.get(&id) {
                        edit.set_asset(
                            id,
                            &source_role,
                            AssetInput::new(
                                Arc::clone(asset),
                                "image/png",
                                AssetMetadata {
                                    width: Some(total_width),
                                    height: Some(total_height),
                                    attributes: Default::default(),
                                },
                            ),
                        )?;
                    }
                    if offset == 0.0 {
                        continue;
                    }
                    if let Some(geometry) = snapshot.component::<SceneGeometry>(id)? {
                        edit.set(
                            id,
                            &SceneGeometry {
                                origin: geometry.origin,
                                points: geometry
                                    .points
                                    .into_iter()
                                    .map(|point| ScenePoint {
                                        x: point.x + offset,
                                        y: point.y,
                                    })
                                    .collect(),
                            },
                        )?;
                    }
                }
            }

            for &page in &pages {
                edit.remove_entity(page, RemovePolicy::Cascade)?;
            }
            Ok(())
        })?;
        let merged = merged_page.context("the merged page was not created")?;
        let commit = self.commit(patch).await?;
        Ok((commit, merged))
    }

    /// Selects a run of unique, in-project pages, requires them to be adjacent in
    /// reading order, and returns them left to right.
    fn contiguous_run(order: &[EntityId], pages: Vec<EntityId>) -> Result<Vec<EntityId>> {
        let mut selected = pages;
        selected.sort_unstable();
        selected.dedup();
        if selected.len() < 2 {
            bail!("merging a spread needs at least two pages");
        }
        let mut indices = Vec::with_capacity(selected.len());
        for page in &selected {
            let index = order
                .iter()
                .position(|candidate| candidate == page)
                .with_context(|| format!("page {page} does not belong to this project"))?;
            indices.push(index);
        }
        indices.sort_unstable();
        if indices.windows(2).any(|window| window[1] != window[0] + 1) {
            bail!("only adjacent pages can be merged into a spread");
        }
        Ok(indices.into_iter().map(|index| order[index]).collect())
    }

    pub async fn add_point_text(
        &mut self,
        page: EntityId,
        point: Point,
    ) -> Result<(Commit, EntityId)> {
        if !point.x.is_finite() || !point.y.is_finite() {
            bail!("text position must contain finite coordinates");
        }
        let value = self.snapshot().page(page)?.page()?;
        if point.x < 0.0 || point.y < 0.0 || point.x >= value.width || point.y >= value.height {
            bail!("text position must be inside the page");
        }
        self.add_text(
            page,
            Frame {
                x: point.x as f32,
                y: point.y as f32,
                width: (value.width - point.x).clamp(1.0, 320.0) as f32,
                height: (value.height - point.y).clamp(1.0, 120.0) as f32,
                angle_degrees: 0.0,
            },
            TextLayoutKind::Point,
        )
        .await
    }

    pub async fn add_text_box(
        &mut self,
        page: EntityId,
        frame: Frame,
    ) -> Result<(Commit, EntityId)> {
        self.add_text(page, frame, TextLayoutKind::Paragraph).await
    }

    pub(crate) async fn paste_text_layers(
        &mut self,
        page: EntityId,
        layers: Vec<EntityId>,
    ) -> Result<(Commit, Vec<EntityId>)> {
        let snapshot = self.snapshot();
        snapshot.page(page)?;
        let mut seen = HashSet::new();
        let layers = layers
            .into_iter()
            .filter(|layer| seen.insert(*layer))
            .collect::<Vec<_>>();
        if layers.is_empty() {
            bail!("at least one text layer is required");
        }

        let mut added = Vec::with_capacity(layers.len());
        let patch = snapshot.patch(|edit| {
            for source_layer in &layers {
                let source = snapshot.text_layer(*source_layer)?;
                let content = source.content()?;
                let mut layout = source.layout()?;
                layout.origin = Origin::User;
                let mut geometry = source.frame()?;
                if let Some(geometry) = &mut geometry {
                    geometry.origin = Origin::User;
                }
                let mut typography = source.typography()?;
                if let Some(typography) = &mut typography {
                    typography.origin = Origin::User;
                }
                let mut visibility = source.visibility()?.unwrap_or(SceneVisibility {
                    origin: Origin::User,
                    visible: true,
                    opacity: 1.0,
                });
                visibility.origin = Origin::User;
                let source_text = content.source()?;
                let translation = content.translation()?;
                let role = content.role()?;

                let copied_content = edit.add_text_content(page, At::End)?;
                if let Some(mut source_text) = source_text {
                    source_text.text.origin = Origin::User;
                    edit.set(copied_content, &source_text)?;
                }
                if let Some(mut translation) = translation {
                    translation.text.origin = Origin::User;
                    edit.set(copied_content, &translation)?;
                }
                if let Some(role) = role {
                    edit.set(
                        copied_content,
                        &TextRole {
                            origin: Origin::User,
                            role: role.role,
                        },
                    )?;
                }
                let copied_layer = edit.add_text_layer(page, At::End, copied_content, &layout)?;
                if let Some(geometry) = geometry {
                    edit.set(copied_layer, &geometry)?;
                }
                if let Some(typography) = typography {
                    edit.set(copied_layer, &typography)?;
                }
                edit.set(copied_layer, &visibility)?;
                added.push(copied_layer);
            }
            Ok(())
        })?;
        Ok((self.commit(patch).await?, added))
    }

    async fn add_text(
        &mut self,
        page: EntityId,
        frame: Frame,
        kind: TextLayoutKind,
    ) -> Result<(Commit, EntityId)> {
        let snapshot = self.snapshot();
        let geometry = Self::geometry_from_frame(frame)?;
        let mut layer = None;
        let patch = snapshot.patch(|edit| {
            let content = edit.add_text_content(page, At::End)?;
            edit.set(
                content,
                &SceneSourceText {
                    text: Authored::user(String::new()),
                    language: None,
                },
            )?;
            let added_layer = edit.add_text_layer(
                page,
                At::End,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind,
                    angle_degrees: Some(frame.angle_degrees),
                },
            )?;
            layer = Some(added_layer);
            edit.set(added_layer, &geometry)?;
            edit.set(added_layer, &Self::text_typography(kind))?;
            Ok(())
        })?;
        Ok((
            self.commit(patch).await?,
            layer.expect("text layer was added while building the patch"),
        ))
    }

    /// Writes a user-drawn text region where detection missed text. The result is
    /// an ordinary `TextRegion` — kind, geometry, content and presentation layer —
    /// so the OCR stage and downstream stages treat it exactly like a detected
    /// one. The content carries no source text yet, so OCR fills it in.
    pub async fn add_manual_text_region(
        &mut self,
        page: EntityId,
        points: Vec<Point>,
    ) -> Result<(Commit, EntityId, EntityId)> {
        let geometry = Self::geometry_from_points(points)?;
        let snapshot = self.snapshot();
        let mut region = None;
        let mut layer = None;
        let patch = snapshot.patch(|edit| {
            let added_region = edit.add_entity(page, At::End)?;
            edit.set(added_region, &geometry)?;
            edit.set(
                added_region,
                &SceneRegion {
                    origin: Origin::User,
                    kind: TextRegion::kind(),
                    label: Some("manual".to_owned()),
                },
            )?;
            region = Some(added_region);

            let content = edit.add_text_content(page, At::End)?;
            // A manual region has no detected dialogue balloon, so it stays free text.
            edit.set(
                content,
                &TextRole {
                    origin: Origin::User,
                    role: "dev.koharu.text.free-text".to_owned(),
                },
            )?;
            let added_layer = edit.add_text_layer(
                page,
                At::End,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Paragraph,
                    angle_degrees: None,
                },
            )?;
            edit.set(added_layer, &geometry)?;
            edit.set(
                added_layer,
                &Self::text_typography(TextLayoutKind::Paragraph),
            )?;
            edit.relate::<RecognizedFrom>(content, added_region)?;
            edit.relate::<FitsTo>(added_layer, added_region)?;
            layer = Some(added_layer);
            Ok(())
        })?;
        Ok((
            self.commit(patch).await?,
            region.expect("manual region was added while building the patch"),
            layer.expect("manual text layer was added while building the patch"),
        ))
    }

    fn text_typography(kind: TextLayoutKind) -> SceneTypography {
        SceneTypography {
            origin: Origin::User,
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: None,
            auto_fit: true,
            color: None,
            stroke_color: None,
            stroke_width: None,
            alignment: match kind {
                TextLayoutKind::Point => Some(koharu_scene::TextAlignment::Start),
                TextLayoutKind::Paragraph => None,
            },
            writing_mode: None,
            extensions: Default::default(),
        }
    }

    pub async fn set_source_text(&mut self, layer: EntityId, text: String) -> Result<Commit> {
        let snapshot = self.snapshot();
        let content = Self::text_content(&snapshot, layer)?;
        let language = snapshot
            .component::<SceneSourceText>(content)?
            .and_then(|source| source.language);
        let patch = snapshot.patch(|edit| {
            edit.promote_entity_to_user(layer)?;
            edit.promote_entity_to_user(content)?;
            edit.set(
                content,
                &SceneSourceText {
                    text: Authored::user(text),
                    language,
                },
            )
        })?;
        self.commit(patch).await
    }

    pub async fn set_translation(
        &mut self,
        layer: EntityId,
        text: Option<String>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let content = Self::text_content(&snapshot, layer)?;
        let language = snapshot
            .component::<SceneTranslation>(content)?
            .and_then(|translation| translation.language);
        let patch = snapshot.patch(|edit| {
            edit.promote_entity_to_user(layer)?;
            edit.promote_entity_to_user(content)?;
            match text {
                Some(text) => edit.set(
                    content,
                    &SceneTranslation {
                        text: Authored::user(text),
                        language,
                    },
                ),
                None => edit.remove::<SceneTranslation>(content),
            }
        })?;
        self.commit(patch).await
    }

    pub async fn set_typography(&mut self, updates: Vec<TypographyUpdate>) -> Result<Commit> {
        let snapshot = self.snapshot();
        let updates = updates
            .into_iter()
            .map(|update| {
                let content = Self::text_content(&snapshot, update.layer)?;
                Ok((update, content))
            })
            .collect::<Result<Vec<_>>>()?;
        let patch = snapshot.patch(|edit| {
            for (update, content) in updates {
                edit.promote_entity_to_user(update.layer)?;
                edit.promote_entity_to_user(content)?;
                let typography = SceneTypography {
                    origin: Origin::User,
                    preferred_font: update.typography.preferred_font,
                    font_weight: update.typography.font_weight,
                    font_style: update.typography.font_style,
                    size: update.typography.size.filter(|size| *size > 0.0),
                    auto_fit: update.typography.auto_fit,
                    color: update.typography.color,
                    stroke_color: update.typography.stroke_color,
                    stroke_width: update.typography.stroke_width,
                    alignment: update.typography.alignment,
                    writing_mode: update.typography.writing_mode,
                    extensions: Default::default(),
                };
                // The caller only sees "invalid project: <reason>" from a rejection,
                // which cannot be traced back to what was sent. Keep the payload on
                // the failing branch so a report can be matched to its values.
                if let Err(error) = edit.set(update.layer, &typography) {
                    tracing::warn!(
                        layer = %update.layer,
                        content = %content,
                        preferred_font = ?typography.preferred_font,
                        font_weight = ?typography.font_weight,
                        font_style = ?typography.font_style,
                        size = ?typography.size,
                        auto_fit = typography.auto_fit,
                        stroke_width = ?typography.stroke_width,
                        %error,
                        "rejected typography write"
                    );
                    return Err(error);
                }
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub async fn set_geometry(&mut self, updates: Vec<GeometryUpdate>) -> Result<Commit> {
        let snapshot = self.snapshot();
        let updates = updates
            .into_iter()
            .map(|update| {
                let layout = snapshot
                    .component::<SceneTextLayout>(update.layer)?
                    .context("only text layers can change geometry")?;
                let content = Self::text_content(&snapshot, update.layer)?;
                if update.points.is_none()
                    && snapshot
                        .text_layer(update.layer)?
                        .automatic_target()?
                        .is_none()
                {
                    bail!("only automatically placed text can reset its geometry");
                }
                Ok((update, content, layout))
            })
            .collect::<Result<Vec<_>>>()?;
        let patch = snapshot.patch(|edit| {
            for (update, content, mut layout) in updates {
                edit.promote_entity_to_user(update.layer)?;
                edit.promote_entity_to_user(content)?;
                layout.origin = Origin::User;
                layout.angle_degrees = None;
                edit.set(update.layer, &layout)?;
                match update.points {
                    Some(points) => edit.set(
                        update.layer,
                        &SceneGeometry {
                            origin: Origin::User,
                            points: points
                                .into_iter()
                                .map(|point| ScenePoint {
                                    x: point.x,
                                    y: point.y,
                                })
                                .collect(),
                        },
                    )?,
                    None => edit.remove::<SceneGeometry>(update.layer)?,
                }
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub async fn set_visibility(
        &mut self,
        layers: Vec<EntityId>,
        visible: Option<bool>,
        opacity: Option<f32>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let patch = snapshot.patch(|edit| {
            for layer in layers {
                let mut value =
                    snapshot
                        .component::<SceneVisibility>(layer)?
                        .unwrap_or(SceneVisibility {
                            origin: Origin::User,
                            visible: true,
                            opacity: 1.0,
                        });
                if let Some(visible) = visible {
                    value.visible = visible;
                }
                if let Some(opacity) = opacity {
                    value.opacity = opacity;
                }
                value.origin = Origin::User;
                edit.set(layer, &value)?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    pub async fn delete_layers(&mut self, layers: Vec<EntityId>) -> Result<Commit> {
        let snapshot = self.snapshot();
        let mut expanded = Vec::new();
        for layer in layers {
            if snapshot.component::<SceneTextGroup>(layer)?.is_some() {
                expanded.extend(snapshot.children(layer)?);
            } else {
                expanded.push(layer);
            }
        }
        let layers = Self::unique_roots(&snapshot, expanded)?;
        let mut orphaned_contents = Vec::new();
        for layer in &layers {
            let Some(relation) = snapshot.relation_from::<Presents>(*layer)? else {
                continue;
            };
            let content = relation.value().target;
            if snapshot.relations_to_as::<Presents>(content).count() == 1 {
                orphaned_contents.push(content);
            }
        }
        let patch = snapshot.patch(|edit| {
            for layer in layers {
                if snapshot.page(layer).is_ok() {
                    return Err(koharu_scene::Error::Invalid(
                        "delete pages with delete_pages".to_owned(),
                    ));
                }
                edit.remove_entity(layer, RemovePolicy::Cascade)?;
            }
            for content in orphaned_contents {
                if snapshot.entity(content).is_ok() {
                    edit.remove_entity(content, RemovePolicy::Cascade)?;
                }
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    #[tracing::instrument(level = "info", skip_all, fields(layer = %layer, parent = %parent))]
    pub async fn move_layer(
        &mut self,
        layer: EntityId,
        parent: EntityId,
        index: usize,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let siblings = snapshot
            .children(parent)?
            .map(|candidate| Ok(Self::is_layer(&snapshot, candidate)?.then_some(candidate)))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let at = Self::placement(&siblings, layer, index);
        let patch = snapshot.patch(|edit| {
            edit.promote_entity_to_user(layer)?;
            if snapshot.component::<SceneTextGroup>(parent)?.is_some() {
                edit.promote_entity_to_user(parent)?;
            }
            edit.move_entity(layer, Some(parent), at)
        })?;
        self.commit(patch).await
    }

    /// Moves any transformable layer. `commit_transform` routes every element
    /// the desktop reports as transformable through here, so a shape has to be
    /// accepted alongside a text layer and get the shape treatment (rotation on
    /// `ShapeLayer`, endpoints resynced from the polygon).
    pub async fn set_geometries(
        &mut self,
        geometries: impl IntoIterator<Item = (EntityId, SceneGeometry, f32)>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        enum Target {
            Text {
                content: EntityId,
                layout: SceneTextLayout,
            },
            Shape {
                layer: SceneShapeLayer,
            },
        }
        let geometries = geometries
            .into_iter()
            .map(|(element, geometry, angle_degrees)| {
                let target =
                    if let Some(mut layer) = snapshot.component::<SceneShapeLayer>(element)? {
                        layer.origin = Origin::User;
                        layer.angle_degrees = Some(angle_degrees);
                        Target::Shape { layer }
                    } else {
                        let mut layout = snapshot
                            .component::<SceneTextLayout>(element)?
                            .context("only text and shape layers can change geometry")?;
                        layout.origin = Origin::User;
                        layout.angle_degrees = Some(angle_degrees);
                        let content = Self::text_content(&snapshot, element)?;
                        Target::Text { content, layout }
                    };
                Ok((element, geometry, target))
            })
            .collect::<Result<Vec<_>>>()?;
        let patch = snapshot.patch(|edit| {
            for (element, mut geometry, target) in geometries {
                edit.promote_entity_to_user(element)?;
                geometry.origin = Origin::User;
                match target {
                    Target::Text { content, layout } => {
                        edit.promote_entity_to_user(content)?;
                        edit.set(element, &geometry)?;
                        edit.set(element, &layout)?;
                    }
                    Target::Shape { mut layer } => {
                        layer.geometry.resync_with_polygon(&geometry.points);
                        edit.set(element, &geometry)?;
                        edit.set(element, &layer)?;
                    }
                }
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    /// Creates a user shape layer. The outline polygon is the entity's
    /// `Geometry` and the regenerable parameters live in a separate
    /// `ShapeLayer` on the same entity (decision D8); the entity is placed
    /// before the page's text group so text keeps painting on top.
    pub async fn add_shape(
        &mut self,
        page: EntityId,
        geometry: SceneGeometry,
        shape: ShapeGeometry,
        style: ShapeStyle,
        angle_degrees: f32,
    ) -> Result<(Commit, EntityId)> {
        if !angle_degrees.is_finite() {
            bail!("shape angle must be finite");
        }
        let snapshot = self.snapshot();
        snapshot.page(page)?;
        let name = format!(
            "{} {}",
            Self::shape_kind_name(&shape),
            snapshot
                .children(page)?
                .filter(|entity| {
                    snapshot
                        .component::<SceneShapeLayer>(*entity)
                        .ok()
                        .flatten()
                        .is_some_and(|layer| Self::same_shape_kind(&layer.geometry, &shape))
                })
                .count()
                + 1
        );
        let mut layer = None;
        let patch = snapshot.patch(|edit| {
            let at = snapshot
                .page(page)?
                .text_group()?
                .map_or(At::End, |group| At::Before(group.id()));
            let added = edit.add_entity(page, at)?;
            edit.set(
                added,
                &SceneShapeLayer {
                    origin: Origin::User,
                    name,
                    geometry: shape,
                    style,
                    angle_degrees: Some(angle_degrees),
                    rasterized: false,
                },
            )?;
            let mut geometry = geometry;
            geometry.origin = Origin::User;
            edit.set(added, &geometry)?;
            layer = Some(added);
            Ok(())
        })?;
        Ok((
            self.commit(patch).await?,
            layer.expect("shape layer was added while building the patch"),
        ))
    }

    pub async fn set_shape_style(
        &mut self,
        updates: Vec<(EntityId, ShapeStyle)>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let updates = updates
            .into_iter()
            .map(|(element, style)| {
                let layer = snapshot
                    .component::<SceneShapeLayer>(element)?
                    .context("only shape layers can change style")?;
                Ok((element, style, layer))
            })
            .collect::<Result<Vec<_>>>()?;
        let patch = snapshot.patch(|edit| {
            for (element, style, mut layer) in updates {
                edit.promote_entity_to_user(element)?;
                layer.origin = Origin::User;
                layer.style = style;
                edit.set(element, &layer)?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    /// Moves shapes: the polygon is replaced and the rotation lands on the
    /// `ShapeLayer`, which is what the renderer rebuilds the frame from. This is
    /// the shape-only entry point, so a text-layer element is rejected rather
    /// than silently treated as one.
    pub async fn set_shape_geometries(
        &mut self,
        updates: Vec<(EntityId, SceneGeometry, f32)>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        for (element, _, _) in &updates {
            if snapshot.component::<SceneShapeLayer>(*element)?.is_none() {
                bail!("only shape layers can change geometry");
            }
        }
        self.set_geometries(updates).await
    }

    /// The layer-name stem, mirroring the raster convention.
    fn shape_kind_name(shape: &ShapeGeometry) -> &'static str {
        match shape {
            ShapeGeometry::Rectangle { .. } => "Rectangle",
            ShapeGeometry::Ellipse => "Ellipse",
            ShapeGeometry::Line { .. } => "Line",
            ShapeGeometry::Polygon => "Polygon",
            ShapeGeometry::Freeform => "Freeform",
        }
    }

    /// Name numbering counts only shapes of the same kind, so a page can hold
    /// "Rectangle 1" and "Ellipse 1" at once.
    fn same_shape_kind(left: &ShapeGeometry, right: &ShapeGeometry) -> bool {
        std::mem::discriminant(left) == std::mem::discriminant(right)
    }

    pub async fn apply_raster_stroke(
        &mut self,
        page: EntityId,
        layer: Option<EntityId>,
        mode: RasterStrokeMode,
        color: [u8; 4],
        diameter: f32,
        points: Vec<ScenePoint>,
    ) -> Result<(Commit, EntityId)> {
        if !diameter.is_finite() || diameter <= 0.0 || points.is_empty() {
            bail!("a raster stroke requires a positive diameter and at least one point");
        }
        if points
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite())
        {
            bail!("raster stroke points must be finite");
        }
        let snapshot = self.snapshot();
        let page_value = snapshot.page(page)?.page()?;
        let width = page_value.width.round() as u32;
        let height = page_value.height.round() as u32;
        if width == 0 || height == 0 {
            bail!("page dimensions must be positive");
        }
        let mut raster_layer = None;
        let mut promote_layer = false;
        let mut image = if let Some(layer) = layer {
            if snapshot.parent(layer)? != Some(page) {
                bail!("the raster target must be a layer on the active page");
            }
            let target = snapshot
                .component::<SceneRasterLayer>(layer)?
                .context("the raster target must be a layer on the active page")?;
            promote_layer = target.origin != Origin::User
                || snapshot
                    .component::<EntityOrigin>(layer)?
                    .is_some_and(|origin| origin.origin != Origin::User);
            raster_layer = Some(target);
            match snapshot.asset(layer, &AssetRole::new("source")?)? {
                Some(asset) => {
                    let bytes = snapshot.read_blob(asset.blob).await?;
                    image::load_from_memory(&bytes)?.to_rgba8()
                }
                None => RgbaImage::new(width, height),
            }
        } else {
            if mode == RasterStrokeMode::Erase {
                bail!("eraser requires a raster layer target");
            }
            RgbaImage::new(width, height)
        };
        if image.dimensions() != (width, height) {
            bail!("raster layer dimensions must match the page");
        }
        rasterize_stroke(&mut image, mode, color, diameter, &points);
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image).write_to(&mut bytes, ImageFormat::Png)?;
        let source = AssetRole::new("source")?;
        let name = format!(
            "Paint {}",
            snapshot
                .children(page)?
                .filter(|entity| {
                    snapshot
                        .component::<SceneRasterLayer>(*entity)
                        .ok()
                        .flatten()
                        .is_some_and(|layer| layer.kind == RasterLayerKind::Paint)
                })
                .count()
                + 1
        );
        let mut committed_layer = layer;
        let patch = snapshot.patch(|edit| {
            let layer = if let Some(layer) = committed_layer {
                if promote_layer {
                    edit.promote_entity_to_user(layer)?;
                    let raster = raster_layer
                        .as_mut()
                        .expect("an existing raster target has a layer component");
                    raster.origin = Origin::User;
                    edit.set(layer, raster)?;
                }
                layer
            } else {
                let at = snapshot
                    .page(page)?
                    .text_group()?
                    .map_or(At::End, |group| At::Before(group.id()));
                let layer = edit.add_entity(page, at)?;
                edit.set(
                    layer,
                    &SceneRasterLayer {
                        origin: Origin::User,
                        name,
                        kind: RasterLayerKind::Paint,
                    },
                )?;
                committed_layer = Some(layer);
                layer
            };
            edit.set_asset(
                layer,
                &source,
                AssetInput::new(
                    bytes.into_inner(),
                    "image/png",
                    AssetMetadata {
                        width: Some(width),
                        height: Some(height),
                        attributes: Default::default(),
                    },
                ),
            )
        })?;
        Ok((
            self.commit(patch).await?,
            committed_layer.expect("raster layer was selected or added while building the patch"),
        ))
    }

    pub async fn undo(&mut self) -> Result<Commit> {
        let revisions = self.undo.pop().ok_or_else(|| anyhow!("nothing to undo"))?;
        let commit = match self.session.undo_many(revisions.iter().copied()).await {
            Ok(commit) => commit,
            Err(error) => {
                self.undo.push(revisions);
                return Err(error.into());
            }
        };
        self.redo.push(vec![commit.revision]);
        Ok(commit)
    }

    pub async fn redo(&mut self) -> Result<Commit> {
        let revisions = self.redo.pop().ok_or_else(|| anyhow!("nothing to redo"))?;
        let commit = match self.session.undo_many(revisions.iter().copied()).await {
            Ok(commit) => commit,
            Err(error) => {
                self.redo.push(revisions);
                return Err(error.into());
            }
        };
        self.undo.push(vec![commit.revision]);
        Ok(commit)
    }

    pub fn record(&mut self, revisions: Vec<Revision>) {
        if !revisions.is_empty() {
            self.undo.push(revisions);
            self.redo.clear();
        }
    }

    pub fn record_commit(&mut self, commit: &Commit) {
        if commit.changes.to != commit.changes.from {
            self.record(vec![commit.revision]);
        }
    }

    pub async fn commit_rebased(&mut self, patch: koharu_scene::Patch) -> Result<Option<Commit>> {
        let current = self.snapshot();
        let patch = match patch.rebase_on(&current) {
            Ok(patch) => patch,
            Err(
                error @ (koharu_scene::Error::PatchConflict(_)
                | koharu_scene::Error::EntityNotFound(_)
                | koharu_scene::Error::RelationNotFound(_)),
            ) => {
                tracing::debug!(%error, "pipeline output was superseded by a document edit");
                return Ok(None);
            }
            Err(error) => return Err(error.into()),
        };
        if patch.is_empty() {
            return Ok(None);
        }
        Ok(Some(self.commit(patch).await?))
    }

    async fn commit(&mut self, patch: koharu_scene::Patch) -> Result<Commit> {
        Ok(self.session.commit(patch).await?)
    }

    pub fn page(snapshot: &Snapshot, page: EntityId) -> Result<Page> {
        let value = snapshot.page(page)?.page()?;
        let source = Self::asset_id(snapshot, page, "source")?;
        let mut layers = Vec::new();
        Self::collect_layer_views(snapshot, page, &mut layers)?;
        if let Some(image) = source.clone() {
            layers.insert(
                0,
                Layer::Artwork {
                    id: page,
                    parent: None,
                    geometry: Geometry {
                        points: vec![
                            Point { x: 0.0, y: 0.0 },
                            Point {
                                x: value.width,
                                y: 0.0,
                            },
                            Point {
                                x: value.width,
                                y: value.height,
                            },
                            Point {
                                x: 0.0,
                                y: value.height,
                            },
                        ],
                    },
                    visibility: LayerVisibility {
                        visible: true,
                        opacity: 1.0,
                    },
                    image,
                },
            );
        }
        let regions = snapshot
            .descendants(page)?
            .map(|entity| {
                let id = entity.id();
                snapshot
                    .component::<SceneRegion>(id)?
                    .map(|region| Self::analysis_region(snapshot, id, region))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        Ok(Page {
            id: page,
            label: value.label,
            size: PageSize {
                width: value.width,
                height: value.height,
            },
            layers,
            regions,
        })
    }

    fn layer_view(snapshot: &Snapshot, layer: EntityId) -> Result<Layer> {
        let parent = snapshot.parent(layer)?;
        let visibility = Self::layer_visibility(snapshot, layer)?;
        if let Some(group) = snapshot.component::<SceneGroup>(layer)? {
            return Ok(Layer::Group {
                id: layer,
                parent,
                visibility,
                name: group.name,
                role: snapshot
                    .component::<SceneTextGroup>(layer)?
                    .map(|_| GroupRole::Text),
            });
        }
        if let Some(layout) = snapshot.component::<SceneTextLayout>(layer)? {
            let text_layer = snapshot.text_layer(layer)?;
            let content = text_layer.content()?;
            let source = content.source()?.map(|source| SourceText {
                text: source.text.value,
                language: source.language.map(|language| language.to_string()),
            });
            let translation = content.translation()?.map(|translation| Translation {
                text: translation.text.value,
                language: translation.language.map(|language| language.to_string()),
            });
            let role = content.role()?.map(|role| role.role);
            let source_region = content.source_region()?.map(|region| region.id());
            let automatic_region = text_layer.automatic_target()?.map(|region| region.id());
            let typography = text_layer.typography()?.map(Self::typography_view);
            return Ok(Layer::Text {
                id: layer,
                parent,
                geometry: snapshot
                    .component::<SceneGeometry>(layer)?
                    .map(Self::geometry_view),
                angle_degrees: layout.angle_degrees,
                visibility,
                content: Box::new(TextContent {
                    id: content.id(),
                    source,
                    translation,
                    role,
                    source_region,
                }),
                typography,
                layout: layout.kind,
                automatic_region,
            });
        }
        if let Some(raster) = snapshot.component::<SceneRasterLayer>(layer)? {
            return Ok(Layer::Raster {
                id: layer,
                parent,
                visibility,
                image: Self::asset_id(snapshot, layer, "source")?,
                name: raster.name,
                kind: raster.kind,
            });
        }
        if let Some(shape) = snapshot.component::<SceneShapeLayer>(layer)? {
            return Ok(Layer::Shape {
                id: layer,
                parent,
                geometry: snapshot
                    .component::<SceneGeometry>(layer)?
                    .map(Self::geometry_view),
                angle_degrees: shape.angle_degrees,
                visibility,
                shape: shape.geometry,
                style: shape.style,
                rasterized: shape.rasterized,
            });
        }
        let geometry = snapshot
            .component::<SceneGeometry>(layer)?
            .map(Self::geometry_view)
            .context("image layer has no geometry")?;
        let image = Self::asset_id(snapshot, layer, "source")?
            .context("image layer has no source asset")?;
        Ok(Layer::Image {
            id: layer,
            parent,
            geometry,
            visibility,
            image,
        })
    }

    fn analysis_region(
        snapshot: &Snapshot,
        id: EntityId,
        region: SceneRegion,
    ) -> Result<AnalysisRegion> {
        let geometry = snapshot
            .component::<SceneGeometry>(id)?
            .map(Self::geometry_view)
            .context("analysis region has no geometry")?;
        Ok(AnalysisRegion {
            id,
            parent: snapshot.parent(id)?,
            geometry,
            kind: region.kind.as_str().to_owned(),
            label: region.label,
        })
    }

    fn layer_visibility(snapshot: &Snapshot, layer: EntityId) -> Result<LayerVisibility> {
        Ok(snapshot.component::<SceneVisibility>(layer)?.map_or(
            LayerVisibility {
                visible: true,
                opacity: 1.0,
            },
            |visibility| LayerVisibility {
                visible: visibility.visible,
                opacity: visibility.opacity,
            },
        ))
    }

    fn typography_view(typography: SceneTypography) -> Typography {
        let writing_mode = if matches!(&typography.origin, Origin::User) {
            typography.writing_mode
        } else {
            None
        };
        Typography {
            preferred_font: typography.preferred_font,
            font_weight: typography.font_weight,
            font_style: typography.font_style,
            size: typography.size,
            auto_fit: typography.auto_fit,
            color: typography.color,
            stroke_color: typography.stroke_color,
            stroke_width: typography.stroke_width,
            alignment: typography.alignment,
            writing_mode,
        }
    }

    fn geometry_view(geometry: SceneGeometry) -> Geometry {
        Geometry {
            points: geometry
                .points
                .into_iter()
                .map(|point| Point {
                    x: point.x,
                    y: point.y,
                })
                .collect(),
        }
    }

    fn asset_id(snapshot: &Snapshot, entity: EntityId, role: &str) -> Result<Option<String>> {
        Ok(snapshot
            .asset(entity, &AssetRole::new(role)?)?
            .map(|asset| asset.blob.to_string()))
    }

    fn is_layer(snapshot: &Snapshot, entity: EntityId) -> Result<bool> {
        Ok(snapshot.component::<SceneGroup>(entity)?.is_some()
            || snapshot.component::<SceneTextLayout>(entity)?.is_some()
            || snapshot.component::<SceneRasterLayer>(entity)?.is_some()
            || snapshot.component::<SceneShapeLayer>(entity)?.is_some()
            || (snapshot.component::<SceneGeometry>(entity)?.is_some()
                && snapshot
                    .asset(entity, &AssetRole::new("source")?)?
                    .is_some()))
    }

    fn is_content_layer(snapshot: &Snapshot, entity: EntityId) -> Result<bool> {
        Ok(
            snapshot.component::<SceneGroup>(entity)?.is_none()
                && Self::is_layer(snapshot, entity)?,
        )
    }

    fn collect_layer_views(
        snapshot: &Snapshot,
        parent: EntityId,
        layers: &mut Vec<Layer>,
    ) -> Result<()> {
        for child in snapshot.children(parent)? {
            if !Self::is_layer(snapshot, child)? {
                continue;
            }
            let group = snapshot.component::<SceneGroup>(child)?.is_some();
            layers.push(Self::layer_view(snapshot, child)?);
            if group {
                Self::collect_layer_views(snapshot, child, layers)?;
            }
        }
        Ok(())
    }

    fn text_content(snapshot: &Snapshot, layer: EntityId) -> Result<EntityId> {
        snapshot
            .text_layer(layer)?
            .content()
            .map(|content| content.id())
            .map_err(Into::into)
    }

    fn placement(siblings: &[EntityId], moving: EntityId, index: usize) -> At {
        siblings
            .iter()
            .copied()
            .filter(|entity| *entity != moving)
            .nth(index)
            .map_or(At::End, At::Before)
    }

    fn unique_roots(snapshot: &Snapshot, entities: Vec<EntityId>) -> Result<Vec<EntityId>> {
        let selected = entities.into_iter().collect::<HashSet<_>>();
        let mut roots = Vec::new();
        for entity in selected.iter().copied() {
            let mut parent = snapshot.parent(entity)?;
            let mut nested = false;
            while let Some(value) = parent {
                if selected.contains(&value) {
                    nested = true;
                    break;
                }
                parent = snapshot.parent(value)?;
            }
            if !nested {
                roots.push(entity);
            }
        }
        roots.sort_unstable();
        Ok(roots)
    }

    fn geometry_from_frame(frame: Frame) -> Result<SceneGeometry> {
        if !frame.x.is_finite()
            || !frame.y.is_finite()
            || !frame.width.is_finite()
            || !frame.height.is_finite()
            || !frame.angle_degrees.is_finite()
            || frame.width <= 0.0
            || frame.height <= 0.0
        {
            bail!("frame must contain finite coordinates and positive dimensions");
        }
        let center_x = f64::from(frame.x + frame.width * 0.5);
        let center_y = f64::from(frame.y + frame.height * 0.5);
        let half_width = f64::from(frame.width) * 0.5;
        let half_height = f64::from(frame.height) * 0.5;
        let (sin, cos) = f64::from(frame.angle_degrees).to_radians().sin_cos();
        Ok(SceneGeometry {
            origin: Origin::User,
            points: [
                (-half_width, -half_height),
                (half_width, -half_height),
                (half_width, half_height),
                (-half_width, half_height),
            ]
            .map(|(x, y)| ScenePoint {
                x: center_x + x * cos - y * sin,
                y: center_y + x * sin + y * cos,
            })
            .into(),
        })
    }

    /// A two-point drag is a rectangle; three or more points are a freeform
    /// polygon. Both become a single closed polygon the OCR stage can crop.
    fn geometry_from_points(points: Vec<Point>) -> Result<SceneGeometry> {
        if points
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite())
        {
            bail!("region points must be finite");
        }
        let points = match points.as_slice() {
            [start, end] => {
                let (left, right) = (start.x.min(end.x), start.x.max(end.x));
                let (top, bottom) = (start.y.min(end.y), start.y.max(end.y));
                if right - left <= 0.0 || bottom - top <= 0.0 {
                    bail!("a manual region must enclose a non-empty area");
                }
                vec![
                    ScenePoint { x: left, y: top },
                    ScenePoint { x: right, y: top },
                    ScenePoint {
                        x: right,
                        y: bottom,
                    },
                    ScenePoint { x: left, y: bottom },
                ]
            }
            points if points.len() >= 3 => points
                .iter()
                .map(|point| ScenePoint {
                    x: point.x,
                    y: point.y,
                })
                .collect(),
            _ => bail!("a manual region needs two corners or at least three points"),
        };
        Ok(SceneGeometry {
            origin: Origin::User,
            points,
        })
    }
}

fn validate_project_name(name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        bail!("project name cannot be empty");
    }
    if name.ends_with(['.', ' '])
        || name
            .chars()
            .any(|character| character.is_control() || r#"<>:"/\|?*"#.contains(character))
    {
        bail!("project name contains characters that cannot be used in a file name");
    }
    let stem = name.split('.').next().unwrap_or(name).to_ascii_uppercase();
    if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || stem
            .strip_prefix("COM")
            .or_else(|| stem.strip_prefix("LPT"))
            .is_some_and(|number| {
                matches!(number, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
            })
    {
        bail!("project name is reserved by Windows");
    }
    Ok(name.to_owned())
}

/// Upper bound for a composed spread. Matches the renderer surface limit so a
/// merged page is always renderable.
const MAX_MERGE_DIMENSION: u32 = 32_768;

/// Composed spread PNG plus each cleanup/paint layer's re-padded PNG, keyed by
/// layer.
type MergedSpread = (Arc<[u8]>, HashMap<EntityId, Arc<[u8]>>);

/// Paints the given images into one transparent canvas at the supplied horizontal
/// offsets, top aligned.
fn compose_row<'a>(
    fragments: impl IntoIterator<Item = (i64, &'a RgbaImage)>,
    width: u32,
    height: u32,
) -> RgbaImage {
    let mut canvas = RgbaImage::new(width, height);
    for (offset, image) in fragments {
        image::imageops::overlay(&mut canvas, image, offset, 0);
    }
    canvas
}

fn encode_png(image: RgbaImage) -> Result<Arc<[u8]>> {
    let mut encoded = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image)
        .write_to(&mut encoded, ImageFormat::Png)
        .context("failed to encode a merged page image")?;
    Ok(Arc::from(encoded.into_inner()))
}

fn rasterize_stroke(
    image: &mut RgbaImage,
    mode: RasterStrokeMode,
    color: [u8; 4],
    diameter: f32,
    points: &[ScenePoint],
) {
    let radius = f64::from(diameter) * 0.5;
    for (start, end) in points
        .iter()
        .zip(points.iter().skip(1))
        .chain(points.last().map(|point| (point, point)))
    {
        let left = (start.x.min(end.x) - radius - 0.5).floor().max(0.0) as u32;
        let top = (start.y.min(end.y) - radius - 0.5).floor().max(0.0) as u32;
        let right = (start.x.max(end.x) + radius + 0.5)
            .ceil()
            .min(f64::from(image.width())) as u32;
        let bottom = (start.y.max(end.y) + radius + 0.5)
            .ceil()
            .min(f64::from(image.height())) as u32;
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let length_squared = dx * dx + dy * dy;
        for y in top..bottom {
            for x in left..right {
                let px = f64::from(x) + 0.5;
                let py = f64::from(y) + 0.5;
                let t = if length_squared == 0.0 {
                    0.0
                } else {
                    (((px - start.x) * dx + (py - start.y) * dy) / length_squared).clamp(0.0, 1.0)
                };
                let distance =
                    ((px - (start.x + t * dx)).powi(2) + (py - (start.y + t * dy)).powi(2)).sqrt();
                let coverage = (radius + 0.5 - distance).clamp(0.0, 1.0) as f32;
                if coverage == 0.0 {
                    continue;
                }
                let pixel = image.get_pixel_mut(x, y);
                match mode {
                    RasterStrokeMode::Paint => {
                        let source_alpha = f32::from(color[3]) / 255.0 * coverage;
                        let destination_alpha = f32::from(pixel[3]) / 255.0;
                        let output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha);
                        if output_alpha > 0.0 {
                            for channel in 0..3 {
                                let source = f32::from(color[channel]) / 255.0;
                                let destination = f32::from(pixel[channel]) / 255.0;
                                pixel[channel] = (((source * source_alpha
                                    + destination * destination_alpha * (1.0 - source_alpha))
                                    / output_alpha)
                                    * 255.0)
                                    .round() as u8;
                            }
                        }
                        pixel[3] = (output_alpha * 255.0).round() as u8;
                    }
                    RasterStrokeMode::Erase => {
                        pixel[3] = (f32::from(pixel[3]) * (1.0 - coverage)).round() as u8;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use koharu_scene::{
        ArrowHead, Generation, LineCap, LineJoin, ProducerId, ShapeFill, ShapeGeometry, ShapeStyle,
        StrokeStyle, WritingMode,
    };

    use super::*;

    /// `Layer` carries its id inside each variant, so tests need this to pick
    /// one out of a page view.
    fn layer_id(layer: &Layer) -> EntityId {
        match layer {
            Layer::Group { id, .. }
            | Layer::Text { id, .. }
            | Layer::Raster { id, .. }
            | Layer::Shape { id, .. }
            | Layer::Image { id, .. }
            | Layer::Artwork { id, .. } => *id,
        }
    }

    #[tokio::test]
    async fn merging_adjacent_pages_composes_a_wide_spread_and_keeps_layers() {
        let mut session = Session::memory().await.unwrap();
        let source = AssetRole::new("source").unwrap();
        let mut setup = session.snapshot().edit();
        let left = setup
            .add_page(PageDraft::new("left", 2.0, 3.0), At::End)
            .unwrap();
        let right = setup
            .add_page(PageDraft::new("right", 4.0, 3.0), At::End)
            .unwrap();
        for (page, width, height) in [(left, 2u32, 3u32), (right, 4u32, 3u32)] {
            let image = RgbaImage::from_pixel(width, height, image::Rgba([0, 0, 0, 255]));
            let mut encoded = Cursor::new(Vec::new());
            DynamicImage::ImageRgba8(image)
                .write_to(&mut encoded, ImageFormat::Png)
                .unwrap();
            setup
                .set_asset(
                    page,
                    &source,
                    AssetInput::new(
                        Arc::<[u8]>::from(encoded.into_inner()),
                        "image/png",
                        AssetMetadata {
                            width: Some(width),
                            height: Some(height),
                            attributes: Default::default(),
                        },
                    ),
                )
                .unwrap();
        }

        let layout = SceneTextLayout {
            origin: Origin::User,
            kind: TextLayoutKind::Point,
            angle_degrees: None,
        };
        let left_content = setup.add_text_content(left, At::End).unwrap();
        setup
            .add_text_layer(left, At::End, left_content, &layout)
            .unwrap();
        let right_content = setup.add_text_content(right, At::End).unwrap();
        let right_layer = setup
            .add_text_layer(right, At::End, right_content, &layout)
            .unwrap();
        let region = setup
            .add_analysis_region::<koharu_scene::PanelRegion>(
                right,
                At::End,
                &SceneGeometry::rectangle(0.0, 0.0, 2.0, 2.0),
                None,
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let (commit, merged) = project.merge_pages(vec![left, right]).await.unwrap();
        project.record_commit(&commit);

        let snapshot = project.snapshot();
        let view = snapshot.page(merged).unwrap().page().unwrap();
        assert_eq!((view.width, view.height), (6.0, 3.0));
        assert_eq!(Project::pages(&snapshot).unwrap().len(), 1);
        // Text layers from both pages survive under the single merged text group.
        assert_eq!(
            snapshot.entities_with::<SceneTextLayout>().unwrap().count(),
            2
        );
        let group = snapshot
            .page(merged)
            .unwrap()
            .text_group()
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.parent(right_layer).unwrap(), Some(group.id()));
        // The right page's region is translated by the left page's width.
        let region = snapshot
            .component::<SceneGeometry>(region)
            .unwrap()
            .unwrap();
        assert_eq!(region.points[0].x, 2.0);

        // Undo restores the two source pages.
        project.undo().await.unwrap();
        assert_eq!(Project::pages(&project.snapshot()).unwrap().len(), 2);
    }

    #[test]
    fn only_user_authored_direction_is_projected_as_an_override() {
        let mut typography = SceneTypography {
            origin: Origin::Generated(Generation::new(
                ProducerId::new("dev.koharu.pipeline.detection").expect("valid producer"),
            )),
            preferred_font: None,
            font_weight: None,
            font_style: None,
            size: None,
            auto_fit: true,
            color: None,
            stroke_color: None,
            stroke_width: None,
            alignment: None,
            writing_mode: Some(WritingMode::Vertical),
            extensions: Default::default(),
        };

        assert_eq!(
            Project::typography_view(typography.clone()).writing_mode,
            None
        );
        typography.origin = Origin::User;
        assert_eq!(
            Project::typography_view(typography).writing_mode,
            Some(WritingMode::Vertical)
        );
    }

    #[tokio::test]
    async fn chapter_start_markers_are_projected_and_survive_page_renames() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("chapter start", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        project.set_page_chapter_start(page, true).await.unwrap();
        assert!(
            Project::pages(&project.snapshot())
                .unwrap()
                .first()
                .unwrap()
                .chapter_start
        );

        project
            .rename_page(page, "renamed chapter".to_owned())
            .await
            .unwrap();
        let summary = Project::pages(&project.snapshot()).unwrap();
        assert_eq!(summary[0].label, "renamed chapter");
        assert!(summary[0].chapter_start);
    }

    #[tokio::test]
    async fn moving_pages_uses_the_destination_index_after_removing_the_source() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let first = setup
            .add_page(PageDraft::new("first", 100.0, 100.0), At::End)
            .unwrap();
        setup
            .add_page(PageDraft::new("middle", 100.0, 100.0), At::End)
            .unwrap();
        let last = setup
            .add_page(PageDraft::new("last", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        project.move_page(first, 2).await.unwrap();
        assert_eq!(
            Project::pages(&project.snapshot())
                .unwrap()
                .iter()
                .map(|page| page.label.as_str())
                .collect::<Vec<_>>(),
            ["middle", "last", "first"]
        );

        project.move_page(last, 0).await.unwrap();
        assert_eq!(
            Project::pages(&project.snapshot())
                .unwrap()
                .iter()
                .map(|page| page.label.as_str())
                .collect::<Vec<_>>(),
            ["last", "middle", "first"]
        );
    }

    #[tokio::test]
    async fn pastes_text_layer_content_and_placement_to_another_page() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let source_page = setup
            .add_page(PageDraft::new("source", 100.0, 100.0), At::End)
            .unwrap();
        let target_page = setup
            .add_page(PageDraft::new("target", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let (_, source_layer) = project
            .add_text_box(
                source_page,
                Frame {
                    x: 12.0,
                    y: 24.0,
                    width: 32.0,
                    height: 18.0,
                    angle_degrees: 0.0,
                },
            )
            .await
            .unwrap();
        project
            .set_source_text(source_layer, "BOOM".to_owned())
            .await
            .unwrap();
        project
            .set_translation(source_layer, Some("Bang!".to_owned()))
            .await
            .unwrap();

        let (_, pasted) = project
            .paste_text_layers(target_page, vec![source_layer])
            .await
            .unwrap();
        let target = Project::page(&project.snapshot(), target_page).unwrap();
        let Some(Layer::Text {
            id,
            geometry,
            content,
            ..
        }) = target
            .layers
            .iter()
            .find(|layer| matches!(layer, Layer::Text { .. }))
        else {
            panic!("expected a pasted text layer");
        };
        assert_eq!(pasted, vec![*id]);
        assert_ne!(*id, source_layer);
        assert_eq!(
            content.source.as_ref().map(|source| source.text.as_str()),
            Some("BOOM")
        );
        assert_eq!(
            content
                .translation
                .as_ref()
                .map(|translation| translation.text.as_str()),
            Some("Bang!")
        );
        assert_eq!(
            geometry.as_ref().map(|geometry| geometry.points[0].x),
            Some(12.0)
        );
        assert_eq!(
            geometry.as_ref().map(|geometry| geometry.points[0].y),
            Some(24.0)
        );
        assert!(matches!(
            target.layers.first(),
            Some(Layer::Group {
                role: Some(GroupRole::Text),
                ..
            })
        ));
    }

    #[tokio::test]
    async fn pipeline_commit_rebases_or_yields_to_the_manual_edit() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let pipeline_page = setup
            .add_page(PageDraft::new("pipeline", 100.0, 100.0), At::End)
            .unwrap();
        let manual_page = setup
            .add_page(PageDraft::new("manual", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let base = project.snapshot();
        let pipeline = base
            .patch(|edit| {
                edit.set_page(
                    pipeline_page,
                    PageDraft::new("pipeline result", 100.0, 100.0),
                )
            })
            .unwrap();
        let manual = base
            .patch(|edit| edit.set_page(manual_page, PageDraft::new("manual edit", 100.0, 100.0)))
            .unwrap();
        project.commit(manual).await.unwrap();

        let commit = project.commit_rebased(pipeline).await.unwrap().unwrap();
        assert_eq!(
            commit
                .snapshot
                .page(pipeline_page)
                .unwrap()
                .page()
                .unwrap()
                .label,
            "pipeline result"
        );
        assert_eq!(
            commit
                .snapshot
                .page(manual_page)
                .unwrap()
                .page()
                .unwrap()
                .label,
            "manual edit"
        );

        let base = project.snapshot();
        let pipeline = base
            .patch(|edit| {
                edit.set_page(
                    pipeline_page,
                    PageDraft::new("stale pipeline", 100.0, 100.0),
                )
            })
            .unwrap();
        let manual = base
            .patch(|edit| {
                edit.set_page(pipeline_page, PageDraft::new("latest manual", 100.0, 100.0))
            })
            .unwrap();
        project.commit(manual).await.unwrap();

        assert!(project.commit_rebased(pipeline).await.unwrap().is_none());
        assert_eq!(
            project
                .snapshot()
                .page(pipeline_page)
                .unwrap()
                .page()
                .unwrap()
                .label,
            "latest manual"
        );
    }

    #[tokio::test]
    async fn a_manual_region_is_an_ocr_ready_text_region() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        // Two corners become a rectangle; the points are sorted into a polygon.
        let (commit, region, layer) = project
            .add_manual_text_region(
                page,
                vec![Point { x: 10.0, y: 20.0 }, Point { x: 40.0, y: 60.0 }],
            )
            .await
            .unwrap();
        let snapshot = commit.snapshot;
        let stored = snapshot.component::<SceneRegion>(region).unwrap().unwrap();
        assert_eq!(stored.kind, TextRegion::kind());
        assert!(matches!(stored.origin, Origin::User));
        let geometry = snapshot
            .component::<SceneGeometry>(region)
            .unwrap()
            .unwrap();
        assert_eq!(
            geometry.points,
            vec![
                ScenePoint { x: 10.0, y: 20.0 },
                ScenePoint { x: 40.0, y: 20.0 },
                ScenePoint { x: 40.0, y: 60.0 },
                ScenePoint { x: 10.0, y: 60.0 },
            ]
        );

        // The content is linked to its region so the OCR stage can find and fill it.
        let content = snapshot.text_layer(layer).unwrap().content().unwrap().id();
        assert!(
            snapshot
                .relations_to_as::<RecognizedFrom>(region)
                .any(|relation| relation.value().source == content)
        );
        assert!(
            snapshot
                .component::<SceneSourceText>(content)
                .unwrap()
                .is_none()
        );

        // A freeform drag keeps every vertex instead of reducing it to a box.
        let (_, polygon, _) = project
            .add_manual_text_region(
                page,
                vec![
                    Point { x: 5.0, y: 5.0 },
                    Point { x: 30.0, y: 5.0 },
                    Point { x: 15.0, y: 25.0 },
                ],
            )
            .await
            .unwrap();
        assert_eq!(
            project
                .snapshot()
                .component::<SceneGeometry>(polygon)
                .unwrap()
                .unwrap()
                .points
                .len(),
            3
        );
    }

    #[test]
    fn raster_strokes_are_continuous_and_erasable() {
        let mut image = RgbaImage::new(32, 16);
        let points = [
            ScenePoint { x: 4.0, y: 8.0 },
            ScenePoint { x: 28.0, y: 8.0 },
        ];
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [210, 40, 20, 255],
            5.0,
            &points,
        );
        for x in 4..28 {
            assert_eq!(image.get_pixel(x, 8).0, [210, 40, 20, 255]);
        }

        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Erase,
            [0, 0, 0, 0],
            5.0,
            &[ScenePoint { x: 16.0, y: 8.0 }],
        );
        assert_eq!(image.get_pixel(16, 8)[3], 0);
        assert_eq!(image.get_pixel(4, 8)[3], 255);

        let mut white = RgbaImage::new(4, 4);
        rasterize_stroke(
            &mut white,
            RasterStrokeMode::Paint,
            [255, 255, 255, 255],
            3.0,
            &[ScenePoint { x: 2.0, y: 2.0 }],
        );
        assert_eq!(white.get_pixel(2, 2).0, [255, 255, 255, 255]);
    }

    #[tokio::test]
    async fn a_typography_write_that_drops_the_size_is_rejected_by_name() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());
        let (_, _, layer) = project
            .add_manual_text_region(
                page,
                vec![Point { x: 10.0, y: 20.0 }, Point { x: 40.0, y: 60.0 }],
            )
            .await
            .unwrap();

        // Auto-fit off without a size is the payload the scene rejects, and the
        // rejection has to name that intent: it is what reaches the desktop toast.
        let error = project
            .set_typography(vec![TypographyUpdate {
                layer,
                typography: Typography {
                    preferred_font: Some("GyeonggiTitle".to_owned()),
                    font_weight: Some(500),
                    font_style: None,
                    size: None,
                    auto_fit: false,
                    color: None,
                    stroke_color: None,
                    stroke_width: None,
                    alignment: None,
                    writing_mode: None,
                },
            }])
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("auto-fit off requires an explicit font size"),
            "{error}"
        );

        // The stored intent is unchanged, so the same layer still accepts a sized one.
        project
            .set_typography(vec![TypographyUpdate {
                layer,
                typography: Typography {
                    preferred_font: Some("GyeonggiTitle".to_owned()),
                    font_weight: Some(500),
                    font_style: None,
                    size: Some(64.0),
                    auto_fit: false,
                    color: None,
                    stroke_color: None,
                    stroke_width: None,
                    alignment: None,
                    writing_mode: None,
                },
            }])
            .await
            .unwrap();
    }

    /// `commit_transform` moves every transformable layer through
    /// `set_geometries`, so that path has to accept shapes too — otherwise
    /// dragging a shape fails outright. It must resync a line the same way
    /// `set_shape_geometries` does.
    #[tokio::test]
    async fn a_line_moves_through_the_general_transform_path() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let (_, element) = project
            .add_shape(
                page,
                SceneGeometry {
                    origin: Origin::User,
                    points: vec![
                        ScenePoint { x: 0.0, y: 0.0 },
                        ScenePoint { x: 5.0, y: 5.0 },
                        ScenePoint { x: 10.0, y: 10.0 },
                    ],
                },
                ShapeGeometry::Line {
                    start: [0.0, 0.0],
                    end: [10.0, 10.0],
                    head: None,
                },
                ShapeStyle {
                    fill: None,
                    stroke: None,
                },
                0.0,
            )
            .await
            .unwrap();

        let moved = SceneGeometry {
            origin: Origin::User,
            points: vec![
                ScenePoint { x: 100.0, y: 50.0 },
                ScenePoint { x: 105.0, y: 55.0 },
                ScenePoint { x: 110.0, y: 60.0 },
            ],
        };
        project
            .set_geometries(vec![(element, moved.clone(), 30.0)])
            .await
            .unwrap();

        let snapshot = project.snapshot();
        let stored = snapshot
            .component::<SceneShapeLayer>(element)
            .unwrap()
            .unwrap();
        assert_eq!(stored.angle_degrees, Some(30.0));
        let ShapeGeometry::Line { start, end, .. } = stored.geometry else {
            panic!("expected a line shape");
        };
        assert_eq!(start, [100.0, 50.0]);
        assert_eq!(end, [110.0, 60.0]);
        assert_eq!(
            snapshot
                .component::<SceneGeometry>(element)
                .unwrap()
                .unwrap()
                .points
                .len(),
            3
        );
    }

    /// The polygon is authoritative for a `Line`; `start`/`end` are derived.
    /// Without the resync the polygon moves, `line_path` keeps drawing the old
    /// endpoints, and the selection box visibly separates from the stroke.
    #[tokio::test]
    async fn moving_a_line_resyncs_its_endpoints_from_the_polygon() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let (_, element) = project
            .add_shape(
                page,
                SceneGeometry {
                    origin: Origin::User,
                    points: vec![
                        ScenePoint { x: 0.0, y: 0.0 },
                        ScenePoint { x: 5.0, y: 5.0 },
                        ScenePoint { x: 10.0, y: 10.0 },
                    ],
                },
                ShapeGeometry::Line {
                    start: [0.0, 0.0],
                    end: [10.0, 10.0],
                    head: None,
                },
                ShapeStyle {
                    fill: None,
                    stroke: None,
                },
                0.0,
            )
            .await
            .unwrap();

        // Translate the polygon by (100, 50); the stored endpoints must follow.
        let moved = SceneGeometry {
            origin: Origin::User,
            points: vec![
                ScenePoint { x: 100.0, y: 50.0 },
                ScenePoint { x: 105.0, y: 55.0 },
                ScenePoint { x: 110.0, y: 60.0 },
            ],
        };
        project
            .set_shape_geometries(vec![(element, moved, 0.0)])
            .await
            .unwrap();

        let stored = project
            .snapshot()
            .component::<SceneShapeLayer>(element)
            .unwrap()
            .unwrap();
        let ShapeGeometry::Line { start, end, .. } = stored.geometry else {
            panic!("expected a line shape");
        };
        assert_eq!(start, [100.0, 50.0]);
        assert_eq!(end, [110.0, 60.0]);
    }

    #[tokio::test]
    async fn a_shape_layer_round_trips_through_the_page_view() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let geometry = SceneGeometry::rectangle(10.0, 20.0, 100.0, 50.0);
        let style = ShapeStyle {
            fill: Some(ShapeFill::Solid([1, 2, 3, 255])),
            stroke: None,
        };

        let (commit, element) = project
            .add_shape(
                page,
                geometry,
                ShapeGeometry::Rectangle { corner_radius: 0.0 },
                style.clone(),
                0.0,
            )
            .await
            .unwrap();
        assert!(commit.revision.get() > 0);

        let view = Project::page(&project.snapshot(), page).unwrap();
        let layer = view
            .layers
            .iter()
            .find(|layer| layer_id(layer) == element)
            .unwrap();
        let Layer::Shape {
            shape,
            style: view_style,
            angle_degrees,
            rasterized,
            ..
        } = layer
        else {
            panic!("expected a shape layer");
        };
        assert!(matches!(shape, ShapeGeometry::Rectangle { .. }));
        assert_eq!(*angle_degrees, Some(0.0));
        assert!(!rasterized);
        assert_eq!(view_style.fill, Some(ShapeFill::Solid([1, 2, 3, 255])));
    }

    #[tokio::test]
    async fn changing_shape_style_keeps_the_geometry() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let (_, element) = project
            .add_shape(
                page,
                SceneGeometry::rectangle(0.0, 0.0, 10.0, 10.0),
                ShapeGeometry::Ellipse,
                ShapeStyle {
                    fill: Some(ShapeFill::Solid([0, 0, 0, 255])),
                    stroke: None,
                },
                0.0,
            )
            .await
            .unwrap();
        let next = ShapeStyle {
            fill: None,
            stroke: None,
        };
        project
            .set_shape_style(vec![(element, next.clone())])
            .await
            .unwrap();

        let view = Project::page(&project.snapshot(), page).unwrap();
        let Layer::Shape { style, shape, .. } = view
            .layers
            .iter()
            .find(|layer| layer_id(layer) == element)
            .unwrap()
        else {
            panic!("expected a shape layer");
        };
        assert_eq!(*style, next);
        assert!(matches!(shape, ShapeGeometry::Ellipse));
    }

    /// The TypeScript client in `packages/bridge/src/protocol.ts` reproduces
    /// these strings by hand, so the wire shape has to be pinned here rather
    /// than inferred from the Rust types. `ShapeFill` is a newtype variant and
    /// `ShapeGeometry` a struct-variant enum, so both use serde's external
    /// tagging: `{"solid":[...]}` and `{"rectangle":{...}}` / `"ellipse"`.
    #[test]
    fn the_shape_wire_shape_matches_the_typescript_client() {
        assert_eq!(
            serde_json::to_string(&ShapeStyle {
                fill: Some(ShapeFill::Solid([1, 2, 3, 255])),
                stroke: None,
            })
            .unwrap(),
            r#"{"fill":{"solid":[1,2,3,255]},"stroke":null}"#
        );
        assert_eq!(
            serde_json::to_string(&ShapeStyle {
                fill: None,
                stroke: Some(StrokeStyle {
                    color: [0, 0, 0, 255],
                    width: 2.0,
                    dash: Some(vec![4.0, 2.0]),
                    cap: LineCap::Round,
                    join: LineJoin::Bevel,
                }),
            })
            .unwrap(),
            r#"{"fill":null,"stroke":{"color":[0,0,0,255],"width":2.0,"dash":[4.0,2.0],"cap":"round","join":"bevel"}}"#
        );
        assert_eq!(
            serde_json::to_string(&ShapeGeometry::Rectangle { corner_radius: 4.0 }).unwrap(),
            r#"{"rectangle":{"corner_radius":4.0}}"#
        );
        assert_eq!(
            serde_json::to_string(&ShapeGeometry::Ellipse).unwrap(),
            r#""ellipse""#
        );
        assert_eq!(
            serde_json::to_string(&ShapeGeometry::Polygon).unwrap(),
            r#""polygon""#
        );
        assert_eq!(
            serde_json::to_string(&ShapeGeometry::Freeform).unwrap(),
            r#""freeform""#
        );
        assert_eq!(
            serde_json::to_string(&ShapeGeometry::Line {
                start: [0.0, 0.0],
                end: [10.0, 0.0],
                head: Some(ArrowHead {
                    length: 4.0,
                    width: 2.0,
                }),
            })
            .unwrap(),
            r#"{"line":{"start":[0.0,0.0],"end":[10.0,0.0],"head":{"length":4.0,"width":2.0}}}"#
        );
    }
}
