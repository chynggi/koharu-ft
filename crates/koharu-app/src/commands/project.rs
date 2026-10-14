use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    path::PathBuf,
    sync::Arc,
};

use anyhow::{Context as _, Result, anyhow, bail};
use image::{DynamicImage, ImageFormat, RgbaImage};
use koharu_desktop::Frame;
use koharu_rasterizer::{BrushRamp, RampAxis};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, Authored, Commit, Edit, EntityId, EntityOrigin,
    FitsTo, Geometry as SceneGeometry, Group as SceneGroup, Origin, PageDraft, PatternSpec,
    Point as ScenePoint, Presents, RasterLayer as SceneRasterLayer, RasterLayerKind,
    RecognizedFrom, Region as SceneRegion, RegionSpec, RemovePolicy, Revision, Session, ShapeFill,
    ShapeGeometry, ShapeLayer as SceneShapeLayer, ShapeStyle, Snapshot,
    SourceText as SceneSourceText, TextGroup as SceneTextGroup, TextLayout as SceneTextLayout,
    TextLayoutKind, TextRegion, TextRole, Translation as SceneTranslation,
    Typography as SceneTypography, Visibility as SceneVisibility,
};
use serde::{Deserialize, Serialize};
use specta::Type;
use tokio::sync::Mutex;
use vello::Scene;

use super::{
    adjust::{ColorAdjust, apply_adjust},
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
        /// Decision D4: a shape identifies itself in the layer list by name,
        /// the way `Raster` does, instead of falling back to "Image N".
        name: String,
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

/// The page an entity ultimately belongs to.
///
/// [`Snapshot::parent`] returns the *immediate* parent and [`Snapshot::page`]
/// only accepts a page id, so a single `parent` call rejects any nested layer
/// with a misleading `EntityNotFound`. Only a page has no parent, so walking up
/// until `parent` returns `None` terminates at the owning page; the caller's
/// `page(entity)` then fails loudly if the chain did not in fact reach one,
/// which is why this helper performs no page check of its own.
fn owning_page(snapshot: &Snapshot, entity: EntityId) -> Result<EntityId> {
    let mut page = entity;
    while let Some(parent) = snapshot.parent(page)? {
        page = parent;
    }
    Ok(page)
}

/// The pixels a raster layer owns, decoded and pinned to the page size.
///
/// A layer that carries no `source` asset — a paint layer a user has not drawn
/// on yet — is an empty page-sized surface rather than an error, so callers can
/// mutate and re-encode it uniformly. A stored image whose dimensions do not
/// match the page is rejected instead of being silently rescaled, because every
/// raster consumer indexes by page coordinates.
async fn read_source_image(
    snapshot: &Snapshot,
    entity: EntityId,
    role: &AssetRole,
    width: u32,
    height: u32,
) -> Result<RgbaImage> {
    let image = match snapshot.asset(entity, role)? {
        Some(asset) => {
            let bytes = snapshot.read_blob(asset.blob).await?;
            image::load_from_memory(&bytes)?.to_rgba8()
        }
        None => RgbaImage::new(width, height),
    };
    if image.dimensions() != (width, height) {
        bail!("raster layer dimensions must match the page");
    }
    Ok(image)
}

/// Whether a user edit must claim both of a raster layer's origin records.
fn raster_layer_needs_promotion(
    snapshot: &Snapshot,
    entity: EntityId,
    raster: &SceneRasterLayer,
) -> Result<bool> {
    Ok(raster.origin != Origin::User
        || snapshot
            .component::<EntityOrigin>(entity)?
            .is_some_and(|origin| origin.origin != Origin::User))
}

/// Encodes a raster surface as the PNG payload of a layer's `source` asset.
fn encode_source_image(image: RgbaImage) -> Result<Vec<u8>> {
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image).write_to(&mut bytes, ImageFormat::Png)?;
    Ok(bytes.into_inner())
}

/// Stores `bytes` as the layer's `source` asset, promoting user ownership first
/// when `promote` is set.
///
/// [`SceneRasterLayer::origin`] is persisted separately from the [`EntityOrigin`]
/// component, and pipeline stages gate on it
/// (`crates/koharu-pipeline/src/stages/inpainting.rs` derives `generated` from
/// both), so a user pixel write has to claim *both* records. Claiming only
/// `EntityOrigin` would leave the layer looking pipeline-owned, and the next
/// stage that reuses such a layer as its overlay would treat the user's pixels
/// as its own generated output. That invariant lives here, next to the write
/// that must uphold it, so no caller can uphold half of it.
///
/// This returns a scene result rather than `anyhow::Result` because it runs
/// inside [`Snapshot::patch`], whose closure is typed by the scene crate; PNG
/// encoding, which is not, stays in [`encode_source_image`].
fn write_source_image(
    edit: &mut Edit,
    entity: EntityId,
    raster: &mut SceneRasterLayer,
    promote: bool,
    role: &AssetRole,
    bytes: Vec<u8>,
    width: u32,
    height: u32,
) -> koharu_scene::Result<()> {
    if promote {
        edit.promote_entity_to_user(entity)?;
        raster.origin = Origin::User;
        edit.set(entity, raster)?;
    }
    edit.set_asset(
        entity,
        role,
        AssetInput::new(
            bytes,
            "image/png",
            AssetMetadata {
                width: Some(width),
                height: Some(height),
                attributes: Default::default(),
            },
        ),
    )
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

        // Cleanup, paint, and rasterized shape layers are full-page pixel
        // layers (`rasterize_shapes` bakes onto the whole page surface), so the
        // renderer stretches them across the spread. Re-pad each to the merged
        // width so its pixels stay in the correct half.
        let mut raster_blobs = Vec::new();
        for (index, &page) in pages.iter().enumerate() {
            for child in snapshot.children(page)? {
                let is_page_pixels = snapshot.component::<SceneRasterLayer>(child)?.is_some()
                    || snapshot
                        .component::<SceneShapeLayer>(child)?
                        .is_some_and(|shape| shape.rasterized);
                if !is_page_pixels {
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
                        // A merge shifts the polygon; a line shape has to take
                        // its derived endpoints along through the same helper
                        // the transform path uses.
                        Self::write_geometry(
                            edit,
                            id,
                            snapshot.component::<SceneShapeLayer>(id)?,
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
                    Target::Shape { layer } => {
                        Self::write_geometry(edit, element, Some(layer), &geometry)?;
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
        let mut added = None;
        let patch = snapshot.patch(|edit| {
            let at = snapshot
                .page(page)?
                .text_group()?
                .map_or(At::End, |group| At::Before(group.id()));
            let entity = edit.add_entity(page, at)?;
            let mut geometry = geometry;
            geometry.origin = Origin::User;
            // The layer is created in this patch and never reaches the
            // snapshot, but the helper takes it by value, so it resyncs the
            // endpoints here exactly as it does on the other write paths.
            Self::write_geometry(
                edit,
                entity,
                Some(SceneShapeLayer {
                    origin: Origin::User,
                    name,
                    geometry: shape,
                    style,
                    angle_degrees: Some(angle_degrees),
                    rasterized: false,
                }),
                &geometry,
            )?;
            added = Some(entity);
            Ok(())
        })?;
        Ok((
            self.commit(patch).await?,
            added.expect("shape layer was added while building the patch"),
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

    /// Sets or clears the image pattern of a shape layer's fill.
    ///
    /// `None` restores [`ShapeFill::Solid`], keeping the colour the user already
    /// sees: a solid fill keeps its own colour and every other fill falls back to
    /// [`PatternSpec::fallback`], which is the colour a pattern draws when its
    /// asset is missing (P-D6). So switching a pattern off never changes what is
    /// on screen.
    ///
    /// The pattern's image is *not* written here — [`Project::set_shape_pattern_asset`]
    /// owns that. `PatternSpec` references no asset (P-D2), so this method only
    /// records how the owning layer's `pattern` asset should repeat.
    pub async fn set_shape_pattern(
        &mut self,
        layer: EntityId,
        pattern: Option<PatternSpec>,
    ) -> Result<(Commit, EntityId)> {
        if let Some(spec) = &pattern {
            spec.validate()?;
        }
        let snapshot = self.snapshot();
        let mut shape = snapshot
            .component::<SceneShapeLayer>(layer)?
            .context("only shape layers can carry a pattern")?;
        let fill = match pattern {
            Some(spec) => ShapeFill::Pattern(spec),
            None => ShapeFill::Solid(match shape.style.fill {
                Some(ShapeFill::Solid(color)) => color,
                Some(ShapeFill::Pattern(spec)) => spec.fallback,
                // A gradient has no colour of its own to keep and an absent fill
                // has none at all, so both take the white a fresh pattern starts
                // with — the same default `defaultPatternSpec` uses in the editor.
                Some(ShapeFill::Gradient(_)) | None => [255, 255, 255, 255],
            }),
        };
        let patch = snapshot.patch(|edit| {
            edit.promote_entity_to_user(layer)?;
            shape.origin = Origin::User;
            shape.style.fill = Some(fill);
            edit.set(layer, &shape)?;
            Ok(())
        })?;
        Ok((self.commit(patch).await?, layer))
    }

    /// Writes the `pattern` asset a shape layer's pattern fill samples.
    ///
    /// Only a shape layer may own this role. A raster layer does not carry a
    /// pattern (P-D2): the shape layer owns the asset so the two live and die
    /// together, and accepting the write on a raster layer would create a
    /// second owner that no fill ever reads.
    ///
    /// The bytes are decoded here rather than at read time because the stored
    /// [`AssetMetadata`] dimensions are what the renderer uploads; a caller that
    /// supplied dimensions the pixels do not have would produce a brush that
    /// disagrees with its own texture.
    pub async fn set_shape_pattern_asset(
        &mut self,
        layer: EntityId,
        bytes: Vec<u8>,
        media_type: String,
    ) -> Result<(Commit, EntityId)> {
        let snapshot = self.snapshot();
        snapshot
            .component::<SceneShapeLayer>(layer)?
            .context("only shape layers can carry a pattern")?;
        let dimensions = image::load_from_memory(&bytes)
            .context("the pattern asset is not a decodable image")?
            .into_rgba8()
            .dimensions();
        // A zero-sized image is a valid PNG but not a usable pattern: the brush
        // would divide by its extents. The renderer draws `fallback` for it, so
        // refusing the write here keeps the stored asset always drawable.
        if dimensions.0 == 0 || dimensions.1 == 0 {
            bail!("the pattern asset must have positive dimensions");
        }
        let role = AssetRole::new("pattern")?;
        let patch = snapshot.patch(|edit| {
            edit.set_asset(
                layer,
                &role,
                AssetInput::new(
                    bytes,
                    media_type,
                    AssetMetadata {
                        width: Some(dimensions.0),
                        height: Some(dimensions.1),
                        attributes: Default::default(),
                    },
                ),
            )
        })?;
        Ok((self.commit(patch).await?, layer))
    }

    /// Writes a polygon and keeps a `ShapeLayer`'s derived values in step with it.
    ///
    /// The polygon is authoritative and `Line`'s `start`/`end` are derived from
    /// its first and last points, while `line_path` strokes the endpoints and
    /// `element_frame` reads the polygon. Every path that writes a polygon for a
    /// shape must therefore come through here, or a moved line's selection box
    /// separates from its stroke. The callers are `add_shape` (the creation
    /// path), `set_geometries` (the transform path), and `merge_pages` (the
    /// spread path); all three are covered by a regression test. A new write
    /// path must call this too.
    fn write_geometry(
        edit: &mut koharu_scene::Edit,
        entity: EntityId,
        mut layer: Option<SceneShapeLayer>,
        geometry: &SceneGeometry,
    ) -> koharu_scene::Result<()> {
        if let Some(layer) = layer.as_mut() {
            layer.geometry.resync_with_polygon(&geometry.points);
        }
        edit.set(entity, geometry)?;
        if let Some(layer) = layer {
            edit.set(entity, &layer)?;
        }
        Ok(())
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

    /// Bakes shapes into pixels and marks them rasterized.
    ///
    /// This is a one-way action: the baked asset becomes authoritative for
    /// rendering (`rasterized == true` makes the renderer consume the raster),
    /// so the shape can no longer be re-rendered from its parameters.
    ///
    /// One scene and one bake per shape, because each shape owns its own
    /// `source` asset: batching several shapes into one surface would hand them
    /// all the same image. Supersampling happens inside the rasterizer, so the
    /// per-shape cost is one page-sized render either way.
    pub async fn rasterize_shapes(
        &mut self,
        page: EntityId,
        shapes: Vec<EntityId>,
    ) -> Result<Commit> {
        let snapshot = self.snapshot();
        let page_value = snapshot.page(page)?.page()?;
        let width = page_value.width.round() as u32;
        let height = page_value.height.round() as u32;
        if width == 0 || height == 0 {
            bail!("page dimensions must be positive");
        }
        let mut targets = Vec::with_capacity(shapes.len());
        let pattern_role = AssetRole::new("pattern")?;
        for shape in shapes {
            if snapshot.parent(shape)? != Some(page) {
                bail!("a rasterized shape must be a layer on the active page");
            }
            let layer = snapshot
                .component::<SceneShapeLayer>(shape)?
                .context("only shape layers can be rasterized")?;
            let geometry = snapshot
                .component::<SceneGeometry>(shape)?
                .context("a shape layer must have an outline geometry")?;
            // A shape cannot be measured without a usable frame; skipping it
            // matches the renderer, which draws nothing for it either.
            let Some(frame) =
                koharu_renderer::shape_renderer::shape_frame(&geometry, layer.angle_degrees)
            else {
                continue;
            };
            // `shape_layer_path` is the single entry point: it delegates
            // `Polygon`/`Freeform` to the entity's points and every other kind
            // to the shape's parameters. Calling `shape_path` directly would
            // bake the two point-backed kinds as an empty path.
            let mut path = koharu_renderer::shape_renderer::shape_layer_path(
                &layer.geometry,
                &geometry.points,
                &layer.style,
                frame,
            );
            // A pattern fill needs its pixels before it can draw. `PatternSpec`
            // references no asset (P-D2), so this is the only place that can
            // resolve the owning layer's `pattern` blob, and without it
            // `append_to` would paint `fallback` — a baked pattern would become a
            // flat colour. Decoding failures are deliberately not fatal here:
            // `append_to` draws the fallback for a missing image (P-D6), and one
            // broken pattern asset must not make rasterizing a page fail.
            if let Some(asset) = snapshot.asset(shape, &pattern_role)? {
                let bytes = snapshot.read_blob(asset.blob).await?;
                match image::load_from_memory(&bytes) {
                    Ok(decoded) => {
                        let decoded = decoded.to_rgba8();
                        let attached = koharu_renderer::shape_renderer::attach_pattern_image(
                            &mut path,
                            koharu_rasterizer::PreparedPatternImage {
                                width: decoded.width(),
                                height: decoded.height(),
                                pixels: Arc::from(decoded.into_raw()),
                            },
                        );
                        if !attached {
                            tracing::warn!(
                                ?shape,
                                "a pattern asset was found but the fill is not a pattern"
                            );
                        }
                    }
                    Err(error) => {
                        tracing::warn!(?shape, %error, "the pattern asset could not be decoded");
                    }
                }
            }
            targets.push((shape, layer, path));
        }
        if targets.is_empty() {
            return self.commit(snapshot.patch(|_| Ok(()))?).await;
        }
        let baked = tokio_rayon::spawn(
            move || -> Result<Vec<(EntityId, SceneShapeLayer, Vec<u8>)>> {
                // `Rasterizer::new` and the readback both block on WGPU, so the
                // whole bake stays off the async executor.
                let rasterizer = koharu_rasterizer::Rasterizer::new()?;
                targets
                    .into_iter()
                    .map(|(entity, layer, path)| {
                        // The scene is drawn through the same Vello entry point
                        // as the live preview (`PreparedPath::append_to`), so a
                        // baked shape cannot look different from the shape the
                        // user dragged. `rasterize_scene` is used rather than a
                        // pixel loop because it supersamples and Lanczos
                        // downsamples that draw for antialiasing.
                        let mut scene = Scene::new();
                        path.append_to(&mut scene);
                        // A transparent background is load-bearing: any other
                        // alpha paints the whole page rectangle over whatever
                        // sits underneath the shape.
                        let image = rasterizer.rasterize_scene(
                            &scene,
                            width,
                            height,
                            [0, 0, 0, 0],
                            koharu_rasterizer::RasterOptions::supersampled(4),
                        )?;
                        let mut bytes = Cursor::new(Vec::new());
                        DynamicImage::ImageRgba8(image).write_to(&mut bytes, ImageFormat::Png)?;
                        Ok((entity, layer, bytes.into_inner()))
                    })
                    .collect()
            },
        )
        .await?;
        let source = AssetRole::new("source")?;
        // The flag and the asset land in one patch, so one commit either writes
        // both or neither. Splitting them would let a crash leave a rasterized
        // shape with no image (it renders nothing) or an unattached image.
        let patch = snapshot.patch(|edit| {
            for (entity, mut layer, png) in baked {
                edit.promote_entity_to_user(entity)?;
                layer.origin = Origin::User;
                layer.rasterized = true;
                edit.set(entity, &layer)?;
                edit.set_asset(
                    entity,
                    &source,
                    AssetInput::new(
                        png,
                        "image/png",
                        AssetMetadata {
                            width: Some(width),
                            height: Some(height),
                            attributes: Default::default(),
                        },
                    ),
                )?;
            }
            Ok(())
        })?;
        self.commit(patch).await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn apply_raster_stroke(
        &mut self,
        page: EntityId,
        layer: Option<EntityId>,
        mode: RasterStrokeMode,
        color: [u8; 4],
        ramp: Option<BrushRamp>,
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
        let source = AssetRole::new("source")?;
        let mut image = if let Some(layer) = layer {
            if snapshot.parent(layer)? != Some(page) {
                bail!("the raster target must be a layer on the active page");
            }
            let target = snapshot
                .component::<SceneRasterLayer>(layer)?
                .context("the raster target must be a layer on the active page")?;
            promote_layer = raster_layer_needs_promotion(&snapshot, layer, &target)?;
            raster_layer = Some(target);
            read_source_image(&snapshot, layer, &source, width, height).await?
        } else {
            if mode == RasterStrokeMode::Erase {
                bail!("eraser requires a raster layer target");
            }
            RgbaImage::new(width, height)
        };
        rasterize_stroke(&mut image, mode, color, ramp, diameter, &points);
        let bytes = encode_source_image(image)?;
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
            if let Some(layer) = committed_layer {
                let raster = raster_layer
                    .as_mut()
                    .expect("an existing raster target has a layer component");
                write_source_image(
                    edit,
                    layer,
                    raster,
                    promote_layer,
                    &source,
                    bytes,
                    width,
                    height,
                )?;
            } else {
                let at = snapshot
                    .page(page)?
                    .text_group()?
                    .map_or(At::End, |group| At::Before(group.id()));
                let layer = edit.add_entity(page, at)?;
                // The creation branch belongs to this method alone: it is the
                // only raster command that can conjure a layer. A freshly added
                // entity already carries `Origin::User`, so the asset write is
                // not a promotion.
                let mut raster = SceneRasterLayer {
                    origin: Origin::User,
                    name,
                    kind: RasterLayerKind::Paint,
                };
                edit.set(layer, &raster)?;
                write_source_image(
                    edit,
                    layer,
                    &mut raster,
                    false,
                    &source,
                    bytes,
                    width,
                    height,
                )?;
                committed_layer = Some(layer);
            }
            Ok(())
        })?;
        Ok((
            self.commit(patch).await?,
            committed_layer.expect("raster layer was selected or added while building the patch"),
        ))
    }

    /// Rewrites an existing raster layer's pixels in place.
    ///
    /// Unlike [`Project::apply_raster_stroke`] this never creates a layer: an
    /// adjustment edits the pixels a layer already owns, so a target without a
    /// [`SceneRasterLayer`] component is an error rather than a new layer. The
    /// adjustment itself is not persisted — the committed result is the pixels.
    pub async fn adjust_raster(
        &mut self,
        layer: EntityId,
        adjust: ColorAdjust,
    ) -> Result<(Commit, EntityId)> {
        // `apply_adjust` documents that its caller must validate first:
        // `Levels { low, high }` with `low >= high` trips the
        // `assert!(input_min < input_max)` inside `stretch_contrast_mut`, and a
        // far-out `Brightness` overflows the kernel's addition. Validating here
        // is the other half of that contract.
        adjust.validate()?;
        let snapshot = self.snapshot();
        let page = owning_page(&snapshot, layer)?;
        let page_value = snapshot.page(page)?.page()?;
        let width = page_value.width.round() as u32;
        let height = page_value.height.round() as u32;
        if width == 0 || height == 0 {
            bail!("page dimensions must be positive");
        }
        // `RasterLayer::origin` is persisted separately from `EntityOrigin`, and
        // the pipeline stages gate on it (`inpainting.rs` computes `generated`
        // from both), so a user adjustment has to claim both. Claiming only
        // `EntityOrigin` would leave the layer looking pipeline-owned, and the
        // next stage that reuses such a layer as its overlay would treat the
        // user's pixels as its own generated output. `write_source_image` owns
        // the promotion step; this comment records why it must happen at all.
        let mut raster_layer = snapshot
            .component::<SceneRasterLayer>(layer)?
            .context("only raster layers can be adjusted")?;
        let promote_layer = raster_layer_needs_promotion(&snapshot, layer, &raster_layer)?;
        let source = AssetRole::new("source")?;
        // `apply_raster_stroke` treats a missing asset as an empty
        // page-sized surface rather than an error; an adjustment over it is
        // then a no-op image that still commits.
        let mut image = read_source_image(&snapshot, layer, &source, width, height).await?;
        apply_adjust(&mut image, adjust);
        let bytes = encode_source_image(image)?;
        let patch = snapshot.patch(|edit| {
            write_source_image(
                edit,
                layer,
                &mut raster_layer,
                promote_layer,
                &source,
                bytes,
                width,
                height,
            )
        })?;
        Ok((self.commit(patch).await?, layer))
    }

    /// Flood-fills the connected region of `layer` under `point` with `color`.
    ///
    /// Only a [`SceneRasterLayer`] owns pixels, so a target without that
    /// component is an error rather than a new layer. The colour is not
    /// persisted: the committed result is the pixels.
    ///
    /// A fill whose start pixel already has `color` commits with no pixel
    /// change. That is deliberate rather than an error — the user asked for the
    /// pixel to be that colour and it is, so there is nothing to report.
    pub async fn fill_raster(
        &mut self,
        layer: EntityId,
        point: Point,
        color: [u8; 4],
    ) -> Result<(Commit, EntityId)> {
        let snapshot = self.snapshot();
        let page = owning_page(&snapshot, layer)?;
        let page_value = snapshot.page(page)?.page()?;
        let width = page_value.width.round() as u32;
        let height = page_value.height.round() as u32;
        if width == 0 || height == 0 {
            bail!("page dimensions must be positive");
        }
        // `RasterLayer::origin` is persisted separately from `EntityOrigin`, and
        // the pipeline stages gate on it (`inpainting.rs` computes `generated`
        // from both), so a user fill has to claim both. Claiming only
        // `EntityOrigin` would leave the layer looking pipeline-owned, and the
        // next stage that reuses such a layer as its overlay would treat the
        // user's pixels as its own generated output. `write_source_image` owns
        // the promotion step; this comment records why it must happen at all.
        let mut raster_layer = snapshot
            .component::<SceneRasterLayer>(layer)?
            .context("only raster layers can be filled")?;
        let promote_layer = raster_layer_needs_promotion(&snapshot, layer, &raster_layer)?;
        // The click point is `f64` page space and `flood_fill_mut` takes `u32`,
        // so both axes are validated before the cast: a negative value would
        // become a huge `u32` and a `NaN` slips past `< 0.0` and `>= width`
        // because every comparison with it is false. `floor` rounds towards the
        // pixel that contains the point, so `2.7` lands on pixel 2.
        let x = point.x.floor();
        let y = point.y.floor();
        if !x.is_finite()
            || !y.is_finite()
            || x < 0.0
            || y < 0.0
            || x >= f64::from(width)
            || y >= f64::from(height)
        {
            bail!("the fill point is outside the page");
        }
        let (x, y) = (x as u32, y as u32);
        let source = AssetRole::new("source")?;
        // `apply_raster_stroke` treats a missing asset as an empty
        // page-sized surface rather than an error; a fill over it is then a
        // region of transparency that still commits.
        let mut image = read_source_image(&snapshot, layer, &source, width, height).await?;
        // `flood_fill_mut` is 4-directional and compares with `PartialEq`, so
        // the region is exactly the set of pixels equal to the start pixel —
        // there is no tolerance to give it, and none is wanted here.
        //
        // It must be skipped when the colour already matches, because its loop
        // terminates only once `inside` turns false and `inside` is "the pixel
        // still equals the start pixel". Writing the same colour back never
        // changes that, so every span re-covers its row and pushes fresh spans
        // to the rows above and below without bound — a hang, not a no-op. A
        // single-row image cannot expose this (the neighbouring rows are out of
        // bounds), so the guard cannot be validated with one. Skipping is
        // semantically identical: nothing would have changed.
        if image.get_pixel(x, y).0 != color {
            imageproc::drawing::flood_fill_mut(&mut image, x, y, image::Rgba(color));
        }
        let bytes = encode_source_image(image)?;
        let patch = snapshot.patch(|edit| {
            write_source_image(
                edit,
                layer,
                &mut raster_layer,
                promote_layer,
                &source,
                bytes,
                width,
                height,
            )
        })?;
        Ok((self.commit(patch).await?, layer))
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
                name: shape.name,
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
    ramp: Option<BrushRamp>,
    diameter: f32,
    points: &[ScenePoint],
) {
    let radius = f64::from(diameter) * 0.5;
    // An `AlongStroke` ramp runs over the whole polyline, so it needs each
    // point's *cumulative* distance: `t` below is only a per-segment projection
    // parameter and restarts at 0 for every segment. A single-point stroke
    // leaves `total` at zero, and the `total > 0.0` guard in the pixel loop
    // turns that into position 0.0 instead of a NaN.
    let mut cumulative = Vec::with_capacity(points.len());
    let mut total = 0.0;
    for (index, point) in points.iter().enumerate() {
        if let Some(previous) = index.checked_sub(1).map(|previous| &points[previous]) {
            total += (point.x - previous.x).hypot(point.y - previous.y);
        }
        cumulative.push(total);
    }
    let page_width = f64::from(image.width());
    let page_height = f64::from(image.height());
    for (index, (start, end)) in points
        .iter()
        .zip(points.iter().skip(1))
        .chain(points.last().map(|point| (point, point)))
        .enumerate()
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
                // The ramp position is page-normalized for `FixedAxis`, so
                // several strokes can share one ramp; a frame or viewport
                // coordinate would make the colour depend on the camera.
                let paint = ramp.map_or(color, |ramp| {
                    let position = match ramp {
                        BrushRamp::AlongStroke { .. } => {
                            if total > 0.0 {
                                (cumulative[index] + t * length_squared.sqrt()) / total
                            } else {
                                0.0
                            }
                        }
                        BrushRamp::FixedAxis { axis, .. } => match axis {
                            RampAxis::Horizontal => px / page_width,
                            RampAxis::Vertical => py / page_height,
                        },
                    };
                    ramp.color_at(position as f32)
                });
                let pixel = image.get_pixel_mut(x, y);
                match mode {
                    RasterStrokeMode::Paint => {
                        let source_alpha = f32::from(paint[3]) / 255.0 * coverage;
                        let destination_alpha = f32::from(pixel[3]) / 255.0;
                        let output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha);
                        if output_alpha > 0.0 {
                            for channel in 0..3 {
                                let source = f32::from(paint[channel]) / 255.0;
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
                    // `Erase` composites by coverage alone and never reads the
                    // colour, so a ramp deliberately does not affect it.
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
        ArrowHead, Generation, GradientExtend, GradientKind, GradientSpec, GradientStop, LineCap,
        LineJoin, PatternRepeat, ProducerId, ShapeFill, ShapeGeometry, ShapeStyle, StrokeStyle,
        WritingMode,
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

    /// A merge shifts every entity's polygon by the page offset. A `Line` draws
    /// from its stored endpoints while `element_frame` reads the polygon, so
    /// shifting only the polygon recreates the defect `set_geometries` resyncs
    /// away: the selection box moves and the drawn stroke stays behind.
    #[tokio::test]
    async fn merging_pages_shifts_a_line_shape_with_its_polygon() {
        let mut session = Session::memory().await.unwrap();
        let source = AssetRole::new("source").unwrap();
        let mut setup = session.snapshot().edit();
        let left = setup
            .add_page(PageDraft::new("left", 20.0, 10.0), At::End)
            .unwrap();
        let right = setup
            .add_page(PageDraft::new("right", 30.0, 10.0), At::End)
            .unwrap();
        for (page, width, height) in [(left, 20u32, 10u32), (right, 30u32, 10u32)] {
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
        let entity = setup.add_entity(right, At::End).unwrap();
        setup
            .set(
                entity,
                &SceneShapeLayer {
                    origin: Origin::User,
                    name: "Line 1".to_owned(),
                    geometry: ShapeGeometry::Line {
                        start: [0.0, 0.0],
                        end: [10.0, 4.0],
                        head: None,
                    },
                    style: ShapeStyle {
                        fill: None,
                        stroke: None,
                    },
                    angle_degrees: Some(0.0),
                    rasterized: false,
                },
            )
            .unwrap();
        setup
            .set(
                entity,
                &SceneGeometry {
                    origin: Origin::User,
                    points: vec![
                        ScenePoint { x: 0.0, y: 0.0 },
                        ScenePoint { x: 5.0, y: 2.0 },
                        ScenePoint { x: 10.0, y: 4.0 },
                    ],
                },
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let (commit, merged) = project.merge_pages(vec![left, right]).await.unwrap();
        project.record_commit(&commit);

        let snapshot = project.snapshot();
        assert_eq!(
            snapshot.page(merged).unwrap().page().unwrap().width,
            50.0,
            "the spread must be as wide as both pages"
        );
        let geometry = snapshot
            .component::<SceneGeometry>(entity)
            .unwrap()
            .unwrap();
        let first = *geometry.points.first().unwrap();
        let last = *geometry.points.last().unwrap();
        assert_eq!(first.x, 20.0, "the right page's polygon must shift left");

        let ShapeGeometry::Line { start, end, .. } = snapshot
            .component::<SceneShapeLayer>(entity)
            .unwrap()
            .unwrap()
            .geometry
        else {
            panic!("expected a line shape");
        };
        assert_eq!(
            start,
            [first.x, first.y],
            "Line.start must follow the polygon"
        );
        assert_eq!(end, [last.x, last.y], "Line.end must follow the polygon");

        // `element_frame` restores its frame from the polygon, so the drawn
        // endpoints have to stay inside that frame.
        let frame = koharu_renderer::shape_renderer::shape_frame(&geometry, Some(0.0)).unwrap();
        for endpoint in [start, end] {
            assert!(
                endpoint[0] >= f64::from(frame.x) - 1e-6
                    && endpoint[0] <= f64::from(frame.x + frame.width) + 1e-6,
                "{endpoint:?} is outside the frame {frame:?}"
            );
            assert!(
                endpoint[1] >= f64::from(frame.y) - 1e-6
                    && endpoint[1] <= f64::from(frame.y + frame.height) + 1e-6,
                "{endpoint:?} is outside the frame {frame:?}"
            );
        }
    }

    /// A rasterized shape's asset is a full page surface and the renderer fits
    /// it to the layer's frame, which the shape branch repoints at the merged
    /// page rectangle. Re-padding it is therefore what keeps its pixels in the
    /// correct half; an old page-width image would be stretched over the spread.
    #[tokio::test]
    async fn merging_pages_repads_a_rasterized_shape_asset() {
        let mut session = Session::memory().await.unwrap();
        let source = AssetRole::new("source").unwrap();
        let mut setup = session.snapshot().edit();
        let left = setup
            .add_page(PageDraft::new("left", 20.0, 10.0), At::End)
            .unwrap();
        let right = setup
            .add_page(PageDraft::new("right", 30.0, 10.0), At::End)
            .unwrap();
        for (page, width, height) in [(left, 20u32, 10u32), (right, 30u32, 10u32)] {
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
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        // A rasterized shape owns a page-sized `source` asset, exactly like a
        // cleanup or paint raster layer.
        let (_, element) = project
            .add_shape(
                right,
                SceneGeometry::rectangle(0.0, 0.0, 10.0, 10.0),
                ShapeGeometry::Rectangle { corner_radius: 0.0 },
                ShapeStyle {
                    fill: Some(ShapeFill::Solid([255, 0, 0, 255])),
                    stroke: None,
                },
                0.0,
            )
            .await
            .unwrap();
        project
            .rasterize_shapes(right, vec![element])
            .await
            .unwrap();
        assert_eq!(
            asset_dimensions(&project.snapshot(), element).await,
            (30, 10),
            "the bake is the source page's surface"
        );

        let (commit, _) = project.merge_pages(vec![left, right]).await.unwrap();
        project.record_commit(&commit);

        assert_eq!(
            asset_dimensions(&project.snapshot(), element).await,
            (50, 10),
            "a rasterized shape must be re-padded to the merged page width"
        );
    }

    /// Merging pages must carry a non-rasterized shape's `pattern` asset across,
    /// so the merged page's pattern still draws.
    ///
    /// `merge_pages` moves every child of each source page onto the merged page
    /// and shifts its polygon. Assets are stored per entity rather than per page,
    /// so they should move with the entity — but 1a's Critical was exactly this
    /// function losing a piece of shape state while moving the geometry, so the
    /// asset is asserted rather than assumed.
    #[tokio::test]
    async fn merging_pages_keeps_a_shapes_pattern_asset() {
        let mut session = Session::memory().await.unwrap();
        let source = AssetRole::new("source").unwrap();
        let pattern_role = AssetRole::new("pattern").unwrap();
        let mut setup = session.snapshot().edit();
        let left = setup
            .add_page(PageDraft::new("left", 20.0, 10.0), At::End)
            .unwrap();
        let right = setup
            .add_page(PageDraft::new("right", 30.0, 10.0), At::End)
            .unwrap();
        for (page, width, height) in [(left, 20u32, 10u32), (right, 30u32, 10u32)] {
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
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let (_, element) = project
            .add_shape(
                right,
                SceneGeometry::rectangle(0.0, 0.0, 10.0, 10.0),
                ShapeGeometry::Rectangle { corner_radius: 0.0 },
                ShapeStyle {
                    fill: Some(ShapeFill::Pattern(PatternSpec {
                        repeat: PatternRepeat::Tile,
                        origin: [0.0, 0.0],
                        scale: 0.5,
                        rotation_degrees: 0,
                        fallback: [1, 2, 3, 255],
                    })),
                    stroke: None,
                },
                0.0,
            )
            .await
            .unwrap();
        let tile = {
            let mut bytes = Cursor::new(Vec::new());
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, image::Rgba([9, 8, 7, 255])))
                .write_to(&mut bytes, ImageFormat::Png)
                .unwrap();
            bytes.into_inner()
        };
        project
            .set_shape_pattern_asset(element, tile, "image/png".to_owned())
            .await
            .unwrap();

        let (commit, merged) = project.merge_pages(vec![left, right]).await.unwrap();
        project.record_commit(&commit);
        let snapshot = project.snapshot();

        // The shape moved to the merged page...
        assert_eq!(snapshot.parent(element).unwrap(), Some(merged));
        // ...and its pattern asset came with it, with its pixels intact.
        let asset = snapshot
            .asset(element, &pattern_role)
            .unwrap()
            .expect("the merged page's shape must still own its pattern asset");
        let bytes = snapshot.read_blob(asset.blob).await.unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgba8();
        assert_eq!(decoded.dimensions(), (2, 2));
        assert_eq!(decoded.get_pixel(0, 0)[0], 9);
    }

    /// The pixel dimensions of an entity's `source` asset.
    async fn asset_dimensions(snapshot: &Snapshot, entity: EntityId) -> (u32, u32) {
        let asset = snapshot
            .asset(entity, &AssetRole::new("source").unwrap())
            .unwrap()
            .expect("the entity must own a source asset");
        let bytes = snapshot.read_blob(asset.blob).await.unwrap();
        image::load_from_memory(&bytes)
            .unwrap()
            .to_rgba8()
            .dimensions()
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
            None,
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
            None,
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
            None,
            3.0,
            &[ScenePoint { x: 2.0, y: 2.0 }],
        );
        assert_eq!(white.get_pixel(2, 2).0, [255, 255, 255, 255]);
    }

    #[test]
    fn an_along_stroke_ramp_varies_over_the_whole_stroke() {
        // Two well-separated segments, so cumulative length differs per pixel.
        let mut image = RgbaImage::new(64, 8);
        let points = vec![
            ScenePoint { x: 4.0, y: 4.0 },
            ScenePoint { x: 60.0, y: 4.0 },
        ];
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [0, 0, 0, 255],
            Some(BrushRamp::AlongStroke {
                from: [255, 0, 0, 255],
                to: [0, 0, 255, 255],
            }),
            4.0,
            &points,
        );
        let start = image.get_pixel(4, 4);
        let end = image.get_pixel(58, 4);
        assert!(
            start[0] > start[2],
            "the stroke start must be red-heavy, got {start:?}"
        );
        assert!(
            end[2] > end[0],
            "the stroke end must be blue-heavy, got {end:?}"
        );
    }

    /// A ramp must advance with *cumulative* length, not restart per segment.
    /// Two segments of equal length should give the same colour at equal
    /// cumulative fractions, and the second must not return to the first's
    /// colour.
    #[test]
    fn an_along_stroke_ramp_does_not_restart_at_each_segment() {
        let mut image = RgbaImage::new(64, 8);
        let points = vec![
            ScenePoint { x: 4.0, y: 4.0 },
            ScenePoint { x: 32.0, y: 4.0 },
            ScenePoint { x: 60.0, y: 4.0 },
        ];
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [0, 0, 0, 255],
            Some(BrushRamp::AlongStroke {
                from: [255, 0, 0, 255],
                to: [0, 0, 255, 255],
            }),
            4.0,
            &points,
        );
        let first = image.get_pixel(6, 4);
        let second = image.get_pixel(34, 4);
        // The second segment's start is past the halfway point, so it must be
        // bluer than the first segment's start. A per-segment restart would make
        // them nearly identical.
        assert!(
            second[2] > first[2],
            "the ramp must not restart per segment: first {first:?}, second {second:?}"
        );
    }

    #[test]
    fn a_fixed_axis_ramp_uses_page_coordinates() {
        // Horizontal: colour depends on x, and two points at the same x but
        // different y must match.
        let mut image = RgbaImage::new(64, 8);
        let ramp = BrushRamp::FixedAxis {
            from: [255, 0, 0, 255],
            to: [0, 0, 255, 255],
            axis: RampAxis::Horizontal,
        };
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [0, 0, 0, 255],
            Some(ramp),
            4.0,
            &[
                ScenePoint { x: 4.0, y: 2.0 },
                ScenePoint { x: 4.0, y: 6.0 },
                ScenePoint { x: 60.0, y: 6.0 },
            ],
        );
        let left = image.get_pixel(4, 2);
        let right = image.get_pixel(60, 6);
        assert!(left[0] > left[2], "the left edge must be red-heavy");
        assert!(right[2] > right[0], "the right edge must be blue-heavy");
    }

    #[test]
    fn a_vertical_fixed_axis_ramp_ignores_x() {
        let mut image = RgbaImage::new(8, 64);
        let ramp = BrushRamp::FixedAxis {
            from: [255, 0, 0, 255],
            to: [0, 0, 255, 255],
            axis: RampAxis::Vertical,
        };
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [0, 0, 0, 255],
            Some(ramp),
            4.0,
            &[
                ScenePoint { x: 4.0, y: 4.0 },
                ScenePoint { x: 4.0, y: 60.0 },
            ],
        );
        assert!(image.get_pixel(4, 4)[0] > image.get_pixel(4, 4)[2]);
        assert!(image.get_pixel(4, 60)[2] > image.get_pixel(4, 60)[0]);
    }

    /// The two ramp kinds must be computed by the same arithmetic the preview
    /// uses, so a single-point stroke cannot divide by zero and a reversed
    /// stroke must not run its ramp backwards.
    #[test]
    fn a_ramp_survives_degenerate_strokes() {
        let mut image = RgbaImage::new(16, 16);
        let ramp = BrushRamp::AlongStroke {
            from: [255, 0, 0, 255],
            to: [0, 0, 255, 255],
        };
        // A single point has zero length.
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [0, 0, 0, 255],
            Some(ramp),
            6.0,
            &[ScenePoint { x: 8.0, y: 8.0 }],
        );
        let single = image.get_pixel(8, 8);
        assert!(
            single[0] > single[2],
            "a zero-length stroke takes the ramp's start colour, got {single:?}"
        );

        // A stroke drawn right-to-left must ramp from its own start, not invert.
        let mut reversed = RgbaImage::new(64, 8);
        rasterize_stroke(
            &mut reversed,
            RasterStrokeMode::Paint,
            [0, 0, 0, 255],
            Some(ramp),
            4.0,
            &[
                ScenePoint { x: 58.0, y: 4.0 },
                ScenePoint { x: 6.0, y: 4.0 },
            ],
        );
        let start = reversed.get_pixel(58, 4);
        let end = reversed.get_pixel(8, 4);
        assert!(
            start[0] > start[2],
            "the ramp follows the stroke direction, so its start is red: {start:?}"
        );
        assert!(end[2] > end[0], "and its end is blue: {end:?}");
    }

    #[test]
    fn no_ramp_keeps_the_plain_stroke_colour() {
        let mut image = RgbaImage::new(32, 8);
        rasterize_stroke(
            &mut image,
            RasterStrokeMode::Paint,
            [12, 34, 56, 255],
            None,
            4.0,
            &[
                ScenePoint { x: 4.0, y: 4.0 },
                ScenePoint { x: 28.0, y: 4.0 },
            ],
        );
        // Every painted pixel must carry the flat colour; a ramp must not leak in.
        for pixel in image.pixels().filter(|pixel| pixel[3] > 0) {
            assert_eq!(
                [pixel[0], pixel[1], pixel[2]],
                [12, 34, 56],
                "without a ramp the colour is uniform"
            );
        }
    }

    /// A committed project holding one 100x100 page whose single raster layer
    /// owns a page-sized `source` asset filled with `pixel`.
    ///
    /// The layer is authored by a pipeline generation, which is how a cleanup or
    /// import actually produces one: `Edit::prepare_value` stamps `Origin::User`
    /// on any component written by the plain `edit()`, so only `edit_as` can
    /// build a genuinely generated fixture.
    async fn project_with_raster(image: RgbaImage) -> (Project, EntityId) {
        let generation = Generation::new(
            ProducerId::new("dev.koharu.pipeline.inpainting").expect("valid producer"),
        );
        let (width, height) = image.dimensions();
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit_as(generation);
        let page = setup
            .add_page(PageDraft::new("page", width as f64, height as f64), At::End)
            .unwrap();
        let layer = setup.add_entity(page, At::End).unwrap();
        setup
            .set(
                layer,
                &SceneRasterLayer {
                    origin: Origin::User,
                    name: "Paint 1".to_owned(),
                    kind: RasterLayerKind::Paint,
                },
            )
            .unwrap();
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image)
            .write_to(&mut encoded, ImageFormat::Png)
            .unwrap();
        setup
            .set_asset(
                layer,
                &AssetRole::new("source").unwrap(),
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
        session.commit(setup.finish().unwrap()).await.unwrap();
        (Project::new(session, "test".to_owned()), layer)
    }

    /// A committed project holding one 100x100 page and one text layer.
    async fn project_with_text_layer() -> (Project, EntityId) {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let content = setup.add_text_content(page, At::End).unwrap();
        let layer = setup
            .add_text_layer(
                page,
                At::End,
                content,
                &SceneTextLayout {
                    origin: Origin::User,
                    kind: TextLayoutKind::Point,
                    angle_degrees: None,
                },
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        (Project::new(session, "test".to_owned()), layer)
    }

    /// Decodes the `source` asset `entity` committed, so a test observes the
    /// pixels that were actually persisted rather than the in-memory ones.
    async fn source_image(snapshot: &Snapshot, entity: EntityId) -> RgbaImage {
        let asset = snapshot
            .asset(entity, &AssetRole::new("source").unwrap())
            .unwrap()
            .expect("the entity must own a source asset");
        let bytes = snapshot.read_blob(asset.blob).await.unwrap();
        image::load_from_memory(&bytes).unwrap().to_rgba8()
    }

    #[tokio::test]
    async fn adjusting_a_raster_rewrites_its_pixels() {
        let (mut project, layer) = project_with_raster(RgbaImage::from_pixel(
            100,
            100,
            image::Rgba([100, 100, 100, 255]),
        ))
        .await;
        assert_eq!(
            project
                .snapshot()
                .component::<SceneRasterLayer>(layer)
                .unwrap()
                .unwrap()
                .origin,
            Origin::Generated(Generation::new(
                ProducerId::new("dev.koharu.pipeline.inpainting").unwrap()
            )),
            "the fixture must start generated for the promotion below to mean anything"
        );

        let before = project.revision().get();
        let (commit, element) = project
            .adjust_raster(layer, ColorAdjust::Brightness { delta: 50 })
            .await
            .unwrap();
        assert_eq!(element, layer);
        // Pin exactly one commit, not merely "some revision": B-D5 requires the
        // pixels, the promotion, and the new asset to land atomically, so a
        // regression that split this into two commits must fail here. The
        // fixture already committed once when it built the page, hence the
        // delta rather than an absolute revision.
        assert_eq!(commit.revision.get(), before + 1);

        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(image.get_pixel(0, 0)[0], 150);

        // The layer is a generated one, so the adjustment has to claim both
        // origins or a later pipeline stage would treat these pixels as its own.
        let snapshot = project.snapshot();
        let raster = snapshot
            .component::<SceneRasterLayer>(layer)
            .unwrap()
            .unwrap();
        assert_eq!(raster.origin, Origin::User);
        assert_eq!(
            snapshot
                .component::<EntityOrigin>(layer)
                .unwrap()
                .unwrap()
                .origin,
            Origin::User
        );
    }

    #[tokio::test]
    async fn adjusting_a_raster_keeps_its_transparent_area_transparent() {
        // One transparent and one opaque pixel: the opaque pixel proves the
        // adjustment actually ran, so a no-op write cannot satisfy the alpha
        // assertion below.
        let mut original = RgbaImage::new(100, 100);
        original.put_pixel(0, 0, image::Rgba([200, 100, 50, 0]));
        original.put_pixel(1, 0, image::Rgba([200, 100, 50, 255]));
        let (mut project, layer) = project_with_raster(original).await;

        project
            .adjust_raster(layer, ColorAdjust::Grayscale)
            .await
            .unwrap();

        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(image.get_pixel(0, 0)[3], 0);
        assert_eq!(
            image.get_pixel(1, 0).0,
            [118, 118, 118, 255],
            "the opaque pixel must have been greyed"
        );
    }

    /// `apply_raster_stroke` treats a missing `source` asset as an empty
    /// page-sized surface rather than an error. Adjusting such a layer must do
    /// the same instead of failing on the absent blob.
    #[tokio::test]
    async fn adjusting_a_raster_without_a_source_asset_keeps_the_page_size() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let layer = setup.add_entity(page, At::End).unwrap();
        setup
            .set(
                layer,
                &SceneRasterLayer {
                    origin: Origin::User,
                    name: "Paint 1".to_owned(),
                    kind: RasterLayerKind::Paint,
                },
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        project
            .adjust_raster(layer, ColorAdjust::Grayscale)
            .await
            .unwrap();

        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(image.dimensions(), (100, 100));
    }

    #[tokio::test]
    async fn adjusting_a_non_raster_layer_is_rejected() {
        let (mut project, text) = project_with_text_layer().await;
        let error = project
            .adjust_raster(text, ColorAdjust::Grayscale)
            .await
            .expect_err("a text layer cannot be adjusted");
        // Assert the *raster* guard fired, not that page resolution happened to
        // fail. A text layer lives under a `TextGroup`, so if the page walk
        // regressed to a single `parent` call this would instead be an
        // `EntityNotFound` and the assertion below would catch it.
        assert!(
            error.to_string().contains("only raster layers"),
            "expected the raster guard, got: {error}"
        );
    }

    #[tokio::test]
    async fn adjusting_a_grouped_raster_layer_still_finds_its_page() {
        // A raster layer nested in a group is still a layer on that page, so it
        // must be adjustable. This is the case the immediate-parent lookup
        // broke: `snapshot.parent` returns the group, and `snapshot.page(group)`
        // is an error, so the adjustment used to fail with `EntityNotFound`.
        //
        // The layer is created directly under the group rather than moved there,
        // because `Project` exposes no edit entry point to its tests.
        let generation = Generation::new(
            ProducerId::new("dev.koharu.pipeline.inpainting").expect("valid producer"),
        );
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit_as(generation.clone());
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        let group = setup.add_entity(page, At::End).unwrap();
        setup
            .set(
                group,
                &SceneGroup {
                    origin: Origin::Generated(generation.clone()),
                    name: "Group 1".to_owned(),
                },
            )
            .unwrap();
        let layer = setup.add_entity(group, At::End).unwrap();
        setup
            .set(
                layer,
                &SceneRasterLayer {
                    origin: Origin::User,
                    name: "Paint 1".to_owned(),
                    kind: RasterLayerKind::Paint,
                },
            )
            .unwrap();
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            100,
            100,
            image::Rgba([100, 100, 100, 255]),
        ))
        .write_to(&mut encoded, ImageFormat::Png)
        .unwrap();
        setup
            .set_asset(
                layer,
                &AssetRole::new("source").unwrap(),
                AssetInput::new(
                    Arc::<[u8]>::from(encoded.into_inner()),
                    "image/png",
                    AssetMetadata {
                        width: Some(100),
                        height: Some(100),
                        attributes: Default::default(),
                    },
                ),
            )
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let snapshot = project.snapshot();
        assert_eq!(
            snapshot.parent(layer).unwrap(),
            Some(group),
            "the fixture must actually nest the layer under the group"
        );
        assert_eq!(
            snapshot.page(group).is_err(),
            true,
            "the group is not a page, which is what broke the old lookup"
        );

        project
            .adjust_raster(layer, ColorAdjust::Brightness { delta: 50 })
            .await
            .expect("a grouped raster layer is adjustable");

        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(image.get_pixel(0, 0)[0], 150);
    }

    #[tokio::test]
    async fn adjusting_with_an_invalid_parameter_is_rejected() {
        let (mut project, layer) = project_with_raster(RgbaImage::from_pixel(
            100,
            100,
            image::Rgba([100, 100, 100, 255]),
        ))
        .await;
        assert!(
            project
                .adjust_raster(
                    layer,
                    ColorAdjust::Levels {
                        low: 200,
                        high: 100
                    }
                )
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn filling_replaces_only_the_connected_region() {
        // A 4x1 image: the two left pixels are identical, the right two differ.
        let mut source = RgbaImage::new(4, 1);
        source.put_pixel(0, 0, image::Rgba([10, 10, 10, 255]));
        source.put_pixel(1, 0, image::Rgba([10, 10, 10, 255]));
        source.put_pixel(2, 0, image::Rgba([20, 20, 20, 255]));
        source.put_pixel(3, 0, image::Rgba([20, 20, 20, 255]));
        let (mut project, layer) = project_with_raster(source).await;

        project
            .fill_raster(layer, Point { x: 0.0, y: 0.0 }, [255, 0, 0, 255])
            .await
            .unwrap();

        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(
            image.get_pixel(0, 0),
            &image::Rgba([255, 0, 0, 255]),
            "the start pixel is filled"
        );
        assert_eq!(
            image.get_pixel(1, 0),
            &image::Rgba([255, 0, 0, 255]),
            "the connected neighbour is filled"
        );
        assert_eq!(
            image.get_pixel(2, 0),
            &image::Rgba([20, 20, 20, 255]),
            "a differently coloured neighbour is untouched"
        );
        assert_eq!(
            image.get_pixel(3, 0),
            &image::Rgba([20, 20, 20, 255]),
            "the region must not leak past the boundary"
        );
    }

    /// A raster layer's empty area is fully transparent, and a bucket must treat
    /// it as a region like any other — otherwise the tool cannot fill the blank
    /// space authors actually use it on.
    #[tokio::test]
    async fn filling_covers_a_transparent_region() {
        let mut source = RgbaImage::new(4, 1);
        source.put_pixel(0, 0, image::Rgba([0, 0, 0, 0]));
        source.put_pixel(1, 0, image::Rgba([0, 0, 0, 0]));
        source.put_pixel(2, 0, image::Rgba([200, 0, 0, 255]));
        source.put_pixel(3, 0, image::Rgba([0, 0, 0, 0]));
        let (mut project, layer) = project_with_raster(source).await;

        project
            .fill_raster(layer, Point { x: 0.0, y: 0.0 }, [255, 255, 0, 255])
            .await
            .unwrap();

        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(image.get_pixel(0, 0), &image::Rgba([255, 255, 0, 255]));
        assert_eq!(image.get_pixel(1, 0), &image::Rgba([255, 255, 0, 255]));
        assert_eq!(
            image.get_pixel(2, 0),
            &image::Rgba([200, 0, 0, 255]),
            "the opaque divider stops the fill"
        );
        assert_eq!(
            image.get_pixel(3, 0),
            &image::Rgba([0, 0, 0, 0]),
            "the far transparent pixel is a different region and stays empty"
        );
    }

    /// Filling with the colour already present must commit with no pixel change.
    ///
    /// The image is deliberately taller than one row. `flood_fill_mut`'s loop
    /// only ends when `inside` turns false, and `inside` asks whether a pixel
    /// still equals the start pixel — so passing the same colour back makes it
    /// push spans forever. A single-row image cannot reproduce that, because the
    /// rows above and below are out of bounds and nothing gets pushed; that is
    /// why this regression needs `H >= 2` to be worth anything.
    #[tokio::test]
    async fn filling_with_the_existing_colour_is_a_harmless_commit() {
        let (mut project, layer) =
            project_with_raster(RgbaImage::from_pixel(4, 4, image::Rgba([10, 10, 10, 255]))).await;

        let (commit, _) = project
            .fill_raster(layer, Point { x: 1.0, y: 1.0 }, [10, 10, 10, 255])
            .await
            .expect("filling with the same colour is not an error");

        assert!(commit.revision.get() > 0);
        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(image.dimensions(), (4, 4));
        for (x, y) in [(0, 0), (3, 0), (1, 1), (3, 3)] {
            assert_eq!(
                image.get_pixel(x, y),
                &image::Rgba([10, 10, 10, 255]),
                "no pixel changes when the fill colour is already present"
            );
        }
    }

    /// The click point is `f64` page space and `flood_fill_mut` takes `u32`.
    /// Casting before checking would turn `-1.0` into a huge `u32` and panic or
    /// fill the wrong pixel, so the bounds check must come first.
    #[tokio::test]
    async fn filling_outside_the_page_is_rejected() {
        let (mut project, layer) =
            project_with_raster(RgbaImage::from_pixel(4, 4, image::Rgba([10, 10, 10, 255]))).await;

        for point in [
            Point { x: -1.0, y: 0.0 },
            Point { x: 0.0, y: -1.0 },
            Point { x: 4.0, y: 0.0 },
            Point { x: 0.0, y: 4.0 },
            Point {
                x: f64::NAN,
                y: 0.0,
            },
        ] {
            assert!(
                project
                    .fill_raster(layer, point, [1, 2, 3, 255])
                    .await
                    .is_err(),
                "a point outside the page must be rejected: {point:?}"
            );
        }
    }

    #[tokio::test]
    async fn filling_a_non_raster_layer_is_rejected() {
        let (mut project, text) = project_with_text_layer().await;
        let error = project
            .fill_raster(text, Point { x: 0.0, y: 0.0 }, [1, 2, 3, 255])
            .await
            .expect_err("a text layer cannot be filled");
        // Pin the *fill* wording, not just the raster guard: "can be adjusted"
        // would satisfy a looser substring check while telling the user the
        // wrong tool failed.
        assert!(
            error
                .to_string()
                .contains("only raster layers can be filled"),
            "expected the fill-specific raster guard, got: {error}"
        );
    }

    /// The bottom-right corner is the boundary case: the start pixel is valid
    /// but has no in-bounds neighbour, so an off-by-one in the fill or the
    /// bounds check shows up here and nowhere else.
    #[tokio::test]
    async fn filling_at_the_corner_stops_at_the_edge() {
        let (mut project, layer) =
            project_with_raster(RgbaImage::from_pixel(3, 3, image::Rgba([5, 5, 5, 255]))).await;

        project
            .fill_raster(layer, Point { x: 2.0, y: 2.0 }, [9, 9, 9, 255])
            .await
            .unwrap();

        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(image.get_pixel(2, 2), &image::Rgba([9, 9, 9, 255]));
        assert_eq!(
            image.get_pixel(0, 0),
            &image::Rgba([9, 9, 9, 255]),
            "a uniform image fills entirely, corner included"
        );
    }

    /// `apply_raster_stroke` is the only caller of the shared asset path that can
    /// **create** its layer, and until now it had no end-to-end test — only the
    /// free `rasterize_stroke` function was covered. That branch and the shared
    /// write helper are now one path, so a regression here would break painting
    /// without any test failing.
    #[tokio::test]
    async fn painting_creates_a_layer_and_claims_both_origins() {
        // A page with no raster layer yet, so the stroke has to create one.
        let (mut project, existing) = project_with_raster(RgbaImage::new(100, 100)).await;
        let page = project
            .snapshot()
            .parent(existing)
            .unwrap()
            .expect("the fixture layer has a page");

        let (commit, layer) = project
            .apply_raster_stroke(
                page,
                None,
                RasterStrokeMode::Paint,
                [255, 0, 0, 255],
                None,
                4.0,
                vec![
                    ScenePoint { x: 8.0, y: 8.0 },
                    ScenePoint { x: 16.0, y: 8.0 },
                ],
            )
            .await
            .expect("painting without a target layer creates one");

        assert!(commit.revision.get() > 0);
        assert_ne!(layer, existing, "a new layer must be created");
        let snapshot = project.snapshot();
        let raster = snapshot
            .component::<SceneRasterLayer>(layer)
            .unwrap()
            .expect("the new entity is a raster layer");
        assert_eq!(
            raster.origin,
            Origin::User,
            "a user stroke owns its layer, or a later pipeline pass may discard it"
        );
        assert_eq!(raster.kind, RasterLayerKind::Paint);

        // `RasterLayer::origin` is persisted separately from the `EntityOrigin`
        // component and the pipeline gates on both, so assert both were claimed.
        assert_eq!(
            snapshot
                .component::<koharu_scene::EntityOrigin>(layer)
                .unwrap()
                .expect("the entity records an origin")
                .origin,
            Origin::User
        );

        let image = source_image(&snapshot, layer).await;
        assert_eq!(image.dimensions(), (100, 100));
    }

    /// `apply_raster_stroke`'s existing-layer path goes through the same shared
    /// read/write helpers as `adjust_raster` and `fill_raster`.
    #[tokio::test]
    async fn painting_an_existing_layer_keeps_its_identity() {
        let (mut project, layer) = project_with_raster(RgbaImage::new(100, 100)).await;
        let page = project
            .snapshot()
            .parent(layer)
            .unwrap()
            .expect("the layer has a page");

        let (_, returned) = project
            .apply_raster_stroke(
                page,
                Some(layer),
                RasterStrokeMode::Paint,
                [0, 255, 0, 255],
                None,
                6.0,
                vec![
                    ScenePoint { x: 40.0, y: 40.0 },
                    ScenePoint { x: 60.0, y: 40.0 },
                ],
            )
            .await
            .expect("painting an existing layer succeeds");

        assert_eq!(
            returned, layer,
            "the existing layer is reused, not replaced"
        );
        let image = source_image(&project.snapshot(), layer).await;
        assert_eq!(image.dimensions(), (100, 100));
        assert!(
            image.pixels().any(|pixel| pixel.0[1] > 0),
            "the stroke must have written green pixels somewhere"
        );
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
    /// dragging a shape fails outright. It must resync a line's endpoints the
    /// same way every other polygon write does (`Project::write_geometry`).
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
            .set_geometries(vec![(element, moved, 0.0)])
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

    /// `add_shape` is a polygon write path too, so it resyncs a `Line`'s derived
    /// endpoints through the same helper as the transform and spread paths. The
    /// client always sends matching values, so this drives the mismatch in
    /// directly: the polygon is authoritative whatever the caller passed.
    #[tokio::test]
    async fn adding_a_line_resyncs_its_endpoints_from_the_polygon() {
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
                        ScenePoint { x: 1.0, y: 2.0 },
                        ScenePoint { x: 5.0, y: 5.0 },
                        ScenePoint { x: 9.0, y: 8.0 },
                    ],
                },
                ShapeGeometry::Line {
                    start: [0.0, 0.0],
                    end: [999.0, 999.0],
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

        let stored = project
            .snapshot()
            .component::<SceneShapeLayer>(element)
            .unwrap()
            .unwrap();
        let ShapeGeometry::Line { start, end, .. } = stored.geometry else {
            panic!("expected a line shape");
        };
        assert_eq!(start, [1.0, 2.0]);
        assert_eq!(end, [9.0, 8.0]);
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
            name,
            shape,
            style: view_style,
            angle_degrees,
            rasterized,
            ..
        } = layer
        else {
            panic!("expected a shape layer");
        };
        // Decision D4: the layer list must be able to identify a shape by name
        // rather than falling back to "Image N".
        assert_eq!(name, "Rectangle 1");
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

    /// A committed project holding one 100x100 page and one rectangle shape.
    async fn project_with_shape(fill: Option<ShapeFill>) -> (Project, EntityId) {
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
                ShapeGeometry::Rectangle { corner_radius: 0.0 },
                ShapeStyle { fill, stroke: None },
                0.0,
            )
            .await
            .unwrap();
        (project, element)
    }

    /// Reads back the fill `set_shape_pattern` stored on `element`.
    fn stored_fill(project: &Project, element: EntityId) -> Option<ShapeFill> {
        project
            .snapshot()
            .component::<SceneShapeLayer>(element)
            .unwrap()
            .expect("the element is a shape layer")
            .style
            .fill
    }

    fn pattern_spec() -> PatternSpec {
        PatternSpec {
            repeat: PatternRepeat::Tile,
            origin: [0.25, 0.5],
            scale: 0.75,
            rotation_degrees: 45,
            fallback: [7, 8, 9, 255],
        }
    }

    /// A PNG of `width` by `height`, which is what the asset command has to
    /// decode to learn the true dimensions the renderer will upload.
    fn png_bytes(width: u32, height: u32) -> Vec<u8> {
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            width,
            height,
            image::Rgba([1, 2, 3, 255]),
        ))
        .write_to(&mut encoded, ImageFormat::Png)
        .unwrap();
        encoded.into_inner()
    }

    #[tokio::test]
    async fn clearing_a_pattern_restores_the_previous_solid_colour() {
        let (mut project, element) =
            project_with_shape(Some(ShapeFill::Solid([4, 5, 6, 255]))).await;
        project
            .set_shape_pattern(element, Some(pattern_spec()))
            .await
            .unwrap();
        assert!(matches!(
            stored_fill(&project, element),
            Some(ShapeFill::Pattern(_))
        ));

        let (_, returned) = project.set_shape_pattern(element, None).await.unwrap();
        assert_eq!(returned, element);
        assert_eq!(
            stored_fill(&project, element),
            Some(ShapeFill::Solid([7, 8, 9, 255])),
            "clearing a pattern keeps the fallback colour the pattern was drawing"
        );
    }

    /// A gradient and an absent fill carry no colour to keep, so clearing a
    /// pattern over them must still produce a usable solid rather than failing.
    #[tokio::test]
    async fn clearing_a_pattern_over_a_gradient_uses_the_default_colour() {
        let gradient = ShapeFill::Gradient(GradientSpec {
            kind: GradientKind::Linear {
                start: [0.0, 0.0],
                end: [1.0, 0.0],
            },
            stops: vec![
                GradientStop {
                    offset: 0.0,
                    color: [255, 255, 255, 255],
                },
                GradientStop {
                    offset: 1.0,
                    color: [0, 0, 0, 255],
                },
            ],
            extend: GradientExtend::Pad,
        });
        let (mut project, element) = project_with_shape(Some(gradient)).await;

        project.set_shape_pattern(element, None).await.unwrap();

        assert_eq!(
            stored_fill(&project, element),
            Some(ShapeFill::Solid([255, 255, 255, 255]))
        );
    }

    #[tokio::test]
    async fn setting_a_pattern_on_a_raster_layer_is_rejected() {
        let (mut project, layer) = project_with_raster(RgbaImage::new(100, 100)).await;
        let error = project
            .set_shape_pattern(layer, Some(pattern_spec()))
            .await
            .expect_err("a raster layer cannot carry a pattern");
        assert!(
            error.to_string().contains("only shape layers"),
            "expected the shape guard, got: {error}"
        );
    }

    #[tokio::test]
    async fn the_pattern_spec_round_trips_through_the_layer() {
        let (mut project, element) = project_with_shape(None).await;
        let spec = pattern_spec();

        project
            .set_shape_pattern(element, Some(spec))
            .await
            .unwrap();

        assert_eq!(
            stored_fill(&project, element),
            Some(ShapeFill::Pattern(spec)),
            "every field must survive the store, not just the variant"
        );
    }

    /// Validating before the patch is what keeps a rejected spec from leaving a
    /// half-written layer behind: `ShapeLayer::validate` would reject it anyway,
    /// but after the promotion had already been committed.
    #[tokio::test]
    async fn setting_an_invalid_pattern_is_rejected() {
        let (mut project, element) = project_with_shape(None).await;
        let mut spec = pattern_spec();
        spec.scale = 0.0;

        let error = project
            .set_shape_pattern(element, Some(spec))
            .await
            .expect_err("a non-positive scale is not a usable pattern");
        assert!(error.to_string().contains("scale"), "{error}");
        assert_eq!(stored_fill(&project, element), None);
    }

    #[tokio::test]
    async fn writing_a_pattern_asset_to_a_raster_layer_is_rejected() {
        let (mut project, layer) = project_with_raster(RgbaImage::new(100, 100)).await;
        let error = project
            .set_shape_pattern_asset(layer, png_bytes(2, 2), "image/png".to_owned())
            .await
            .expect_err("a raster layer does not own a pattern");
        assert!(
            error.to_string().contains("only shape layers"),
            "expected the shape guard, got: {error}"
        );
    }

    #[tokio::test]
    async fn writing_undecodable_pattern_bytes_is_rejected() {
        let (mut project, element) = project_with_shape(None).await;
        let error = project
            .set_shape_pattern_asset(element, b"not an image".to_vec(), "image/png".to_owned())
            .await
            .expect_err("undecodable bytes must not be stored");
        assert!(
            error.to_string().contains("decodable"),
            "expected the decode guard, got: {error}"
        );
    }

    #[tokio::test]
    async fn a_pattern_asset_records_the_decoded_dimensions() {
        let (mut project, element) = project_with_shape(None).await;

        let (_, returned) = project
            .set_shape_pattern_asset(element, png_bytes(3, 7), "image/webp".to_owned())
            .await
            .unwrap();
        assert_eq!(returned, element);

        let snapshot = project.snapshot();
        let asset = snapshot
            .asset(element, &AssetRole::new("pattern").unwrap())
            .unwrap()
            .expect("the shape layer owns a pattern asset");
        // The dimensions come from the decoded pixels, not from the caller, so a
        // renderer upload can never disagree with the texture it uploaded.
        assert_eq!(asset.metadata.width, Some(3));
        assert_eq!(asset.metadata.height, Some(7));
        assert_eq!(asset.media_type, "image/webp");
        assert_eq!(
            image::load_from_memory(&snapshot.read_blob(asset.blob).await.unwrap())
                .unwrap()
                .into_rgba8()
                .dimensions(),
            (3, 7)
        );
    }

    /// The pattern half of the wire contract, alongside the shape and gradient
    /// assertions above. `ShapeFill::Pattern` is a newtype variant, so it stays
    /// externally tagged, and `PatternRepeat` is a unit-variant enum, so it is a
    /// bare lowercase string. The field order and the integer formatting of
    /// `rotation_degrees` follow the struct declaration.
    #[test]
    fn the_pattern_wire_shape_matches_the_typescript_client() {
        let tile = PatternSpec {
            repeat: PatternRepeat::Tile,
            origin: [0.0, 0.0],
            scale: 0.25,
            rotation_degrees: 0,
            fallback: [255, 255, 255, 255],
        };
        assert_eq!(
            serde_json::to_string(&ShapeFill::Pattern(tile)).unwrap(),
            r#"{"pattern":{"repeat":"tile","origin":[0.0,0.0],"scale":0.25,"rotation_degrees":0,"fallback":[255,255,255,255]}}"#
        );
        // The fill is one member of `ShapeStyle`, so pin the enclosing key too:
        // the TypeScript client reads `fill.pattern`, not a bare pattern.
        assert_eq!(
            serde_json::to_string(&ShapeStyle {
                fill: Some(ShapeFill::Pattern(tile)),
                stroke: None,
            })
            .unwrap(),
            r#"{"fill":{"pattern":{"repeat":"tile","origin":[0.0,0.0],"scale":0.25,"rotation_degrees":0,"fallback":[255,255,255,255]}},"stroke":null}"#
        );
        assert_eq!(
            serde_json::to_string(&ShapeFill::Pattern(PatternSpec {
                repeat: PatternRepeat::Stretch,
                ..tile
            }))
            .unwrap(),
            r#"{"pattern":{"repeat":"stretch","origin":[0.0,0.0],"scale":0.25,"rotation_degrees":0,"fallback":[255,255,255,255]}}"#
        );
    }

    #[tokio::test]
    async fn a_rasterized_shape_keeps_a_transparent_background() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let style = ShapeStyle {
            fill: Some(ShapeFill::Solid([255, 0, 0, 255])),
            stroke: None,
        };
        let (_, element) = project
            .add_shape(
                page,
                SceneGeometry::rectangle(10.0, 10.0, 20.0, 20.0),
                ShapeGeometry::Rectangle { corner_radius: 0.0 },
                style,
                0.0,
            )
            .await
            .unwrap();
        project.rasterize_shapes(page, vec![element]).await.unwrap();

        let snapshot = project.snapshot();
        let asset = snapshot
            .asset(element, &AssetRole::new("source").unwrap())
            .unwrap()
            .expect("rasterized shape must own a source asset");
        let bytes = snapshot.read_blob(asset.blob).await.unwrap();
        let image = image::load_from_memory(&bytes).unwrap().to_rgba8();

        // 도형 밖은 완전히 투명해야 한다.
        assert_eq!(image.get_pixel(0, 0)[3], 0);
        assert_eq!(image.get_pixel(image.width() - 1, image.height() - 1)[3], 0);
        // 도형 안쪽은 불투명한 빨강이어야 한다.
        let center = image.get_pixel(20, 20);
        assert_eq!(center[3], 255);
        assert!(center[0] > 200);

        let Layer::Shape { rasterized, .. } = Project::page(&snapshot, page)
            .unwrap()
            .layers
            .into_iter()
            .find(|layer| layer_id(layer) == element)
            .unwrap()
        else {
            panic!("expected a shape layer");
        };
        assert!(rasterized);
    }

    /// A rasterized shape with a pattern fill must bake **the pattern**, not its
    /// fallback colour.
    ///
    /// `rasterize_shapes` builds the path with `shape_layer_path` and draws it
    /// directly, and `shape_layer_path` has no way to reach the layer's `pattern`
    /// asset — `PatternSpec` references no asset by design (P-D2), so only a
    /// caller holding the snapshot can attach the pixels. Without that
    /// attachment `append_to` sees `image: None` and paints `fallback`, so a
    /// baked pattern would silently become a flat colour.
    ///
    /// The uploaded pattern is one red pixel, and the fallback is a distinct
    /// blue, so the two outcomes cannot be confused.
    #[tokio::test]
    async fn a_rasterized_shape_bakes_its_pattern_not_its_fallback() {
        let mut session = Session::memory().await.unwrap();
        let mut setup = session.snapshot().edit();
        let page = setup
            .add_page(PageDraft::new("page", 100.0, 100.0), At::End)
            .unwrap();
        session.commit(setup.finish().unwrap()).await.unwrap();
        let mut project = Project::new(session, "test".to_owned());

        let fallback = [0, 0, 255, 255];
        let style = ShapeStyle {
            fill: Some(ShapeFill::Pattern(PatternSpec {
                repeat: PatternRepeat::Tile,
                origin: [0.0, 0.0],
                scale: 0.25,
                rotation_degrees: 0,
                fallback,
            })),
            stroke: None,
        };
        let (_, element) = project
            .add_shape(
                page,
                SceneGeometry::rectangle(10.0, 10.0, 80.0, 80.0),
                ShapeGeometry::Rectangle { corner_radius: 0.0 },
                style,
                0.0,
            )
            .await
            .unwrap();
        // One opaque red pixel: every tile is red, so a correct bake is red
        // everywhere inside the shape and never the blue fallback.
        let red = {
            let mut bytes = Cursor::new(Vec::new());
            DynamicImage::ImageRgba8(image::RgbaImage::from_raw(1, 1, vec![255, 0, 0, 255]).unwrap())
                .write_to(&mut bytes, ImageFormat::Png)
                .unwrap();
            bytes.into_inner()
        };
        project
            .set_shape_pattern_asset(element, red, "image/png".to_owned())
            .await
            .unwrap();
        project.rasterize_shapes(page, vec![element]).await.unwrap();

        let snapshot = project.snapshot();
        let asset = snapshot
            .asset(element, &AssetRole::new("source").unwrap())
            .unwrap()
            .expect("a rasterized shape must own a source asset");
        let bytes = snapshot.read_blob(asset.blob).await.unwrap();
        let baked = image::load_from_memory(&bytes).unwrap().to_rgba8();

        let center = baked.get_pixel(50, 50);
        assert_eq!(center[3], 255, "the pattern must fill the shape");
        assert!(
            center[0] > 200 && center[2] < 60,
            "the baked shape must show the uploaded red pattern, not the blue \
             fallback; got {center:?}"
        );
    }

    /// A `Polygon` has no drawing parameters of its own: its path comes from
    /// the polygon. Baking it therefore proves `rasterize_shapes` goes through
    /// `shape_layer_path` — `shape_path` would produce an empty path and this
    /// asset would be entirely transparent.
    #[tokio::test]
    async fn a_rasterized_polygon_draws_from_its_points() {
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
                SceneGeometry::rectangle(20.0, 20.0, 60.0, 60.0),
                ShapeGeometry::Polygon,
                ShapeStyle {
                    fill: Some(ShapeFill::Solid([0, 0, 255, 255])),
                    stroke: None,
                },
                0.0,
            )
            .await
            .unwrap();
        project.rasterize_shapes(page, vec![element]).await.unwrap();

        let snapshot = project.snapshot();
        let asset = snapshot
            .asset(element, &AssetRole::new("source").unwrap())
            .unwrap()
            .expect("rasterized shape must own a source asset");
        let bytes = snapshot.read_blob(asset.blob).await.unwrap();
        let image = image::load_from_memory(&bytes).unwrap().to_rgba8();

        assert_eq!(image.get_pixel(0, 0)[3], 0);
        let inside = image.get_pixel(50, 50);
        assert_eq!(inside[3], 255);
        assert!(inside[2] > 200);
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

    /// The gradient half of the same contract. `ShapeFill::Gradient` is a
    /// newtype variant, so it stays externally tagged; `GradientKind` is a
    /// struct-variant enum, so each kind is `{"kind_name":{...}}`; and
    /// `GradientExtend` is a unit-variant enum, so it is a bare lowercase
    /// string. The `f32` fields follow the `2.0` notation the `StrokeStyle`
    /// assertion above already pins.
    ///
    /// The `Solid` regression is deliberately *not* repeated here: the first
    /// assertion of `the_shape_wire_shape_matches_the_typescript_client` is
    /// already byte-identical to the one this task's brief specifies, so a copy
    /// would assert the same literal twice.
    #[test]
    fn the_gradient_wire_shape_matches_the_typescript_client() {
        let style = ShapeStyle {
            fill: Some(ShapeFill::Gradient(GradientSpec {
                kind: GradientKind::Linear {
                    start: [0.0, 0.0],
                    end: [1.0, 1.0],
                },
                stops: vec![
                    GradientStop {
                        offset: 0.0,
                        color: [1, 2, 3, 255],
                    },
                    GradientStop {
                        offset: 1.0,
                        color: [4, 5, 6, 255],
                    },
                ],
                extend: GradientExtend::Repeat,
            })),
            stroke: None,
        };
        assert_eq!(
            serde_json::to_string(&style).unwrap(),
            r#"{"fill":{"gradient":{"kind":{"linear":{"start":[0.0,0.0],"end":[1.0,1.0]}},"stops":[{"offset":0.0,"color":[1,2,3,255]},{"offset":1.0,"color":[4,5,6,255]}],"extend":"repeat"}},"stroke":null}"#
        );

        let radial = ShapeFill::Gradient(GradientSpec {
            kind: GradientKind::Radial {
                center: [0.5, 0.5],
                radius: 0.25,
            },
            stops: vec![
                GradientStop {
                    offset: 0.0,
                    color: [0, 0, 0, 255],
                },
                GradientStop {
                    offset: 1.0,
                    color: [255, 255, 255, 255],
                },
            ],
            extend: GradientExtend::Pad,
        });
        assert_eq!(
            serde_json::to_string(&radial).unwrap(),
            r#"{"gradient":{"kind":{"radial":{"center":[0.5,0.5],"radius":0.25}},"stops":[{"offset":0.0,"color":[0,0,0,255]},{"offset":1.0,"color":[255,255,255,255]}],"extend":"pad"}}"#
        );

        let sweep = ShapeFill::Gradient(GradientSpec {
            kind: GradientKind::Sweep {
                center: [0.5, 0.5],
                start_degrees: 0.0,
                end_degrees: 360.0,
            },
            stops: vec![
                GradientStop {
                    offset: 0.0,
                    color: [255, 0, 0, 255],
                },
                GradientStop {
                    offset: 1.0,
                    color: [0, 0, 255, 255],
                },
            ],
            extend: GradientExtend::Reflect,
        });
        assert_eq!(
            serde_json::to_string(&sweep).unwrap(),
            r#"{"gradient":{"kind":{"sweep":{"center":[0.5,0.5],"start_degrees":0.0,"end_degrees":360.0}},"stops":[{"offset":0.0,"color":[255,0,0,255]},{"offset":1.0,"color":[0,0,255,255]}],"extend":"reflect"}}"#
        );
    }
}
