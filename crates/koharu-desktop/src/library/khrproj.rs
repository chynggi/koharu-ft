//! A `.khrproj` folder, the project format of Koharu 0.83 and of the NeKoharu
//! releases up to 0.85.3-N, made into a `.khr` file. The folder holds two
//! checksummed `KHRSTATE` slots, the newer one current, each a `revision`
//! encoding of the scene's entities, their components, and their relations,
//! and content-addressed blobs beside them. The reader mirrors the stored
//! layout of NeKoharu v0.85.3-N (4142f905), `koharu-storage/src/format.rs`
//! and `koharu-scene/src/{state,component,components}`, which reads every
//! file those releases wrote; it decodes only the components the conversion
//! uses. The folder is only read.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, ensure};
use icu_properties::{CodePointMapData, props::Script};
use koharu_scene::{Edit, Hash, Meta, Op, Rect, Session, Transform};
use kurbo::{Affine, Point};
use revision::DeserializeRevisioned;
use serde_json::{Value, json};

/// The types as v0.85.3-N stored them. Fields the conversion doesn't read
/// stay, because each one's bytes sit in the stream.
#[allow(dead_code)]
mod stored {
    use std::collections::BTreeMap;

    use revision::{DeserializeRevisioned, Error, Revisioned, SerializeRevisioned, revisioned};
    use uuid::Uuid;

    /// A `DocumentId`, `EntityId`, or `RelationId`.
    #[revisioned(revision = 1)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct Id(pub Uuid);

    #[revisioned(revision = 1)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct BlobId(pub [u8; 32]);

    #[revisioned(revision = 1)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Revision(pub u64);

    /// A slot's body, `koharu-storage`'s `StoredState`.
    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Slot {
        pub document: Id,
        pub revision: Revision,
        pub blobs: Vec<BlobId>,
        pub payload: Vec<u8>,
    }

    /// The scene's checkpoint, `koharu-scene`'s `StoredState`.
    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Checkpoint {
        pub page_order_epoch: u64,
        pub project_components: Vec<Entry>,
        pub pages: Vec<Page>,
        pub relations: Vec<StoredRelation>,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Entry {
        pub key: Key,
        pub value: Component,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Key {
        pub kind: String,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Component {
        pub schema: u32,
        pub payload: Vec<u8>,
        pub record_refs: Vec<Id>,
        pub blob_refs: Vec<BlobId>,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Entity {
        pub id: Id,
        pub parent: Option<Id>,
        pub components: Vec<Entry>,
    }

    /// Its root entity first, then each entity after its parent, a parent's
    /// children in order.
    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Page {
        pub id: Id,
        pub epoch: u64,
        pub entities: Vec<Entity>,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct StoredRelation {
        pub id: Id,
        pub value: Relation,
        pub components: Vec<Entry>,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Relation {
        pub origin: Origin,
        pub kind: Name,
        pub source: Id,
        pub target: Id,
    }

    /// A `RegionKind`, `RelationKind`, `LanguageTag`, `AssetRole`, or
    /// `ProducerId`: a revisioned string.
    #[revisioned(revision = 1)]
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Name(pub String);

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Generation {
        pub producer: Name,
        pub model: Option<String>,
        pub confidence: Option<f32>,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub enum Origin {
        User,
        Generated(Generation),
    }

    /// `Authored<String>`, written by hand without a revision of its own.
    #[derive(Debug)]
    pub struct Authored {
        pub value: String,
        pub origin: Origin,
    }

    impl Revisioned for Authored {
        fn revision() -> u16 {
            1
        }
    }

    impl SerializeRevisioned for Authored {
        fn serialize_revisioned<W: std::io::Write>(&self, writer: &mut W) -> Result<(), Error> {
            self.value.serialize_revisioned(writer)?;
            self.origin.serialize_revisioned(writer)
        }
    }

    impl DeserializeRevisioned for Authored {
        fn deserialize_revisioned<R: std::io::Read>(reader: &mut R) -> Result<Self, Error> {
            Ok(Self {
                value: String::deserialize_revisioned(reader)?,
                origin: Origin::deserialize_revisioned(reader)?,
            })
        }
    }

    /// `SourceText` and `Translation`, also written by hand.
    #[derive(Debug)]
    pub struct Words {
        pub text: Authored,
        pub language: Option<Name>,
    }

    impl Revisioned for Words {
        fn revision() -> u16 {
            1
        }
    }

    impl SerializeRevisioned for Words {
        fn serialize_revisioned<W: std::io::Write>(&self, writer: &mut W) -> Result<(), Error> {
            self.text.serialize_revisioned(writer)?;
            self.language.serialize_revisioned(writer)
        }
    }

    impl DeserializeRevisioned for Words {
        fn deserialize_revisioned<R: std::io::Read>(reader: &mut R) -> Result<Self, Error> {
            Ok(Self {
                text: Authored::deserialize_revisioned(reader)?,
                language: Option::deserialize_revisioned(reader)?,
            })
        }
    }

    #[revisioned(revision = 1)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Point {
        pub x: f64,
        pub y: f64,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Geometry {
        pub origin: Origin,
        pub points: Vec<Point>,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Visibility {
        pub origin: Origin,
        pub visible: bool,
        pub opacity: f32,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TextDirection {
        Auto,
        Horizontal,
        Vertical,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct OcrAnalysis {
        pub origin: Origin,
        pub direction: TextDirection,
        pub confidence: Option<f32>,
        pub line_boundaries: Vec<[Point; 4]>,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Region {
        pub origin: Origin,
        pub kind: Name,
        pub label: Option<String>,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct AssetMetadata {
        pub width: Option<u32>,
        pub height: Option<u32>,
        pub attributes: BTreeMap<String, String>,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Asset {
        pub origin: Origin,
        pub blob: BlobId,
        pub media_type: String,
        pub metadata: AssetMetadata,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Assets {
        pub values: BTreeMap<Name, Asset>,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct Group {
        pub origin: Origin,
        pub name: String,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RasterLayerKind {
        Cleanup,
        Paint,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct RasterLayer {
        pub origin: Origin,
        pub name: String,
        pub kind: RasterLayerKind,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub enum TextLayoutKind {
        Point,
        Paragraph,
    }

    #[revisioned(revision = 2)]
    #[derive(Debug)]
    pub struct TextLayout {
        pub origin: Origin,
        pub kind: TextLayoutKind,
        #[revision(start = 2)]
        pub angle_degrees: Option<f32>,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug, Clone, Copy)]
    pub enum TextAlignment {
        Start,
        Center,
        End,
        Justify,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum WritingMode {
        Horizontal,
        Vertical,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FontStyle {
        Normal,
        Italic,
        Oblique,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug, Clone, Copy)]
    pub enum StrokeMethod {
        Expansion,
        Contour,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug, Clone, Copy)]
    pub enum StrokePosition {
        Outside,
        Center,
        Inside,
    }

    #[revisioned(revision = 1)]
    #[derive(Debug, Clone, Copy)]
    pub enum StrokeJoin {
        Round,
        Miter,
        Bevel,
    }

    #[revisioned(revision = 4)]
    #[derive(Debug)]
    pub struct Typography {
        pub origin: Origin,
        pub preferred_font: Option<String>,
        pub font_weight: Option<u16>,
        pub font_style: Option<FontStyle>,
        pub size: Option<f32>,
        pub auto_fit: bool,
        pub color: Option<[u8; 4]>,
        pub stroke_color: Option<[u8; 4]>,
        pub stroke_width: Option<f32>,
        pub alignment: Option<TextAlignment>,
        pub writing_mode: Option<WritingMode>,
        pub extensions: BTreeMap<String, String>,
        #[revision(start = 2)]
        pub stroke_method: Option<StrokeMethod>,
        #[revision(start = 2)]
        pub stroke_position: Option<StrokePosition>,
        #[revision(start = 2)]
        pub stroke_join: Option<StrokeJoin>,
        #[revision(start = 3)]
        pub outer_stroke_color: Option<[u8; 4]>,
        #[revision(start = 3)]
        pub outer_stroke_width: Option<f32>,
        #[revision(start = 4)]
        pub line_spacing: Option<f32>,
    }

    /// The `Page` component.
    #[revisioned(revision = 2)]
    #[derive(Debug)]
    pub struct PageInfo {
        pub label: String,
        pub width: f64,
        pub height: f64,
        #[revision(start = 2, default_fn = "default_chapter_start")]
        pub chapter_start: bool,
    }

    impl PageInfo {
        fn default_chapter_start(_revision: u16) -> Result<bool, Error> {
            Ok(false)
        }
    }

    #[revisioned(revision = 1)]
    #[derive(Debug)]
    pub struct TextRole {
        pub origin: Origin,
        pub role: String,
    }
}

use stored::*;

const MAGIC: &[u8; 8] = b"KHRSTATE";
const HEADER_BYTES: usize = 8 + 4 + 8 + 32;

const PAGE: &str = "dev.koharu.page";
const ASSETS: &str = "dev.koharu.assets";
const GEOMETRY: &str = "dev.koharu.geometry";
const VISIBILITY: &str = "dev.koharu.visibility";
const REGION: &str = "dev.koharu.analysis.region";
const OCR: &str = "dev.koharu.analysis.ocr";
const GROUP: &str = "dev.koharu.group";
const TEXT_GROUP: &str = "dev.koharu.text.group";
const RASTER: &str = "dev.koharu.layer.raster";
const TEXT_LAYOUT: &str = "dev.koharu.layer.text";
const TYPOGRAPHY: &str = "dev.koharu.text.typography";
const SOURCE: &str = "dev.koharu.text.source";
const TRANSLATION: &str = "dev.koharu.text.translation";
const ROLE: &str = "dev.koharu.text.role";
const PRESENTS: &str = "dev.koharu.relation.presents";
const RECOGNIZED_FROM: &str = "dev.koharu.relation.recognized-from";
const FITS_TO: &str = "dev.koharu.relation.fits-to";
const FLOWS_IN: &str = "dev.koharu.relation.flows-in";
const INSIDE: &str = "dev.koharu.relation.inside";

/// Reads the project in `folder` and writes it as a new project at `target`,
/// which must not exist. A failed conversion leaves no file there.
pub(super) fn migrate(folder: &Path, target: &Path, meta: Meta) -> Result<()> {
    ensure!(!target.exists(), "{} already exists", target.display());
    let scene = Scene::read(folder)?;
    let written = (|| {
        let mut session = Session::create(target, meta)?;
        for (index, &page) in scene.pages.iter().enumerate() {
            let ops = page_ops(&scene, page, index)
                .with_context(|| format!("failed to convert page {}", index + 1))?;
            session.apply(Edit {
                label: "Upgrade".into(),
                ops,
            })?;
        }
        anyhow::Ok(())
    })();
    if written.is_err() && target.exists() {
        let _ = Session::remove(target);
    }
    written
}

/// A slot's body, checked as v0.85.3-N's `format::decode` checks it; none for
/// a slot that isn't there.
fn slot(path: &Path) -> Result<Option<Slot>> {
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs_err::read(path)?;
    ensure!(
        bytes.len() >= HEADER_BYTES && &bytes[..8] == MAGIC,
        "{} isn't a project state",
        path.display()
    );
    let version = u32::from_le_bytes(bytes[8..12].try_into()?);
    ensure!(version == 1, "project state version {version} is unknown");
    let length = u64::from_le_bytes(bytes[12..20].try_into()?);
    let body = &bytes[HEADER_BYTES..];
    ensure!(
        body.len() as u64 == length,
        "the project state is cut short"
    );
    ensure!(
        blake3::hash(body).as_bytes() == &bytes[20..52],
        "the project state's checksum doesn't match"
    );
    Ok(Some(revision::from_slice(body)?))
}

/// A blob's file, which `koharu-storage` names by its BLAKE3 hash.
fn blob_path(root: &Path, id: BlobId) -> PathBuf {
    let name = blake3::Hash::from_bytes(id.0).to_hex();
    root.join("blobs").join(&name[..2]).join(name.as_str())
}

struct Node {
    parent: Option<Id>,
    components: HashMap<String, Vec<u8>>,
}

/// The current checkpoint as the conversion reads it.
struct Scene {
    root: PathBuf,
    nodes: HashMap<Id, Node>,
    children: HashMap<Id, Vec<Id>>,
    /// Each entity's relations out, as kind and target.
    relations: HashMap<Id, Vec<(String, Id)>>,
    pages: Vec<Id>,
}

impl Scene {
    /// The newest slot whose blobs are all there, as v0.85.3-N's
    /// `open_project` picks it.
    fn read(root: &Path) -> Result<Self> {
        let mut slots = Vec::new();
        for name in ["state-a.khr", "state-b.khr"] {
            slots.extend(slot(&root.join(name))?);
        }
        slots.sort_by_key(|slot| std::cmp::Reverse(slot.revision));
        let current = slots
            .into_iter()
            .find(|slot| (slot.blobs.iter()).all(|&blob| blob_path(root, blob).is_file()))
            .context("the project has no state whose files are all there")?;
        let checkpoint: Checkpoint = revision::from_slice(&current.payload)?;
        let mut scene = Self {
            root: root.to_owned(),
            nodes: HashMap::new(),
            children: HashMap::new(),
            relations: HashMap::new(),
            pages: Vec::new(),
        };
        for page in checkpoint.pages {
            scene.pages.push(page.id);
            for entity in page.entities {
                if let Some(parent) = entity.parent {
                    scene.children.entry(parent).or_default().push(entity.id);
                }
                let components = (entity.components.into_iter())
                    .map(|entry| (entry.key.kind, entry.value.payload))
                    .collect();
                scene.nodes.insert(
                    entity.id,
                    Node {
                        parent: entity.parent,
                        components,
                    },
                );
            }
        }
        for relation in checkpoint.relations {
            let value = relation.value;
            (scene.relations.entry(value.source).or_default()).push((value.kind.0, value.target));
        }
        Ok(scene)
    }

    fn get<T: DeserializeRevisioned>(&self, id: Id, kind: &str) -> Result<Option<T>> {
        let Some(bytes) = (self.nodes.get(&id)).and_then(|node| node.components.get(kind)) else {
            return Ok(None);
        };
        let value =
            revision::from_slice(bytes).with_context(|| format!("failed to read {kind}"))?;
        Ok(Some(value))
    }

    fn has(&self, id: Id, kind: &str) -> bool {
        (self.nodes.get(&id)).is_some_and(|node| node.components.contains_key(kind))
    }

    fn points(&self, id: Option<Id>) -> Result<Option<Vec<stored::Point>>> {
        Ok(match id {
            Some(id) => self
                .get::<Geometry>(id, GEOMETRY)?
                .map(|geometry| geometry.points),
            None => None,
        })
    }

    fn children(&self, id: Id) -> &[Id] {
        self.children.get(&id).map_or(&[], Vec::as_slice)
    }

    fn page_of(&self, mut id: Id) -> Id {
        while let Some(parent) = self.nodes.get(&id).and_then(|node| node.parent) {
            id = parent;
        }
        id
    }

    /// The target of `id`'s relation of `kind`, which the old scene kept to
    /// one.
    fn target(&self, id: Id, kind: &str) -> Option<Id> {
        (self.relations.get(&id)?.iter())
            .find(|(each, _)| each == kind)
            .map(|&(_, target)| target)
    }

    /// The entities whose relation of `kind` points at `id`.
    fn sources(&self, id: Id, kind: &str) -> Vec<Id> {
        let mut sources: Vec<Id> = (self.relations.iter())
            .filter(|(_, out)| {
                out.iter()
                    .any(|(each, target)| each == kind && *target == id)
            })
            .map(|(&source, _)| source)
            .collect();
        sources.sort();
        sources
    }

    fn asset(&self, id: Id, role: &str) -> Result<Option<Vec<u8>>> {
        let Some(assets) = self.get::<Assets>(id, ASSETS)? else {
            return Ok(None);
        };
        let Some(asset) = assets.values.get(&Name(role.to_owned())) else {
            return Ok(None);
        };
        Ok(Some(fs_err::read(blob_path(&self.root, asset.blob))?))
    }
}

/// What makes one page: its background, then its layers in the old order,
/// first drawn first.
fn page_ops(scene: &Scene, page: Id, index: usize) -> Result<Vec<Op>> {
    let info: PageInfo = scene
        .get(page, PAGE)?
        .context("a page has no page values")?;
    let mut writer = Writer {
        scene,
        ops: Vec::new(),
        held: HashMap::new(),
        added: 0,
        texts_placed: false,
    };
    match scene.asset(page, "source")? {
        Some(bytes) => {
            let file = Hash::of(&bytes);
            writer.ops.push(Op::AddFile { bytes });
            writer.push(
                json!({"op": "add_page", "id": "$page", "index": index, "background": file}),
            )?;
        }
        None => writer.push(json!({
            "op": "add_page", "id": "$page", "index": index,
            "size": {"width": info.width.round() as u32, "height": info.height.round() as u32},
        }))?,
    }
    // A chapter is named where it starts; the old page only marked the start.
    writer.push(json!({
        "op": "update_page", "page": "$page",
        "patch": {"name": info.label, "chapter": info.chapter_start.then_some(&info.label)},
    }))?;
    writer.layers(page, None)?;
    if !writer.texts_placed {
        writer.texts(page, None)?;
    }
    Ok(writer.ops)
}

struct Writer<'a> {
    scene: &'a Scene,
    ops: Vec<Op>,
    /// How many layers each parent, or the page for `None`, holds so far:
    /// the next one's index, on top.
    held: HashMap<Option<String>, usize>,
    added: usize,
    /// Whether the page's panels, balloons, and blocks are in: where its
    /// first text group was, or on top of it all.
    texts_placed: bool,
}

impl Writer<'_> {
    fn push(&mut self, op: Value) -> Result<()> {
        self.ops.push(serde_json::from_value(op)?);
        Ok(())
    }

    /// Adds `layer` on top under `parent`, or at the page's top level, and
    /// returns its label.
    fn add(&mut self, parent: Option<&str>, layer: Value) -> Result<String> {
        self.added += 1;
        let label = format!("$l{}", self.added);
        let held = self.held.entry(parent.map(str::to_owned)).or_default();
        let index = *held;
        *held += 1;
        self.push(json!({
            "op": "add_layer", "id": label, "page": "$page", "parent": parent,
            "index": index, "layer": layer,
        }))?;
        Ok(label)
    }

    fn visibility(&self, id: Id) -> Result<(bool, f32)> {
        Ok(self
            .scene
            .get::<Visibility>(id, VISIBILITY)?
            .map_or((true, 1.0), |visibility| {
                (visibility.visible, visibility.opacity.clamp(0.0, 1.0))
            }))
    }

    /// The drawn children of `parent` in order: groups, rasters, and, where
    /// the first text group is, the page's texts.
    fn layers(&mut self, parent: Id, under: Option<&str>) -> Result<()> {
        for &child in self.scene.children(parent) {
            if self.scene.has(child, TEXT_GROUP) {
                if !self.texts_placed {
                    self.texts_placed = true;
                    self.texts(self.scene.page_of(child), under)?;
                }
            } else if let Some(group) = self.scene.get::<Group>(child, GROUP)? {
                let (visible, opacity) = self.visibility(child)?;
                let label = self.add(
                    under,
                    json!({"kind": "group", "name": group.name, "visible": visible, "opacity": opacity}),
                )?;
                self.layers(child, Some(&label))?;
            } else if let Some(raster) = self.scene.get::<RasterLayer>(child, RASTER)? {
                self.raster(child, &raster, under)?;
            }
        }
        Ok(())
    }

    fn raster(&mut self, id: Id, raster: &RasterLayer, under: Option<&str>) -> Result<()> {
        let (visible, opacity) = self.visibility(id)?;
        let label = self.add(
            under,
            json!({"kind": "raster", "name": raster.name, "visible": visible, "opacity": opacity}),
        )?;
        let Some(bytes) = self.scene.asset(id, "source")? else {
            return Ok(());
        };
        let image = image::load_from_memory(&bytes)
            .with_context(|| format!("failed to decode the pixels of {}", raster.name))?
            .into_rgba8();
        if image.pixels().all(|pixel| pixel[3] == 0) {
            return Ok(());
        }
        let (width, height) = image.dimensions();
        self.ops.push(Op::WritePixels {
            layer: serde_json::from_value(json!(label))?,
            rect: Rect {
                x: 0,
                y: 0,
                width,
                height,
            },
            pixels: image.into_raw(),
        });
        Ok(())
    }

    /// The page's panels, its balloons in them, and its texts in those, as
    /// the old page related them: a region inside another, and a text flowing
    /// in a balloon or recognized from a region inside one.
    fn texts(&mut self, page: Id, under: Option<&str>) -> Result<()> {
        let scene = self.scene;
        let mut kinds: Vec<(Id, String)> = Vec::new();
        let mut contents: Vec<Id> = Vec::new();
        let mut walk = vec![page];
        while let Some(id) = walk.pop() {
            walk.extend(scene.children(id).iter().rev());
            if let Some(region) = scene.get::<Region>(id, REGION)? {
                kinds.push((id, region.kind.0));
            }
            if scene.has(id, SOURCE) || scene.has(id, TRANSLATION) {
                contents.push(id);
            }
        }
        let kind_of: HashMap<Id, &str> = kinds
            .iter()
            .map(|(id, kind)| (*id, kind.as_str()))
            .collect();
        let is = |id: Id, suffix: &str| kind_of.get(&id).is_some_and(|kind| kind.ends_with(suffix));
        // The panel or balloon a region sits in.
        let container = |id: Id| {
            (scene.target(id, INSIDE))
                .filter(|&target| is(target, ".panel") || is(target, ".bubble"))
        };
        let mut labels: HashMap<Id, String> = HashMap::new();
        // Panels before balloons, so a balloon finds the panel it's in. Only
        // a panel holds a region, since a balloon holds only blocks and texts:
        // one in a balloon goes where that balloon is.
        for (suffix, kind, name) in [
            (".panel", "panel", "Panel"),
            (".bubble", "balloon", "Balloon"),
        ] {
            for &(id, _) in kinds.iter().filter(|(id, _)| is(*id, suffix)) {
                let Some(outline) = scene.points(Some(id))?.as_deref().and_then(path) else {
                    continue;
                };
                let mut parent = container(id);
                while let Some(balloon) = parent.filter(|&each| is(each, ".bubble")) {
                    parent = container(balloon);
                }
                let parent = parent.and_then(|parent| labels.get(&parent).cloned());
                let label = self.add(
                    parent.as_deref().or(under),
                    json!({"kind": kind, "name": name, "outline": outline}),
                )?;
                labels.insert(id, label);
            }
        }
        for content in contents {
            let layout = (scene.sources(content, PRESENTS).into_iter())
                .find(|&layout| scene.has(layout, TEXT_LAYOUT));
            let region = scene.target(content, RECOGNIZED_FROM);
            let holder = (layout.and_then(|layout| scene.target(layout, FLOWS_IN)))
                .or_else(|| region.and_then(container));
            let parent = holder.and_then(|holder| labels.get(&holder).cloned());
            if let Some(block) = self.block(content, layout, region, holder)? {
                self.add(parent.as_deref().or(under), block)?;
            }
        }
        Ok(())
    }

    /// A block: its source where it was recognized, its translation where
    /// the old layout set it, or where the source is without one; none for a
    /// text with no place on its page, which the old renderer didn't draw.
    fn block(
        &self,
        content: Id,
        layout: Option<Id>,
        region: Option<Id>,
        holder: Option<Id>,
    ) -> Result<Option<Value>> {
        let scene = self.scene;
        let source: Option<Words> = scene.get(content, SOURCE)?;
        let translation: Option<Words> = scene.get(content, TRANSLATION)?;
        let ocr: Option<OcrAnalysis> = match region {
            Some(region) => scene.get(region, OCR)?,
            None => None,
        };
        let recognized = scene.points(region)?;
        let own = scene.points(layout)?;
        let flowed = scene.points(holder)?;
        let fitted = scene.points(layout.and_then(|layout| scene.target(layout, FITS_TO)))?;
        let source_text = source
            .as_ref()
            .map_or("", |words| words.text.value.as_str());
        let Some((source_frame, source_transform)) = [&recognized, &flowed, &fitted, &own]
            .into_iter()
            .find_map(|points| frame(points.as_deref()?, "shrink"))
        else {
            return Ok(None);
        };
        let source_direction = match ocr.as_ref().map(|ocr| ocr.direction) {
            Some(TextDirection::Vertical) => "vertical",
            Some(TextDirection::Horizontal) => "horizontal",
            _ if cjk(source_text) => "vertical",
            _ => "horizontal",
        };
        let mut block = json!({
            "kind": "block",
            "name": "",
            "role": self.role(content, region)?,
            "source": {
                "content": source_text,
                "frame": source_frame,
                "transform": source_transform,
                "direction": source_direction,
                "language": language(source.as_ref()),
            },
        });
        if let Some(layout) = layout {
            let (visible, opacity) = self.visibility(layout)?;
            block["visible"] = json!(visible);
            block["opacity"] = json!(opacity);
        }
        let Some(translation) = translation.filter(|words| !words.text.value.trim().is_empty())
        else {
            return Ok(Some(block));
        };
        let typography: Option<Typography> = match layout {
            Some(layout) => scene.get(layout, TYPOGRAPHY)?,
            None => None,
        };
        // The old renderer fitted text unless its style said not to.
        let fit = match &typography {
            Some(typography) if !typography.auto_fit => "none",
            _ => "shrink",
        };
        // Placed as the old renderer placed it: its own geometry, then the
        // balloon it flows in, then the region it fits, then its source's.
        let (frame, transform) = [&own, &flowed, &fitted, &recognized]
            .into_iter()
            .find_map(|points| frame(points.as_deref()?, fit))
            .unwrap_or_else(|| {
                (
                    block["source"]["frame"].clone(),
                    block["source"]["transform"].clone(),
                )
            });
        let text = translation.text.value.as_str();
        let direction = direction(text, typography.as_ref(), ocr.as_ref());
        // v0.85.3-N's `resolve_alignment`: a row centred, a column from its
        // start, where the style set none.
        let align = typography
            .as_ref()
            .and_then(|typography| typography.alignment)
            .map_or(
                if direction == "vertical" {
                    "start"
                } else {
                    "center"
                },
                |align| match align {
                    TextAlignment::Start => "start",
                    TextAlignment::Center => "center",
                    TextAlignment::End => "end",
                    TextAlignment::Justify => "justify",
                },
            );
        let mut style = json!({"align": align});
        if let Some(typography) = &typography {
            let mut chars = json!({
                "family": typography.preferred_font,
                "weight": typography.font_weight,
                "italic": typography.font_style.map(|style| style != FontStyle::Normal),
                "size": typography.size,
                "fill": typography.color.map(rgba),
                "stroke": typography.stroke_color.map(rgba),
                "stroke_width": typography.stroke_width,
                "stroke_position": typography.stroke_position.map(|position| match position {
                    StrokePosition::Outside => "outside",
                    StrokePosition::Center => "center",
                    StrokePosition::Inside => "inside",
                }),
                "stroke_join": typography.stroke_join.map(|join| match join {
                    StrokeJoin::Round => "round",
                    StrokeJoin::Miter => "miter",
                    StrokeJoin::Bevel => "bevel",
                }),
                "outer_stroke": typography.outer_stroke_color.map(rgba),
                "outer_stroke_width": typography.outer_stroke_width,
            });
            if let Value::Object(map) = &mut chars {
                map.retain(|_, value| !value.is_null());
            }
            style["chars"] = chars;
            if let Some(line_spacing) = typography.line_spacing {
                style["line_height"] = json!(line_spacing);
            }
        }
        block["translation"] = json!({
            "content": text,
            "style": style,
            "frame": frame,
            "transform": transform,
            "direction": direction,
            "language": language(Some(&translation)),
        });
        Ok(Some(block))
    }

    fn role(&self, content: Id, region: Option<Id>) -> Result<&'static str> {
        let role = self
            .scene
            .get::<TextRole>(content, ROLE)?
            .map(|role| role.role);
        let label = match region {
            Some(region) => self
                .scene
                .get::<Region>(region, REGION)?
                .and_then(|region| region.label),
            None => None,
        };
        let sfx = |value: &str| value.contains("sfx") || value.contains("onomatopoeia");
        Ok(
            match role.as_deref().is_some_and(sfx) || label.as_deref().is_some_and(sfx) {
                true => "sfx",
                false => "text",
            },
        )
    }
}

/// A frame and a transform from old geometry, as the old renderer read it:
/// four points a turned box, its top edge giving its angle; more a closed
/// shape the text fills, as text flowing in a balloon did. `fit` is a box's
/// and a path's both. The text sits in the middle, as v0.85.3-N's
/// `placement` centred its block.
fn frame(points: &[stored::Point], fit: &str) -> Option<(Value, Value)> {
    if points
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        return None;
    }
    let identity = json!(Transform::default());
    match points {
        [a, b, c, _] => {
            let (top, right) = ((b.x - a.x, b.y - a.y), (c.x - b.x, c.y - b.y));
            let (width, height) = (top.0.hypot(top.1), right.0.hypot(right.1));
            if width <= f64::EPSILON || height <= f64::EPSILON {
                return None;
            }
            let center = Point::new(
                points.iter().map(|point| point.x).sum::<f64>() / 4.0,
                points.iter().map(|point| point.y).sum::<f64>() / 4.0,
            );
            let angle = top.1.atan2(top.0);
            let bounds = json!({
                "x": center.x - width / 2.0, "y": center.y - height / 2.0,
                "width": width, "height": height,
            });
            let transform = match angle.abs() < 1e-6 {
                true => identity,
                false => json!(Transform::from(Affine::rotate_about(angle, center))),
            };
            Some((
                json!({"box": {"bounds": bounds, "fit": fit, "anchor": "center"}}),
                transform,
            ))
        }
        _ => Some((
            json!({"path": {"path": path(points)?, "fit": fit, "anchor": "center"}}),
            identity,
        )),
    }
}

/// A closed SVG path through `points`, three or more.
fn path(points: &[stored::Point]) -> Option<String> {
    let (first, rest) = points.split_first()?;
    if rest.len() < 2 {
        return None;
    }
    let mut path = format!("M{} {}", first.x, first.y);
    for point in rest {
        path.push_str(&format!(" L{} {}", point.x, point.y));
    }
    path.push_str(" Z");
    Some(path)
}

fn rgba(color: [u8; 4]) -> [f32; 4] {
    color.map(|channel| f32::from(channel) / 255.0)
}

fn language(words: Option<&Words>) -> String {
    (words.and_then(|words| words.language.as_ref()))
        .map_or_else(|| "und".to_owned(), |tag| tag.0.clone())
}

/// Whether `text` has Chinese or Japanese, as v0.85.3-N's
/// `is_chinese_or_japanese_text` found it.
fn cjk(text: &str) -> bool {
    let scripts = CodePointMapData::<Script>::new();
    text.chars().any(|character| {
        matches!(
            scripts.get(character),
            Script::Han | Script::Hiragana | Script::Katakana | Script::Bopomofo
        )
    })
}

/// A translation's direction as v0.85.3-N's `resolve_writing_mode` chose it:
/// a user's writing mode, then horizontal for text with no Chinese or
/// Japanese, then a generated writing mode, then the source's direction.
fn direction(
    text: &str,
    typography: Option<&Typography>,
    ocr: Option<&OcrAnalysis>,
) -> &'static str {
    let name = |mode: WritingMode| match mode {
        WritingMode::Horizontal => "horizontal",
        WritingMode::Vertical => "vertical",
    };
    let mode = typography.and_then(|typography| typography.writing_mode);
    if let (Some(typography), Some(mode)) = (typography, mode)
        && matches!(typography.origin, Origin::User)
    {
        return name(mode);
    }
    if !cjk(text) {
        return "horizontal";
    }
    if let Some(mode) = mode {
        return name(mode);
    }
    match ocr.map(|ocr| ocr.direction) {
        Some(TextDirection::Vertical) => "vertical",
        _ => "horizontal",
    }
}
