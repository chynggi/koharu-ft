//! Versioned, backend-neutral display data.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use vello::kurbo::Affine;

use crate::{Error, Result};

const MANIFEST_MAGIC: [u8; 8] = *b"KHRMANF\0";
const RESOURCE_MAGIC: [u8; 8] = *b"KHRRSRC\0";
pub const PREPARED_FRAME_MANIFEST_VERSION: u16 = 4;
pub const PREPARED_RESOURCE_FORMAT_VERSION: u16 = 3;
/// Logical raster tile edge used by portable frames. The one-pixel sampling
/// gutter is stored outside this logical extent where adjacent pixels exist.
pub const PREPARED_RASTER_TILE_DIMENSION: u32 = 1_024;
const MAX_SURFACE_DIMENSION: u32 = 32_768;
const MAX_SURFACE_PIXELS: u64 = 268_435_456;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LayerId([u8; 16]);

impl LayerId {
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl std::fmt::Display for LayerId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Revision(u64);

impl Revision {
    pub const ZERO: Self = Self(0);

    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResourceId([u8; 32]);

impl ResourceId {
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    #[must_use]
    pub fn for_font(bytes: &[u8]) -> Self {
        Self::hash(b"koharu-prepared-font-v1\0", &[], bytes)
    }

    #[must_use]
    pub fn for_encoded_raster(width: u32, height: u32, media_type: &str, bytes: &[u8]) -> Self {
        let mut metadata = Vec::with_capacity(9 + media_type.len());
        metadata.extend_from_slice(&width.to_le_bytes());
        metadata.extend_from_slice(&height.to_le_bytes());
        metadata.extend_from_slice(media_type.as_bytes());
        metadata.push(0);
        Self::hash(b"koharu-prepared-encoded-raster-v1\0", &metadata, bytes)
    }

    #[must_use]
    pub fn for_raster_tile(source: Self, tile: PreparedRasterTile) -> Self {
        let mut metadata = [0; 56];
        metadata[..32].copy_from_slice(source.as_bytes());
        metadata[32..36].copy_from_slice(&tile.x.to_le_bytes());
        metadata[36..40].copy_from_slice(&tile.y.to_le_bytes());
        metadata[40..44].copy_from_slice(&tile.width.to_le_bytes());
        metadata[44..48].copy_from_slice(&tile.height.to_le_bytes());
        for (index, gutter) in tile.gutter.into_iter().enumerate() {
            metadata[48 + index * 2..50 + index * 2]
                .copy_from_slice(&(gutter as u16).to_le_bytes());
        }
        Self::hash(b"koharu-prepared-raster-tile-v1\0", &metadata, &[])
    }

    fn hash(domain: &[u8], metadata: &[u8], bytes: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(domain);
        hasher.update(metadata);
        hasher.update(bytes);
        Self(*hasher.finalize().as_bytes())
    }
}

impl std::fmt::Display for ResourceId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl std::str::FromStr for ResourceId {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        if value.len() != 64 {
            return Err(Error::invalid(
                "resource id must contain exactly 64 hexadecimal characters",
            ));
        }
        let mut bytes = [0; 32];
        for (index, pair) in value.as_bytes().as_chunks::<2>().0.iter().enumerate() {
            let high = hex_nibble(pair[0])?;
            let low = hex_nibble(pair[1])?;
            bytes[index] = high << 4 | low;
        }
        Ok(Self(bytes))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedFrameBundle {
    pub frame: PreparedFrame,
    pub resources: Vec<PreparedResource>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedFrameManifest {
    pub frame: PreparedFrame,
    pub resources: Vec<PreparedResourceRef>,
}

#[derive(Serialize)]
struct ManifestPacketRef<'a> {
    magic: [u8; 8],
    version: u16,
    manifest: &'a PreparedFrameManifest,
}

#[derive(Deserialize, Serialize)]
struct ManifestPacket {
    magic: [u8; 8],
    version: u16,
    manifest: PreparedFrameManifest,
}

#[derive(Serialize)]
struct ResourcePacketRef<'a> {
    magic: [u8; 8],
    version: u16,
    resource: &'a PreparedResource,
}

#[derive(Deserialize, Serialize)]
struct ResourcePacket {
    magic: [u8; 8],
    version: u16,
    resource: PreparedResource,
}

/// One independently validated, content-addressed resource packet.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedResourcePacket {
    resource: PreparedResource,
}

#[derive(Debug, Default)]
pub struct PreparedResourceStore {
    resources: HashMap<ResourceId, PreparedResourcePacket>,
    total_bytes: u64,
}

impl PreparedFrameBundle {
    pub fn manifest(&self) -> Result<PreparedFrameManifest> {
        let manifest = PreparedFrameManifest {
            frame: self.frame.clone(),
            resources: self
                .resources
                .iter()
                .map(PreparedResource::reference)
                .collect(),
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn resource_packet(&self, id: ResourceId) -> Result<PreparedResourcePacket> {
        let resource = self
            .resources
            .iter()
            .find(|resource| resource.id() == id)
            .cloned()
            .ok_or_else(|| Error::invalid(format!("prepared frame has no resource {id}")))?;
        PreparedResourcePacket::from_resource(resource)
    }

    pub fn validate(&self) -> Result<()> {
        for resource in &self.resources {
            resource.validate()?;
        }
        self.manifest().map(|_| ())
    }
}

impl PreparedFrameManifest {
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        postcard::to_stdvec(&ManifestPacketRef {
            magic: MANIFEST_MAGIC,
            version: PREPARED_FRAME_MANIFEST_VERSION,
            manifest: self,
        })
        .map_err(Error::Encode)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let packet: ManifestPacket = postcard::from_bytes(bytes).map_err(Error::Decode)?;
        if packet.magic != MANIFEST_MAGIC {
            return Err(Error::invalid(
                "prepared frame manifest magic does not match",
            ));
        }
        if packet.version != PREPARED_FRAME_MANIFEST_VERSION {
            return Err(Error::UnsupportedManifestVersion(packet.version));
        }
        packet.manifest.validate()?;
        Ok(packet.manifest)
    }

    #[must_use]
    pub fn required_resources(&self) -> &[PreparedResourceRef] {
        &self.resources
    }

    #[must_use]
    pub fn missing_resources(&self, store: &PreparedResourceStore) -> Vec<ResourceId> {
        store.missing_resources(self)
    }

    pub fn compile(&self, store: &PreparedResourceStore) -> Result<crate::Frame> {
        store.compile(self)
    }

    pub fn validate(&self) -> Result<()> {
        validate_frame(&self.frame, &self.resources)
    }
}

impl PreparedResourcePacket {
    pub fn from_resource(resource: PreparedResource) -> Result<Self> {
        resource.validate()?;
        Ok(Self { resource })
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        postcard::to_stdvec(&ResourcePacketRef {
            magic: RESOURCE_MAGIC,
            version: PREPARED_RESOURCE_FORMAT_VERSION,
            resource: &self.resource,
        })
        .map_err(Error::Encode)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let packet: ResourcePacket = postcard::from_bytes(bytes).map_err(Error::Decode)?;
        if packet.magic != RESOURCE_MAGIC {
            return Err(Error::invalid("prepared resource magic does not match"));
        }
        if packet.version != PREPARED_RESOURCE_FORMAT_VERSION {
            return Err(Error::UnsupportedResourceVersion(packet.version));
        }
        Self::from_resource(packet.resource)
    }

    #[must_use]
    pub const fn id(&self) -> ResourceId {
        self.resource.id()
    }

    #[must_use]
    pub fn kind(&self) -> PreparedResourceKind {
        self.resource.kind()
    }

    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.resource.byte_len()
    }

    #[must_use]
    pub fn resource(&self) -> &PreparedResource {
        &self.resource
    }
}

impl PreparedResourceStore {
    pub fn insert(&mut self, packet: PreparedResourcePacket) -> Option<PreparedResourcePacket> {
        let id = packet.id();
        let previous = self.resources.insert(id, packet);
        self.total_bytes = self
            .total_bytes
            .saturating_add(self.resources[&id].byte_len());
        if let Some(previous) = &previous {
            self.total_bytes = self.total_bytes.saturating_sub(previous.byte_len());
        }
        previous
    }

    pub fn insert_packet(&mut self, bytes: &[u8]) -> Result<ResourceId> {
        let packet = PreparedResourcePacket::decode(bytes)?;
        let id = packet.id();
        self.insert(packet);
        Ok(id)
    }

    #[must_use]
    pub fn contains(&self, id: ResourceId) -> bool {
        self.resources.contains_key(&id)
    }

    #[must_use]
    pub fn get(&self, id: ResourceId) -> Option<&PreparedResourcePacket> {
        self.resources.get(&id)
    }

    pub fn remove(&mut self, id: ResourceId) -> Option<PreparedResourcePacket> {
        let removed = self.resources.remove(&id)?;
        self.total_bytes = self.total_bytes.saturating_sub(removed.byte_len());
        Some(removed)
    }

    pub fn clear(&mut self) {
        self.resources.clear();
        self.total_bytes = 0;
    }

    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    #[must_use]
    pub fn missing_resources(&self, manifest: &PreparedFrameManifest) -> Vec<ResourceId> {
        manifest
            .required_resources()
            .iter()
            .filter_map(|resource| (!self.contains(resource.id)).then_some(resource.id))
            .collect()
    }

    pub fn compile(&self, manifest: &PreparedFrameManifest) -> Result<crate::Frame> {
        manifest.validate()?;
        crate::frame::compile_manifest(manifest, self)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedFrame {
    pub revision: Revision,
    pub page: LayerId,
    pub width: u32,
    pub height: u32,
    pub origin: (i32, i32),
    /// Kurbo affine coefficients `[a, b, c, d, e, f]`.
    pub normalization: [f64; 6],
    pub layers: Vec<PreparedLayer>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedLayer {
    pub id: LayerId,
    pub geometry: Vec<Point>,
    pub bounds: Bounds,
    pub local_bounds: Bounds,
    pub presentation: Presentation,
    pub kind: LayerKind,
    /// Kurbo affine coefficients `[a, b, c, d, e, f]`.
    pub placement: [f64; 6],
    pub content: PreparedContent,
    pub element_frame: Option<PreparedElementFrame>,
}

impl PreparedLayer {
    #[must_use]
    pub const fn element_frame(&self) -> Option<PreparedElementFrame> {
        self.element_frame
    }

    fn validate(
        &self,
        resources: &HashMap<ResourceId, &PreparedResourceRef>,
        referenced: &mut HashSet<ResourceId>,
    ) -> Result<()> {
        if self.geometry.len() < 3
            || self
                .geometry
                .iter()
                .any(|point| !point.x.is_finite() || !point.y.is_finite())
        {
            return Err(Error::invalid("layer geometry is invalid"));
        }
        finite_bounds(self.bounds, "layer bounds")?;
        finite_bounds(self.local_bounds, "layer local bounds")?;
        finite_affine(self.placement, "layer placement")?;
        if !self.presentation.opacity.is_finite()
            || !(0.0..=1.0).contains(&self.presentation.opacity)
        {
            return Err(Error::invalid("layer opacity is outside zero through one"));
        }
        if self.element_frame.is_some_and(|frame| !frame.is_valid()) {
            return Err(Error::invalid("layer element frame is invalid"));
        }
        match (&self.kind, &self.content) {
            (LayerKind::Raster, PreparedContent::Raster(raster)) => {
                raster.validate(resources, referenced)?;
            }
            (LayerKind::Text, PreparedContent::Vector(scene)) => {
                scene.validate(resources, referenced)?;
            }
            // A shape's content is a vector scene exactly like text's: the
            // renderer draws the prepared path the shape parameters produce.
            (LayerKind::Shape, PreparedContent::Vector(scene)) => {
                scene.validate(resources, referenced)?;
            }
            _ => {
                return Err(Error::invalid(
                    "prepared layer kind and content do not match",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LayerKind {
    Raster,
    Text,
    Shape,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Presentation {
    pub visible: bool,
    pub opacity: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedElementFrame {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub angle_degrees: f32,
}

impl PreparedElementFrame {
    #[must_use]
    pub fn is_valid(self) -> bool {
        [self.x, self.y, self.width, self.height, self.angle_degrees]
            .into_iter()
            .all(f32::is_finite)
            && self.width > 0.0
            && self.height > 0.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PreparedContent {
    Raster(PreparedRaster),
    Vector(PreparedScene),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedRaster {
    pub source: ResourceId,
    pub width: u32,
    pub height: u32,
    pub tiles: Vec<PreparedRasterTile>,
}

impl PreparedRaster {
    fn validate(
        &self,
        resources: &HashMap<ResourceId, &PreparedResourceRef>,
        referenced: &mut HashSet<ResourceId>,
    ) -> Result<()> {
        validate_surface(self.width, self.height)?;
        let Some(PreparedResourceRef {
            kind:
                PreparedResourceKind::EncodedRaster {
                    width: source_width,
                    height: source_height,
                },
            ..
        }) = resources.get(&self.source).copied()
        else {
            return Err(Error::invalid(
                "raster references a missing encoded source resource",
            ));
        };
        if (*source_width, *source_height) != (self.width, self.height) {
            return Err(Error::invalid(
                "encoded raster source dimensions do not match the raster",
            ));
        }
        referenced.insert(self.source);
        let columns = self.width.div_ceil(PREPARED_RASTER_TILE_DIMENSION);
        let rows = self.height.div_ceil(PREPARED_RASTER_TILE_DIMENSION);
        if usize::try_from(u64::from(columns) * u64::from(rows)).ok() != Some(self.tiles.len()) {
            return Err(Error::invalid("raster tile count does not cover the image"));
        }
        for (index, tile) in self.tiles.iter().enumerate() {
            let index = u32::try_from(index)
                .map_err(|_| Error::invalid("raster tile index exceeds u32"))?;
            let column = index % columns;
            let row = index / columns;
            let x = column * PREPARED_RASTER_TILE_DIMENSION;
            let y = row * PREPARED_RASTER_TILE_DIMENSION;
            let width = (self.width - x).min(PREPARED_RASTER_TILE_DIMENSION);
            let height = (self.height - y).min(PREPARED_RASTER_TILE_DIMENSION);
            let gutter = [
                u32::from(x > 0),
                u32::from(y > 0),
                u32::from(x + width < self.width),
                u32::from(y + height < self.height),
            ];
            if (tile.x, tile.y, tile.width, tile.height, tile.gutter)
                != (x, y, width, height, gutter)
            {
                return Err(Error::invalid("raster tiles are not in canonical order"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PreparedRasterTile {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Left, top, right, and bottom pixels retained for filtered sampling.
    pub gutter: [u32; 4],
}

impl PreparedRasterTile {
    #[must_use]
    pub fn id(self, source: ResourceId) -> ResourceId {
        ResourceId::for_raster_tile(source, self)
    }

    #[must_use]
    pub const fn source_origin(self) -> (u32, u32) {
        (self.x - self.gutter[0], self.y - self.gutter[1])
    }

    #[must_use]
    pub const fn resource_size(self) -> (u32, u32) {
        (
            self.width + self.gutter[0] + self.gutter[2],
            self.height + self.gutter[1] + self.gutter[3],
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PreparedScene {
    pub commands: Vec<PreparedSceneCommand>,
}

impl PreparedScene {
    fn validate(
        &self,
        resources: &HashMap<ResourceId, &PreparedResourceRef>,
        referenced: &mut HashSet<ResourceId>,
    ) -> Result<()> {
        for command in &self.commands {
            match command {
                PreparedSceneCommand::GlyphRun(run) => {
                    let Some(PreparedResourceRef {
                        kind: PreparedResourceKind::Font,
                        ..
                    }) = resources.get(&run.font).copied()
                    else {
                        return Err(Error::invalid(
                            "glyph run references a missing font resource",
                        ));
                    };
                    referenced.insert(run.font);
                    run.validate()?;
                }
                PreparedSceneCommand::FillPath(path) => path.validate()?,
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PreparedSceneCommand {
    GlyphRun(PreparedGlyphRun),
    FillPath(PreparedPath),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedGlyphRun {
    pub font: ResourceId,
    pub font_index: u32,
    pub font_size: f32,
    pub normalized_coords: Vec<i16>,
    pub transform: [f64; 6],
    pub glyph_transform: Option<[f64; 6]>,
    pub hint: bool,
    pub embolden: [f32; 2],
    pub color: [u8; 4],
    pub glyphs: Vec<PreparedGlyph>,
}

impl PreparedGlyphRun {
    fn validate(&self) -> Result<()> {
        if !self.font_size.is_finite() || self.font_size <= 0.0 {
            return Err(Error::invalid("glyph run font size is invalid"));
        }
        finite_affine(self.transform, "glyph run transform")?;
        if let Some(transform) = self.glyph_transform {
            finite_affine(transform, "glyph transform")?;
        }
        if self
            .embolden
            .iter()
            .any(|strength| !strength.is_finite() || *strength < 0.0)
        {
            return Err(Error::invalid("glyph embolden strength is invalid"));
        }
        if self
            .glyphs
            .iter()
            .any(|glyph| !glyph.x.is_finite() || !glyph.y.is_finite())
        {
            return Err(Error::invalid("glyph run contains a non-finite position"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedGlyph {
    pub id: u32,
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedPath {
    pub transform: [f64; 6],
    pub elements: Vec<PathElement>,
    pub fill: Option<PreparedFill>,
    pub stroke: Option<PreparedStroke>,
}

impl PreparedPath {
    /// Public because the renderer builds prepared paths and must be able to
    /// check that what it produced is drawable before handing it to a frame.
    pub fn validate(&self) -> Result<()> {
        finite_affine(self.transform, "path transform")?;
        if self.elements.is_empty() {
            return Err(Error::invalid("prepared path is empty"));
        }
        for element in &self.elements {
            let finite = match element {
                PathElement::MoveTo(values) | PathElement::LineTo(values) => {
                    values.iter().all(|value| value.is_finite())
                }
                PathElement::QuadTo(values) => values.iter().all(|value| value.is_finite()),
                PathElement::CurveTo(values) => values.iter().all(|value| value.is_finite()),
                PathElement::Close => true,
            };
            if !finite {
                return Err(Error::invalid("prepared path contains a non-finite point"));
            }
        }
        if let Some(fill) = &self.fill {
            match fill {
                PreparedFill::Gradient { gradient, .. } => gradient.validate()?,
                PreparedFill::Pattern { pattern, .. } => pattern.validate()?,
                PreparedFill::Solid { .. } => {}
            }
        }
        if let Some(stroke) = &self.stroke {
            if !stroke.width.is_finite() || stroke.width < 0.0 {
                return Err(Error::invalid(
                    "prepared path stroke width must be finite and non-negative",
                ));
            }
            if let Some(dash) = &stroke.dash
                && (dash.is_empty()
                    || dash
                        .iter()
                        .any(|length| !length.is_finite() || *length <= 0.0))
            {
                return Err(Error::invalid(
                    "prepared path dash lengths must be finite and positive",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PreparedFill {
    Solid {
        color: [u8; 4],
        rule: FillRule,
    },
    Gradient {
        gradient: PreparedGradient,
        rule: FillRule,
    },
    Pattern {
        pattern: PreparedPattern,
        rule: FillRule,
    },
}

/// How a [`PreparedPattern`] lays its image out across the shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PreparedPatternRepeat {
    Tile,
    Stretch,
}

/// A pattern ready for a renderer, with its coordinates in the **path's local
/// coordinate system** — the same one [`PreparedPath::elements`] uses.
///
/// Like [`PreparedGradient`], the coordinates are never page coordinates and
/// never normalized frame coordinates, so [`PreparedPath::transform`] applies to
/// the pattern and the outline alike and a rotated shape's pattern rotates with
/// it. The normalized-to-local spread happens before a path is built.
///
/// Unlike a gradient, a pattern is an *image*: no renderer can paint it from
/// numbers alone. [`PreparedPattern::image`] carries the decoded pixels when a
/// caller has resolved the owning layer's `pattern` asset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedPattern {
    pub repeat: PreparedPatternRepeat,
    /// Tile origin in path-local coordinates.
    ///
    /// **Ignored when `repeat` is [`PreparedPatternRepeat::Stretch`]**, which
    /// fits exactly one image to [`PreparedPattern::extent`].
    pub origin: [f32; 2],
    /// Tile pitch in path-local units: the width one tile occupies. Finite and
    /// positive.
    ///
    /// **Ignored when `repeat` is [`PreparedPatternRepeat::Stretch`].**
    pub scale: f32,
    /// Clockwise degrees applied about the tile origin (`Tile`) or the box
    /// centre (`Stretch`).
    ///
    /// This is the *pattern's* own rotation, which the author sets independently
    /// of the shape's angle. The shape's angle reaches the brush through
    /// [`PreparedPath::transform`] and must not be folded in here, or a rotated
    /// shape would rotate its pattern twice.
    pub rotation_degrees: i32,
    /// Drawn as a solid fill when [`PreparedPattern::image`] is `None`.
    pub fallback: [u8; 4],
    /// The shape's box in path-local coordinates: `(0, 0)` to this extent.
    ///
    /// `Stretch` needs it to fit one image to the shape; `Tile` ignores it.
    pub extent: [f32; 2],
    /// The decoded pattern pixels, or `None` when no `pattern` asset has been
    /// resolved for the layer that owns this fill.
    ///
    /// `koharu-scene`'s `PatternSpec` deliberately references no asset (the
    /// shape layer owns it, so a deleted layer cannot leave a dangling
    /// reference), which means the asset must be resolved per layer and handed
    /// down here. The pixels are carried rather than only their [`ResourceId`]
    /// because [`PreparedPath::append_to`] — the one place that assembles the
    /// brush — is given no resource registry to look an id up in.
    pub image: Option<PreparedPatternImage>,
}

/// Decoded, straight-alpha RGBA8 pixels for a [`PreparedPattern`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedPatternImage {
    /// Identity of the source asset, so a caller can tell two look-alike
    /// patterns apart and cache an upload by content.
    pub source: ResourceId,
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes of RGBA8 with straight (unpremultiplied)
    /// alpha.
    pub pixels: Arc<[u8]>,
}

impl PreparedPattern {
    /// The mapping from the pattern image's own pixel space to this path's
    /// local space, as Vello's `brush_transform` argument.
    ///
    /// This is **not** optional and deliberately not `None`. Vello composes the
    /// path transform with this one (`transform * brush_transform`,
    /// `vello-0.10.0/src/scene.rs:328-331`) and the fine rasterizer inverts the
    /// result to index the image (`vello-0.10.0`'s `draw_leaf.wgsl:250`), then
    /// wraps that coordinate with `extend_mode(..., image.extents)`
    /// (`fine.wgsl:1326`) — so the composed transform maps **image pixels** to
    /// page space. With an identity brush transform one image pixel would equal
    /// one path-local unit and [`PreparedPattern::scale`] could not affect the
    /// tile pitch at all.
    ///
    /// Passing the *path* affine here instead would double-apply it, because the
    /// path transform is already the composition's other factor. That was 1b's
    /// Critical bug; it is a different mistake from passing a brush transform.
    #[must_use]
    pub fn brush_transform(&self) -> Affine {
        let Some(image) = self.image.as_ref() else {
            // No pixels, so no brush will be built. The identity keeps the
            // caller's single `fill` call uniform rather than branching here.
            return Affine::IDENTITY;
        };
        let degrees = f64::from(self.rotation_degrees);
        match self.repeat {
            PreparedPatternRepeat::Tile => {
                // Image pixel (0,0) lands on the tile origin, and one pixel
                // spans `scale` path-local units, so `scale` is the pitch the
                // author set. Applied scale -> rotate -> translate.
                Affine::translate((
                    f64::from(self.origin[0]),
                    f64::from(self.origin[1]),
                )) * Affine::rotate(degrees.to_radians())
                    * Affine::scale(f64::from(self.scale))
            }
            PreparedPatternRepeat::Stretch => {
                // One image fills the whole box, so the scale factors are the
                // box/image ratios per axis. Non-uniform on purpose: stretching
                // to the box is what this mode promises, and preserving the
                // image's aspect ratio would leave part of the box unfilled.
                //
                // The rotation is taken about the box centre rather than the
                // origin so a stretched pattern turns in place.
                let width = f64::from(self.extent[0]);
                let height = f64::from(self.extent[1]);
                let center = (width * 0.5, height * 0.5);
                Affine::translate(center)
                    * Affine::rotate(degrees.to_radians())
                    * Affine::translate((-center.0, -center.1))
                    * Affine::scale_non_uniform(
                        width / f64::from(image.width.max(1)),
                        height / f64::from(image.height.max(1)),
                    )
            }
        }
    }

    /// Boundary validation for prepared patterns.
    ///
    /// `koharu-scene` validates the persisted `PatternSpec` with the same rules,
    /// and this crate deliberately does **not** depend on that crate: the
    /// rasterizer must stay usable from a backend that has never seen a scene.
    /// Re-checking here is boundary validation, not duplication.
    fn validate(&self) -> Result<()> {
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(Error::invalid(
                "prepared pattern scale must be finite and positive",
            ));
        }
        if self.origin.iter().any(|value| !value.is_finite()) {
            return Err(Error::invalid("prepared pattern origin must be finite"));
        }
        if !(-360..=360).contains(&self.rotation_degrees) {
            return Err(Error::invalid(
                "prepared pattern rotation must be within -360..=360",
            ));
        }
        if self
            .extent
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(Error::invalid(
                "prepared pattern extent must be finite and non-negative",
            ));
        }
        if let Some(image) = &self.image {
            validate_surface(image.width, image.height)?;
            let expected = usize::try_from(
                u64::from(image.width)
                    .checked_mul(u64::from(image.height))
                    .and_then(|pixels| pixels.checked_mul(4))
                    .ok_or_else(|| Error::invalid("pattern image size overflows"))?,
            )
            .map_err(|_| Error::invalid("pattern image size exceeds usize"))?;
            if image.pixels.len() != expected {
                return Err(Error::invalid(
                    "pattern image byte length does not match its dimensions",
                ));
            }
        }
        Ok(())
    }
}

/// A gradient ready for a renderer, with its control points in the **path's
/// local coordinate system** — the same one [`PreparedPath::elements`] uses.
///
/// They are never page coordinates and never normalized frame coordinates.
/// Keeping the gradient in the path's own space means [`PreparedPath::transform`]
/// applies to the gradient and to the outline alike, so a rotated shape's
/// gradient rotates with it without anyone hand-transforming the control points.
/// The normalized-to-local spread happens before a path is built.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedGradient {
    pub kind: PreparedGradientKind,
    /// May be in any order; renderers sort stably by offset before assembling.
    pub stops: Vec<PreparedGradientStop>,
    pub extend: PreparedGradientExtend,
}

impl PreparedGradient {
    /// Boundary validation for prepared gradients.
    ///
    /// `koharu-scene` validates the persisted `GradientSpec` with the same
    /// rules, and this crate deliberately does **not** depend on that crate: the
    /// rasterizer must stay usable from a backend that has never seen a scene.
    /// Re-checking the rules here is therefore boundary validation, not
    /// duplication — a prepared gradient can arrive from the WASM canvas or a
    /// native caller that never built one from `GradientSpec` at all.
    fn validate(&self) -> Result<()> {
        if !(2..=32).contains(&self.stops.len()) {
            return Err(Error::invalid(
                "prepared gradient must have between 2 and 32 stops",
            ));
        }
        for stop in &self.stops {
            if !stop.offset.is_finite() || !(0.0..=1.0).contains(&stop.offset) {
                return Err(Error::invalid(
                    "prepared gradient stop offset must be finite and within 0.0..=1.0",
                ));
            }
        }
        for (index, stop) in self.stops.iter().enumerate() {
            if self.stops[index + 1..]
                .iter()
                .any(|other| other.offset == stop.offset)
            {
                return Err(Error::invalid(
                    "prepared gradient stop offsets must be unique",
                ));
            }
        }
        match &self.kind {
            PreparedGradientKind::Linear { start, end } => {
                if start.iter().chain(end).any(|value| !value.is_finite()) {
                    return Err(Error::invalid("prepared gradient endpoints must be finite"));
                }
                if start == end {
                    return Err(Error::invalid(
                        "prepared gradient linear endpoints must differ",
                    ));
                }
            }
            PreparedGradientKind::Radial { center, radius } => {
                if center.iter().any(|value| !value.is_finite()) {
                    return Err(Error::invalid("prepared gradient center must be finite"));
                }
                if !radius.is_finite() || *radius <= 0.0 {
                    return Err(Error::invalid(
                        "prepared gradient radial radius must be finite and positive",
                    ));
                }
            }
            PreparedGradientKind::Sweep {
                center,
                start_degrees,
                end_degrees,
            } => {
                if center.iter().any(|value| !value.is_finite()) {
                    return Err(Error::invalid("prepared gradient center must be finite"));
                }
                if !start_degrees.is_finite() || !end_degrees.is_finite() {
                    return Err(Error::invalid(
                        "prepared gradient sweep angles must be finite",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PreparedGradientKind {
    Linear {
        start: [f32; 2],
        end: [f32; 2],
    },
    Radial {
        center: [f32; 2],
        radius: f32,
    },
    Sweep {
        center: [f32; 2],
        /// Degrees, measured from the positive X axis. The renderer converts
        /// these to radians for peniko; the sign convention is peniko's own, so
        /// the angle is passed through without a flip.
        start_degrees: f32,
        end_degrees: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedGradientStop {
    /// `0.0..=1.0`.
    pub offset: f32,
    pub color: [u8; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PreparedGradientExtend {
    Pad,
    Repeat,
    Reflect,
}

/// Stably sorts stops by offset. Unsorted input is legal, so renderers cannot
/// reject it; they normalize it here instead.
///
/// A public helper rather than an inline sort because the stable ordering is a
/// contract the design spec pins, and a test can only pin it if the sort is
/// reachable on its own.
pub fn stops_sorted(stops: &[PreparedGradientStop]) -> Vec<PreparedGradientStop> {
    let mut sorted = stops.to_vec();
    // `total_cmp` is a total order, so sorting is deterministic even for values
    // validation has not yet rejected (for example `NaN`).
    sorted.sort_by(|left, right| left.offset.total_cmp(&right.offset));
    sorted
}

/// Which page-normalized axis a [`BrushRamp::FixedAxis`] ramp runs along.
///
/// Page-normalized coordinates (not frame-local ones) let several strokes share
/// one ramp, so the axis is what the caller picks instead of a pair of points.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RampAxis {
    Horizontal,
    Vertical,
}

/// A two-colour brush ramp: the brush paints `from` at the low end of the ramp
/// and `to` at the high end.
///
/// This is deliberately not [`PreparedGradient`]: a gradient's control points
/// are frame-local and its stops are open-ended, while a ramp is exactly two
/// colours plus a rule for turning a position into a parameter. Reusing the
/// gradient type would store `kind`/`stops`/`extend` fields that mean nothing
/// for a brush.
///
/// It lives in this crate because both the native commit path and the WASM live
/// preview interpolate through [`BrushRamp::color_at`], so the two cannot drift
/// apart. It is a command argument rather than a persisted component, so it is
/// internally tagged (`kind`) like this workspace's command arguments.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BrushRamp {
    /// Runs along the stroke: the stroke start is `from`, the stroke end is `to`.
    AlongStroke { from: [u8; 4], to: [u8; 4] },
    /// Runs along a fixed page axis: the `0.0` side of `axis` is `from`, the
    /// `1.0` side is `to`. Page-normalized coordinates let several strokes
    /// continue one ramp.
    FixedAxis {
        from: [u8; 4],
        to: [u8; 4],
        axis: RampAxis,
    },
}

impl BrushRamp {
    /// The colour at `position`, with `position` clamped to `0.0..=1.0`.
    ///
    /// All four channels interpolate, alpha included: the ramp's alpha
    /// participates in stroke compositing, so a ramp that lerped only RGB would
    /// silently drop transparency.
    pub fn color_at(&self, position: f32) -> [u8; 4] {
        let (from, to) = match self {
            Self::AlongStroke { from, to } | Self::FixedAxis { from, to, .. } => (from, to),
        };
        // Clamp before the cast: a NaN position survives a comparison-based
        // clamp (every comparison with NaN is false) and would then produce a
        // garbage channel via `as u8`. Rounding can likewise overshoot past
        // 255.0, so the value is clamped once more after interpolation.
        let t = if position.is_nan() {
            0.0
        } else {
            position.clamp(0.0, 1.0)
        };
        let mut out = [0u8; 4];
        for (index, channel) in out.iter_mut().enumerate() {
            let start = f32::from(from[index]);
            let end = f32::from(to[index]);
            *channel = (start + (end - start) * t).round().clamp(0.0, 255.0) as u8;
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedStroke {
    pub color: [u8; 4],
    pub width: f32,
    /// On/off lengths in path units; when present, every element must be finite and positive.
    pub dash: Option<Vec<f32>>,
    pub cap: PreparedLineCap,
    pub join: PreparedLineJoin,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PreparedLineCap {
    Butt,
    Round,
    Square,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PreparedLineJoin {
    Miter,
    Round,
    Bevel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PathElement {
    MoveTo([f64; 2]),
    LineTo([f64; 2]),
    QuadTo([f64; 4]),
    CurveTo([f64; 6]),
    Close,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PreparedResourceKind {
    Font,
    EncodedRaster { width: u32, height: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparedResourceRef {
    pub id: ResourceId,
    pub kind: PreparedResourceKind,
    pub byte_len: u64,
}

impl PreparedResourceRef {
    fn validate(self) -> Result<()> {
        match self.kind {
            PreparedResourceKind::Font if self.byte_len == 0 => {
                Err(Error::invalid("font resource reference is empty"))
            }
            PreparedResourceKind::Font => Ok(()),
            PreparedResourceKind::EncodedRaster { width, height } => {
                validate_surface(width, height)?;
                if self.byte_len == 0 {
                    Err(Error::invalid("encoded raster resource reference is empty"))
                } else {
                    Ok(())
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PreparedResource {
    Font {
        id: ResourceId,
        bytes: Arc<[u8]>,
    },
    EncodedRaster {
        id: ResourceId,
        width: u32,
        height: u32,
        media_type: String,
        bytes: Arc<[u8]>,
    },
}

impl PreparedResource {
    #[must_use]
    pub const fn id(&self) -> ResourceId {
        match self {
            Self::Font { id, .. } | Self::EncodedRaster { id, .. } => *id,
        }
    }

    #[must_use]
    pub const fn kind(&self) -> PreparedResourceKind {
        match self {
            Self::Font { .. } => PreparedResourceKind::Font,
            Self::EncodedRaster { width, height, .. } => PreparedResourceKind::EncodedRaster {
                width: *width,
                height: *height,
            },
        }
    }

    #[must_use]
    pub fn byte_len(&self) -> u64 {
        let length = match self {
            Self::Font { bytes, .. } => bytes.len(),
            Self::EncodedRaster { bytes, .. } => bytes.len(),
        };
        u64::try_from(length).unwrap_or(u64::MAX)
    }

    #[must_use]
    pub fn reference(&self) -> PreparedResourceRef {
        PreparedResourceRef {
            id: self.id(),
            kind: self.kind(),
            byte_len: self.byte_len(),
        }
    }

    #[must_use]
    pub fn font(bytes: &[u8]) -> Self {
        Self::font_shared(Arc::from(bytes))
    }

    #[must_use]
    pub fn font_shared(bytes: Arc<[u8]>) -> Self {
        Self::Font {
            id: ResourceId::for_font(&bytes),
            bytes,
        }
    }

    pub fn encoded_raster(
        width: u32,
        height: u32,
        media_type: impl Into<String>,
        bytes: Arc<[u8]>,
    ) -> Result<Self> {
        let media_type = media_type.into();
        let resource = Self::EncodedRaster {
            id: ResourceId::for_encoded_raster(width, height, &media_type, &bytes),
            width,
            height,
            media_type,
            bytes,
        };
        resource.validate()?;
        Ok(resource)
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::Font { id, bytes } => {
                if bytes.is_empty() {
                    return Err(Error::invalid("font resource is empty"));
                }
                if *id != ResourceId::for_font(bytes) {
                    return Err(Error::invalid("font resource content id does not match"));
                }
            }
            Self::EncodedRaster {
                id,
                width,
                height,
                media_type,
                bytes,
            } => {
                validate_surface(*width, *height)?;
                if bytes.is_empty() {
                    return Err(Error::invalid("encoded raster resource is empty"));
                }
                if media_type.is_empty()
                    || media_type.len() > 255
                    || !media_type.contains('/')
                    || media_type.contains('\0')
                {
                    return Err(Error::invalid("encoded raster media type is invalid"));
                }
                if *id != ResourceId::for_encoded_raster(*width, *height, media_type, bytes) {
                    return Err(Error::invalid(
                        "encoded raster resource content id does not match",
                    ));
                }
            }
        }
        Ok(())
    }
}

fn validate_frame(frame: &PreparedFrame, resource_refs: &[PreparedResourceRef]) -> Result<()> {
    validate_surface(frame.width, frame.height)?;
    finite_affine(frame.normalization, "frame normalization")?;
    let mut resources = HashMap::with_capacity(resource_refs.len());
    for resource in resource_refs {
        resource.validate()?;
        if resources.insert(resource.id, resource).is_some() {
            return Err(Error::invalid(
                "prepared frame manifest repeats a resource id",
            ));
        }
    }
    let mut layers = HashSet::with_capacity(frame.layers.len());
    let mut referenced = HashSet::new();
    for layer in &frame.layers {
        if !layers.insert(layer.id) {
            return Err(Error::invalid("prepared frame repeats a layer id"));
        }
        layer.validate(&resources, &mut referenced)?;
    }
    if referenced.len() != resources.len() {
        return Err(Error::invalid(
            "prepared frame manifest contains unreferenced resources",
        ));
    }
    Ok(())
}

fn hex_nibble(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(Error::invalid(
            "resource id contains a non-hexadecimal character",
        )),
    }
}

fn validate_surface(width: u32, height: u32) -> Result<()> {
    if width == 0
        || height == 0
        || width > MAX_SURFACE_DIMENSION
        || height > MAX_SURFACE_DIMENSION
        || u64::from(width) * u64::from(height) > MAX_SURFACE_PIXELS
    {
        return Err(Error::invalid(format!(
            "invalid prepared surface {width}x{height}"
        )));
    }
    Ok(())
}

fn finite_bounds(bounds: Bounds, label: &str) -> Result<()> {
    if [bounds.x, bounds.y, bounds.width, bounds.height]
        .into_iter()
        .all(f32::is_finite)
        && bounds.width >= 0.0
        && bounds.height >= 0.0
    {
        Ok(())
    } else {
        Err(Error::invalid(format!("{label} is invalid")))
    }
}

fn finite_affine(affine: [f64; 6], label: &str) -> Result<()> {
    if affine.into_iter().all(f64::is_finite) {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "{label} contains a non-finite value"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle() -> PreparedFrameBundle {
        let resource =
            PreparedResource::encoded_raster(1, 1, "image/png", Arc::from(&b"png"[..])).unwrap();
        let source = resource.id();
        PreparedFrameBundle {
            frame: PreparedFrame {
                revision: Revision::new(9),
                page: LayerId::from_bytes([1; 16]),
                width: 100,
                height: 80,
                origin: (0, 0),
                normalization: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                layers: vec![PreparedLayer {
                    id: LayerId::from_bytes([2; 16]),
                    geometry: vec![
                        Point { x: 0.0, y: 0.0 },
                        Point { x: 1.0, y: 0.0 },
                        Point { x: 1.0, y: 1.0 },
                        Point { x: 0.0, y: 1.0 },
                    ],
                    bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                    },
                    local_bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                    },
                    presentation: Presentation {
                        visible: true,
                        opacity: 1.0,
                    },
                    kind: LayerKind::Raster,
                    placement: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    content: PreparedContent::Raster(PreparedRaster {
                        source,
                        width: 1,
                        height: 1,
                        tiles: vec![PreparedRasterTile {
                            x: 0,
                            y: 0,
                            width: 1,
                            height: 1,
                            gutter: [0; 4],
                        }],
                    }),
                    element_frame: None,
                }],
            },
            resources: vec![resource],
        }
    }

    #[test]
    fn manifest_and_resource_packets_round_trip_independently() {
        let bundle = bundle();
        let expected_manifest = bundle.manifest().unwrap();
        let manifest = PreparedFrameManifest::decode(&expected_manifest.encode().unwrap()).unwrap();
        assert_eq!(manifest, expected_manifest);

        let resource_id = bundle.resources[0].id();
        let packet = bundle.resource_packet(resource_id).unwrap();
        let packet = PreparedResourcePacket::decode(&packet.encode().unwrap()).unwrap();
        assert_eq!(packet.id(), resource_id);
        assert_eq!(
            manifest.missing_resources(&PreparedResourceStore::default()),
            vec![resource_id]
        );

        let mut store = PreparedResourceStore::default();
        store.insert(packet);
        assert!(manifest.missing_resources(&store).is_empty());
        assert_eq!(store.total_bytes(), 3);
        let frame = manifest.compile(&store).unwrap();
        assert_eq!(
            frame.layers()[0].raster_image().unwrap().source(),
            resource_id
        );
        assert_eq!(
            frame.layers()[0].raster_image().unwrap().tiles()[0].id(),
            PreparedRasterTile {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                gutter: [0; 4],
            }
            .id(resource_id)
        );
    }

    #[test]
    fn resource_packet_checks_content_id_once_at_decode_boundary() {
        let mut resource = bundle().resources.remove(0);
        let PreparedResource::EncodedRaster { bytes, .. } = &mut resource else {
            unreachable!()
        };
        Arc::make_mut(bytes)[0] = 0;
        let bytes = postcard::to_stdvec(&ResourcePacket {
            magic: RESOURCE_MAGIC,
            version: PREPARED_RESOURCE_FORMAT_VERSION,
            resource,
        })
        .unwrap();
        assert!(matches!(
            PreparedResourcePacket::decode(&bytes),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn manifest_tracks_one_encoded_source_for_all_raster_tiles() {
        let source = PreparedResource::encoded_raster(
            PREPARED_RASTER_TILE_DIMENSION + 1,
            1,
            "image/png",
            Arc::from(&b"wide-png"[..]),
        )
        .unwrap();
        let mut bundle = bundle();
        bundle.resources = vec![source.clone()];
        bundle.frame.width = PREPARED_RASTER_TILE_DIMENSION + 1;
        bundle.frame.layers[0].content = PreparedContent::Raster(PreparedRaster {
            source: source.id(),
            width: PREPARED_RASTER_TILE_DIMENSION + 1,
            height: 1,
            tiles: vec![
                PreparedRasterTile {
                    x: 0,
                    y: 0,
                    width: PREPARED_RASTER_TILE_DIMENSION,
                    height: 1,
                    gutter: [0, 0, 1, 0],
                },
                PreparedRasterTile {
                    x: PREPARED_RASTER_TILE_DIMENSION,
                    y: 0,
                    width: 1,
                    height: 1,
                    gutter: [1, 0, 0, 0],
                },
            ],
        });

        let manifest = bundle.manifest().unwrap();
        assert_eq!(manifest.required_resources().len(), 1);
        let mut store = PreparedResourceStore::default();
        assert_eq!(manifest.missing_resources(&store), vec![source.id()]);
        store.insert(bundle.resource_packet(source.id()).unwrap());
        let frame = manifest.compile(&store).unwrap();
        let tiles = frame.layers()[0].raster_image().unwrap().tiles();
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].resource_size(), (1_025, 1));
        assert_eq!(tiles[1].resource_size(), (2, 1));

        let PreparedContent::Raster(raster) = &mut bundle.frame.layers[0].content else {
            unreachable!();
        };
        raster.tiles[1].gutter = [0; 4];
        assert!(matches!(bundle.manifest(), Err(Error::Invalid(_))));
    }

    #[test]
    fn a_shape_layer_accepts_a_vector_scene() {
        // A shape draws through the same vector content path as text: the
        // renderer compiles its parameters into one prepared fill path.
        let mut bundle = bundle();
        bundle.resources.clear();
        let layer = &mut bundle.frame.layers[0];
        layer.kind = LayerKind::Shape;
        layer.content = PreparedContent::Vector(PreparedScene {
            commands: vec![PreparedSceneCommand::FillPath(PreparedPath {
                transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                elements: vec![
                    PathElement::MoveTo([0.0, 0.0]),
                    PathElement::LineTo([10.0, 0.0]),
                    PathElement::LineTo([10.0, 10.0]),
                    PathElement::Close,
                ],
                fill: Some(PreparedFill::Solid {
                    color: [255, 0, 0, 255],
                    rule: FillRule::NonZero,
                }),
                stroke: None,
            })],
        });
        assert!(bundle.manifest().is_ok());
    }

    #[test]
    fn a_shape_layers_vector_content_is_validated() {
        // Accepting the pairing is not enough: the scene must actually go
        // through validation. A non-finite path transform is invalid for any
        // vector kind, so if the shape arm skipped `scene.validate` this would
        // wrongly pass.
        let mut bundle = bundle();
        bundle.resources.clear();
        let layer = &mut bundle.frame.layers[0];
        layer.kind = LayerKind::Shape;
        layer.content = PreparedContent::Vector(PreparedScene {
            commands: vec![PreparedSceneCommand::FillPath(PreparedPath {
                transform: [f64::NAN, 0.0, 0.0, 1.0, 0.0, 0.0],
                elements: vec![PathElement::MoveTo([0.0, 0.0]), PathElement::Close],
                fill: None,
                stroke: None,
            })],
        });
        assert!(matches!(bundle.manifest(), Err(Error::Invalid(_))));
    }

    #[test]
    fn a_shape_layer_rejects_raster_content() {
        // Only the `(Shape, Vector)` pairing is legal; a shape is never
        // raster-backed through this kind.
        let mut bundle = bundle();
        bundle.frame.layers[0].kind = LayerKind::Shape;
        assert!(matches!(bundle.manifest(), Err(Error::Invalid(_))));
    }

    #[test]
    fn packets_reject_unknown_versions() {
        let manifest = bundle().manifest().unwrap();
        let bytes = postcard::to_stdvec(&ManifestPacket {
            magic: MANIFEST_MAGIC,
            version: PREPARED_FRAME_MANIFEST_VERSION + 1,
            manifest,
        })
        .unwrap();
        assert!(matches!(
            PreparedFrameManifest::decode(&bytes),
            Err(Error::UnsupportedManifestVersion(version))
                if version == PREPARED_FRAME_MANIFEST_VERSION + 1
        ));

        let resource = bundle().resources.remove(0);
        let bytes = postcard::to_stdvec(&ResourcePacket {
            magic: RESOURCE_MAGIC,
            version: PREPARED_RESOURCE_FORMAT_VERSION + 1,
            resource,
        })
        .unwrap();
        assert!(matches!(
            PreparedResourcePacket::decode(&bytes),
            Err(Error::UnsupportedResourceVersion(version))
                if version == PREPARED_RESOURCE_FORMAT_VERSION + 1
        ));
    }

    #[test]
    fn resource_id_hex_is_canonical_and_strict() {
        let id = ResourceId::from_bytes([0xab; 32]);
        let text = id.to_string();
        assert_eq!(text, "ab".repeat(32));
        assert_eq!(text.parse::<ResourceId>().unwrap(), id);
        assert_eq!(text.to_ascii_uppercase().parse::<ResourceId>().unwrap(), id);
        assert!("ab".parse::<ResourceId>().is_err());
    }

    #[test]
    fn a_path_may_have_a_fill_and_a_stroke() {
        let path = PreparedPath {
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            elements: vec![
                PathElement::MoveTo([0.0, 0.0]),
                PathElement::LineTo([10.0, 0.0]),
                PathElement::LineTo([10.0, 10.0]),
                PathElement::Close,
            ],
            fill: Some(PreparedFill::Solid {
                color: [255, 0, 0, 255],
                rule: FillRule::NonZero,
            }),
            stroke: Some(PreparedStroke {
                color: [0, 0, 0, 255],
                width: 2.0,
                dash: None,
                cap: PreparedLineCap::Round,
                join: PreparedLineJoin::Round,
            }),
        };
        assert!(path.validate().is_ok());
    }

    #[test]
    fn a_path_without_fill_or_stroke_validates() {
        let path = PreparedPath {
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            elements: vec![PathElement::MoveTo([0.0, 0.0]), PathElement::Close],
            fill: None,
            stroke: None,
        };
        assert!(path.validate().is_ok());
    }

    #[test]
    fn a_negative_stroke_width_is_rejected() {
        let path = PreparedPath {
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            elements: vec![PathElement::MoveTo([0.0, 0.0]), PathElement::Close],
            fill: None,
            stroke: Some(PreparedStroke {
                color: [0, 0, 0, 255],
                width: -1.0,
                dash: None,
                cap: PreparedLineCap::Butt,
                join: PreparedLineJoin::Miter,
            }),
        };
        assert!(path.validate().is_err());
    }

    fn stroked(width: f32, dash: Option<Vec<f32>>) -> PreparedPath {
        PreparedPath {
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            elements: vec![PathElement::MoveTo([0.0, 0.0]), PathElement::Close],
            fill: None,
            stroke: Some(PreparedStroke {
                color: [0, 0, 0, 255],
                width,
                dash,
                cap: PreparedLineCap::Butt,
                join: PreparedLineJoin::Miter,
            }),
        }
    }

    #[test]
    fn a_non_finite_stroke_width_is_rejected() {
        assert!(stroked(f32::NAN, None).validate().is_err());
        assert!(stroked(f32::INFINITY, None).validate().is_err());
    }

    #[test]
    fn a_zero_stroke_width_is_allowed() {
        assert!(stroked(0.0, None).validate().is_ok());
    }

    #[test]
    fn a_dash_list_must_be_non_empty_and_positive() {
        assert!(stroked(1.0, Some(vec![8.0, 0.0])).validate().is_err());
        assert!(stroked(1.0, Some(vec![8.0, -1.0])).validate().is_err());
        assert!(stroked(1.0, Some(vec![f32::NAN])).validate().is_err());
        assert!(stroked(1.0, Some(Vec::new())).validate().is_err());
        assert!(stroked(1.0, Some(vec![4.0, 2.0])).validate().is_ok());
    }

    fn gradient_fill() -> PreparedFill {
        PreparedFill::Gradient {
            gradient: PreparedGradient {
                kind: PreparedGradientKind::Linear {
                    start: [0.0, 0.0],
                    end: [100.0, 0.0],
                },
                stops: vec![
                    PreparedGradientStop {
                        offset: 0.0,
                        color: [0, 0, 0, 255],
                    },
                    PreparedGradientStop {
                        offset: 1.0,
                        color: [255, 255, 255, 255],
                    },
                ],
                extend: PreparedGradientExtend::Pad,
            },
            rule: FillRule::NonZero,
        }
    }

    fn filled(fill: PreparedFill) -> PreparedPath {
        PreparedPath {
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            elements: vec![PathElement::MoveTo([0.0, 0.0]), PathElement::Close],
            fill: Some(fill),
            stroke: None,
        }
    }

    fn with_stops(stops: Vec<PreparedGradientStop>) -> PreparedGradient {
        let PreparedFill::Gradient { gradient, .. } = gradient_fill() else {
            unreachable!()
        };
        PreparedGradient { stops, ..gradient }
    }

    #[test]
    fn a_gradient_fill_validates() {
        assert!(filled(gradient_fill()).validate().is_ok());
    }

    /// The stop-count band is inclusive on both ends: an implementation that
    /// checked `< 2` or `<= 32` instead of `!(2..=32).contains(...)` fails here.
    #[test]
    fn the_stop_count_band_is_inclusive() {
        let stop = |offset| PreparedGradientStop {
            offset,
            color: [1, 2, 3, 255],
        };
        let stops_of = |count: u32| {
            (0..count)
                .map(|index| stop(index as f32 / (count - 1) as f32))
                .collect::<Vec<_>>()
        };
        assert!(
            filled(PreparedFill::Gradient {
                gradient: with_stops(stops_of(2)),
                rule: FillRule::NonZero,
            })
            .validate()
            .is_ok()
        );
        assert!(
            filled(PreparedFill::Gradient {
                gradient: with_stops(stops_of(32)),
                rule: FillRule::NonZero,
            })
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn a_gradient_with_one_stop_is_rejected() {
        let mut gradient = with_stops(vec![PreparedGradientStop {
            offset: 0.0,
            color: [0, 0, 0, 255],
        }]);
        gradient.stops.truncate(1);
        let path = filled(PreparedFill::Gradient {
            gradient,
            rule: FillRule::NonZero,
        });
        assert!(path.validate().is_err());
    }

    #[test]
    fn a_gradient_with_thirty_three_stops_is_rejected() {
        let stops = (0..33)
            .map(|index| PreparedGradientStop {
                offset: index as f32 / 32.0,
                color: [1, 2, 3, 255],
            })
            .collect();
        let path = filled(PreparedFill::Gradient {
            gradient: with_stops(stops),
            rule: FillRule::NonZero,
        });
        assert!(path.validate().is_err());
    }

    #[test]
    fn gradient_stop_offsets_must_be_finite_and_in_range() {
        for offset in [f32::NAN, f32::INFINITY, -0.5, 1.5] {
            let path = filled(PreparedFill::Gradient {
                gradient: with_stops(vec![
                    PreparedGradientStop {
                        offset: 0.0,
                        color: [0, 0, 0, 255],
                    },
                    PreparedGradientStop {
                        offset,
                        color: [255, 255, 255, 255],
                    },
                ]),
                rule: FillRule::NonZero,
            });
            assert!(path.validate().is_err(), "offset {offset} must be rejected");
        }
    }

    #[test]
    fn duplicate_gradient_stop_offsets_are_rejected() {
        let path = filled(PreparedFill::Gradient {
            gradient: with_stops(vec![
                PreparedGradientStop {
                    offset: 0.5,
                    color: [1, 1, 1, 255],
                },
                PreparedGradientStop {
                    offset: 0.5,
                    color: [2, 2, 2, 255],
                },
            ]),
            rule: FillRule::NonZero,
        });
        assert!(path.validate().is_err());
    }

    /// Unsorted stops are legal because the renderer sorts them; rejecting them
    /// here would contradict `stops_sorted` and the design spec's failure table.
    #[test]
    fn unsorted_gradient_stops_validate() {
        let path = filled(PreparedFill::Gradient {
            gradient: with_stops(vec![
                PreparedGradientStop {
                    offset: 0.9,
                    color: [1, 1, 1, 255],
                },
                PreparedGradientStop {
                    offset: 0.1,
                    color: [2, 2, 2, 255],
                },
            ]),
            rule: FillRule::NonZero,
        });
        assert!(path.validate().is_ok());
    }

    #[test]
    fn degenerate_gradient_geometry_is_rejected() {
        let cases = [
            PreparedGradientKind::Linear {
                start: [5.0, 5.0],
                end: [5.0, 5.0],
            },
            PreparedGradientKind::Linear {
                start: [0.0, f32::NAN],
                end: [1.0, 0.0],
            },
            PreparedGradientKind::Radial {
                center: [0.0, 0.0],
                radius: 0.0,
            },
            PreparedGradientKind::Radial {
                center: [0.0, 0.0],
                radius: -1.0,
            },
            PreparedGradientKind::Radial {
                center: [0.0, f32::INFINITY],
                radius: 1.0,
            },
            PreparedGradientKind::Sweep {
                center: [0.0, 0.0],
                start_degrees: f32::NAN,
                end_degrees: 360.0,
            },
            PreparedGradientKind::Sweep {
                center: [0.0, 0.0],
                start_degrees: 0.0,
                end_degrees: f32::INFINITY,
            },
        ];
        for kind in cases {
            let PreparedFill::Gradient { gradient, .. } = gradient_fill() else {
                unreachable!()
            };
            let path = filled(PreparedFill::Gradient {
                gradient: PreparedGradient {
                    kind: kind.clone(),
                    ..gradient
                },
                rule: FillRule::NonZero,
            });
            assert!(path.validate().is_err(), "{kind:?} must be rejected");
        }
    }

    #[test]
    fn non_degenerate_gradient_geometry_validates() {
        let cases = [
            PreparedGradientKind::Linear {
                start: [0.0, 0.0],
                end: [100.0, 0.0],
            },
            PreparedGradientKind::Radial {
                center: [50.0, 50.0],
                radius: 25.0,
            },
            PreparedGradientKind::Sweep {
                center: [50.0, 50.0],
                start_degrees: 0.0,
                end_degrees: 360.0,
            },
        ];
        for kind in cases {
            let PreparedFill::Gradient { gradient, .. } = gradient_fill() else {
                unreachable!()
            };
            let path = filled(PreparedFill::Gradient {
                gradient: PreparedGradient { kind, ..gradient },
                rule: FillRule::NonZero,
            });
            assert!(path.validate().is_ok());
        }
    }

    #[test]
    fn stops_are_sorted_by_offset() {
        let stops = vec![
            PreparedGradientStop {
                offset: 0.8,
                color: [255, 0, 0, 255],
            },
            PreparedGradientStop {
                offset: 0.2,
                color: [0, 0, 255, 255],
            },
        ];
        let sorted = stops_sorted(&stops);
        assert_eq!(sorted[0].offset, 0.2);
        assert_eq!(sorted[1].offset, 0.8);
        assert_eq!(sorted[0].color, [0, 0, 255, 255]);
    }

    #[test]
    fn stops_sorted_leaves_sorted_input_in_order() {
        let stops = vec![
            PreparedGradientStop {
                offset: 0.0,
                color: [0, 0, 0, 255],
            },
            PreparedGradientStop {
                offset: 0.5,
                color: [128, 128, 128, 255],
            },
            PreparedGradientStop {
                offset: 1.0,
                color: [255, 255, 255, 255],
            },
        ];
        assert_eq!(stops_sorted(&stops), stops);
    }

    /// Validation rejects duplicate offsets, but the sort itself must still be
    /// stable so an unsorted duplicate pair never swaps when it passes through
    /// the renderer's assembly path.
    #[test]
    fn stops_sorted_is_stable_for_equal_offsets() {
        let stops = vec![
            PreparedGradientStop {
                offset: 0.5,
                color: [1, 0, 0, 255],
            },
            PreparedGradientStop {
                offset: 0.5,
                color: [0, 1, 0, 255],
            },
            PreparedGradientStop {
                offset: 0.5,
                color: [0, 0, 1, 255],
            },
            PreparedGradientStop {
                offset: 0.1,
                color: [9, 9, 9, 255],
            },
        ];
        let sorted = stops_sorted(&stops);
        assert_eq!(
            sorted.iter().map(|stop| stop.color).collect::<Vec<_>>(),
            vec![
                [9, 9, 9, 255],
                [1, 0, 0, 255],
                [0, 1, 0, 255],
                [0, 0, 1, 255],
            ]
        );
    }

    #[test]
    fn a_ramp_interpolates_between_its_ends() {
        let ramp = BrushRamp::AlongStroke {
            from: [0, 0, 0, 255],
            to: [255, 255, 255, 255],
        };
        assert_eq!(ramp.color_at(0.0), [0, 0, 0, 255]);
        assert_eq!(ramp.color_at(1.0), [255, 255, 255, 255]);
        // Midpoint must be the rounded halfway value, not one end.
        assert_eq!(ramp.color_at(0.5), [128, 128, 128, 255]);
    }

    #[test]
    fn a_ramp_interpolates_alpha_too() {
        let ramp = BrushRamp::AlongStroke {
            from: [10, 20, 30, 0],
            to: [10, 20, 30, 255],
        };
        assert_eq!(ramp.color_at(0.0)[3], 0);
        assert_eq!(ramp.color_at(1.0)[3], 255);
        // Alpha must move; a ramp that only interpolates RGB would return 0 here.
        assert!(ramp.color_at(0.5)[3] > 0, "alpha must interpolate");
    }

    #[test]
    fn a_ramp_clamps_outside_the_unit_interval() {
        let ramp = BrushRamp::AlongStroke {
            from: [0, 0, 0, 255],
            to: [100, 100, 100, 255],
        };
        assert_eq!(ramp.color_at(-1.0), [0, 0, 0, 255]);
        assert_eq!(ramp.color_at(2.0), [100, 100, 100, 255]);
    }

    #[test]
    fn a_ramp_with_equal_ends_is_a_solid_colour() {
        let ramp = BrushRamp::FixedAxis {
            from: [7, 8, 9, 255],
            to: [7, 8, 9, 255],
            axis: RampAxis::Horizontal,
        };
        for position in [0.0, 0.25, 0.5, 1.0] {
            assert_eq!(ramp.color_at(position), [7, 8, 9, 255]);
        }
    }

    #[test]
    fn a_non_finite_position_does_not_panic() {
        let ramp = BrushRamp::AlongStroke {
            from: [0, 0, 0, 255],
            to: [255, 255, 255, 255],
        };
        // NaN must not produce a panic or an out-of-range channel; clamping has
        // to happen before the cast.
        let color = ramp.color_at(f32::NAN);
        assert_eq!(color.len(), 4);
    }

    #[test]
    fn the_ramp_types_are_reachable_from_the_crate_root() {
        // The 1b review found gradient types that could not be named outside the
        // crate. Pin the re-export so the same gap cannot reappear.
        assert_eq!(
            koharu_rasterizer::BrushRamp::AlongStroke {
                from: [0, 0, 0, 0],
                to: [0, 0, 0, 0],
            }
            .color_at(0.0),
            [0, 0, 0, 0]
        );
    }
}
