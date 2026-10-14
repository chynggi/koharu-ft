//! Portable prepared frames and shared Vello/WGPU composition.

// Lets this crate's own tests name items through the public crate path rather
// than a private `crate::` path, so a missing re-export fails a test instead of
// passing under a path no downstream crate can use.
extern crate self as koharu_rasterizer;

mod compositor;
mod error;
mod frame;
#[cfg(feature = "native")]
mod native;
mod prepared;

pub use compositor::{
    CompositionCommand, DEFAULT_RASTER_CACHE_BUDGET_BYTES, GpuCompositor, RasterDraw,
};
pub use error::{Error, Result};
pub use frame::{Frame, Layer, RasterImage, RasterTile};
#[cfg(feature = "native")]
pub use native::{DownsampleFilter, Raster, RasterOptions, Rasterizer};
pub use prepared::{
    Bounds, BrushRamp, FillRule, LayerId, LayerKind, PREPARED_FRAME_MANIFEST_VERSION,
    PREPARED_RASTER_TILE_DIMENSION, PREPARED_RESOURCE_FORMAT_VERSION, PathElement, Point,
    PreparedContent, PreparedElementFrame, PreparedFill, PreparedFrame, PreparedFrameBundle,
    PreparedFrameManifest, PreparedGlyph, PreparedGlyphRun, PreparedGradient,
    PreparedGradientExtend, PreparedGradientKind, PreparedGradientStop, PreparedLayer,
    PreparedLineCap, PreparedLineJoin, PreparedPath, PreparedRaster, PreparedRasterTile,
    PreparedResource, PreparedResourceKind, PreparedResourcePacket, PreparedResourceRef,
    PreparedResourceStore, PreparedScene, PreparedSceneCommand, PreparedStroke, Presentation,
    RampAxis, ResourceId, Revision, stops_sorted,
};
