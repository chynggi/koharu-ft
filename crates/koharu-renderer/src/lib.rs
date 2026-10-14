//! Retained rendering of one semantic Koharu page.

mod bubble;
mod config;
mod error;
mod fonts;
mod frame;
mod images;
mod layout;
mod renderer;
mod script;
mod segment;
mod shape;
// Public because Task 7 rasterizes shapes from `koharu-app` through
// `koharu_renderer::shape_renderer::{shape_frame, shape_layer_path}`.
pub mod shape_renderer;
mod text_renderer;
mod types;

pub use config::TypesettingConfig;
pub use error::{Error, Result};
pub use frame::{
    Frame, ImageKind, ImageMetadata, Layer, LayerKind, Presentation, RasterImage, RenderBounds,
    RenderDependency, RenderDiagnostic, RetentionStats, ShapeMetadata, TextMetadata,
};
pub use layout::WritingMode;
pub use renderer::Renderer;
pub use types::{FontFace, FontFamily, FontMetadata, FontRange, FontSource, FontStyle, TextAlign};

pub(crate) use layout::{HyphenationPolicy, LayoutRun, TextLayout};
