use std::collections::BTreeMap;

use revision::revisioned;
use serde::{Deserialize, Serialize};
use specta::Type;

use crate::{
    Error, Result,
    component::{Component, ValidationContext},
    id::validate_namespaced,
};

use super::Origin;

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum RasterLayerKind {
    Cleanup,
    Paint,
}

/// A full-page transparent pixel layer composited above the page source.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct RasterLayer {
    pub origin: Origin,
    pub name: String,
    pub kind: RasterLayerKind,
}

impl Component for RasterLayer {
    const KIND: &'static str = "dev.koharu.layer.raster";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        self.origin.validate()?;
        if self.name.is_empty() || self.name.len() > 4096 || self.name.contains('\0') {
            return Err(Error::invalid("raster layer name is invalid"));
        }
        Ok(())
    }

    fn origin(&self) -> Option<&Origin> {
        Some(&self.origin)
    }

    fn set_origin(&mut self, origin: Origin) -> bool {
        self.origin = origin;
        true
    }
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum TextLayoutKind {
    Point,
    Paragraph,
}

#[revisioned(revision = 2)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct TextLayout {
    pub origin: Origin,
    pub kind: TextLayoutKind,
    /// Overrides the axes inferred from geometry when a frame is transformed.
    #[revision(start = 2)]
    pub angle_degrees: Option<f32>,
}

impl Component for TextLayout {
    const KIND: &'static str = "dev.koharu.layer.text";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        if self.angle_degrees.is_some_and(|angle| !angle.is_finite()) {
            return Err(Error::invalid("text layout angle must be finite"));
        }
        self.origin.validate()
    }

    fn origin(&self) -> Option<&Origin> {
        Some(&self.origin)
    }

    fn set_origin(&mut self, origin: Origin) -> bool {
        self.origin = origin;
        true
    }
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
pub enum TextAlignment {
    Start,
    Center,
    End,
    Justify,
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
pub enum WritingMode {
    Horizontal,
    Vertical,
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
}

#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
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
}

impl Component for Typography {
    const KIND: &'static str = "dev.koharu.text.typography";

    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        // Each rejection names the intent it protects. The desktop surfaces this
        // string in the failure toast, and one shared "typography intent is
        // invalid" cannot tell the person who hit it which field a caller sent
        // wrong, or whether a font was involved at all.
        if self
            .preferred_font
            .as_ref()
            .is_some_and(|font| font.len() > 4096)
        {
            return Err(Error::invalid("preferred font name exceeds 4096 bytes"));
        }
        if self
            .font_weight
            .is_some_and(|weight| !(1..=1000).contains(&weight))
        {
            return Err(Error::invalid("font weight must be within 1..=1000"));
        }
        if self
            .size
            .is_some_and(|size| !size.is_finite() || size <= 0.0)
        {
            return Err(Error::invalid("font size must be finite and positive"));
        }
        if !self.auto_fit && self.size.is_none() {
            return Err(Error::invalid(
                "auto-fit off requires an explicit font size",
            ));
        }
        if self
            .stroke_width
            .is_some_and(|width| !width.is_finite() || width < 0.0)
        {
            return Err(Error::invalid(
                "stroke width must be finite and non-negative",
            ));
        }
        self.origin.validate()?;
        if self.extensions.len() > 1024
            || self.extensions.iter().any(|(key, value)| {
                validate_namespaced(key, "typography extension").is_err()
                    || value.len() > 64 * 1024
                    || value.contains('\0')
            })
        {
            return Err(Error::invalid("typography extensions are invalid"));
        }
        Ok(())
    }

    fn origin(&self) -> Option<&Origin> {
        Some(&self.origin)
    }

    fn set_origin(&mut self, origin: Origin) -> bool {
        self.origin = origin;
        true
    }
}
