use revision::revisioned;
use serde::{Deserialize, Serialize};
use specta::Type;

use crate::{
    Error, Result,
    component::{Component, ValidationContext},
};

use super::Origin;

/// How a stroked path terminates an open subpath.
#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LineCap {
    Butt,
    Round,
    Square,
}

/// How a stroked path joins two segments.
#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum LineJoin {
    Miter,
    Round,
    Bevel,
}

/// Parameters that regenerate a shape's outline.
///
/// The outline itself is the entity's [`crate::Geometry`] polygon; these are
/// the editable parameters the renderer rebuilds a path from.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ShapeGeometry {
    /// `corner_radius` is in page units and must be finite and non-negative.
    Rectangle {
        corner_radius: f32,
    },
    Ellipse,
    /// A directed segment; `head` draws an arrow head at `end`.
    Line {
        start: [f64; 2],
        end: [f64; 2],
        head: Option<ArrowHead>,
    },
    /// Points live in the entity's `Geometry` component.
    Polygon,
    /// Points live in the entity's `Geometry` component.
    Freeform,
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct ArrowHead {
    pub length: f32,
    pub width: f32,
}

#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ShapeFill {
    Solid([u8; 4]),
}

#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct StrokeStyle {
    pub color: [u8; 4],
    pub width: f32,
    /// On/off lengths in page units; every element must be finite and positive.
    pub dash: Option<Vec<f32>>,
    pub cap: LineCap,
    pub join: LineJoin,
}

#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct ShapeStyle {
    pub fill: Option<ShapeFill>,
    pub stroke: Option<StrokeStyle>,
}

/// A vector shape layer described by regenerable parameters plus local style.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct ShapeLayer {
    pub origin: Origin,
    pub name: String,
    pub geometry: ShapeGeometry,
    pub style: ShapeStyle,
    pub angle_degrees: Option<f32>,
    pub rasterized: bool,
}

impl Component for ShapeLayer {
    const KIND: &'static str = "dev.koharu.layer.shape";

    // A polygon/parameter mismatch is deliberately not checked: `ShapeLayer`
    // carries no polygon, so there is nothing here to disagree with
    // `ShapeGeometry`. The polygon is a separate `Geometry` component on the
    // same entity, and only the renderer pairs the two.
    fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        self.origin.validate()?;
        if self.name.is_empty() || self.name.len() > 4096 || self.name.contains('\0') {
            return Err(Error::invalid("shape layer name is invalid"));
        }
        if let ShapeGeometry::Rectangle { corner_radius } = self.geometry
            && (!corner_radius.is_finite() || corner_radius < 0.0)
        {
            return Err(Error::invalid(
                "shape corner radius must be finite and non-negative",
            ));
        }
        if let ShapeGeometry::Line {
            head: Some(head), ..
        } = self.geometry
            && (!head.length.is_finite()
                || head.length < 0.0
                || !head.width.is_finite()
                || head.width < 0.0)
        {
            return Err(Error::invalid(
                "shape arrow head must be finite and non-negative",
            ));
        }
        if let Some(stroke) = &self.style.stroke {
            if !stroke.width.is_finite() || stroke.width < 0.0 {
                return Err(Error::invalid(
                    "shape stroke width must be finite and non-negative",
                ));
            }
            if stroke.dash.as_ref().is_some_and(|dash| {
                dash.iter()
                    .any(|length| !length.is_finite() || *length <= 0.0)
            }) {
                return Err(Error::invalid(
                    "shape dash lengths must be finite and positive",
                ));
            }
        }
        if self.angle_degrees.is_some_and(|angle| !angle.is_finite()) {
            return Err(Error::invalid("shape angle must be finite"));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Origin, Result, component::ValidationContext};

    fn validate(layer: &ShapeLayer) -> Result<()> {
        let record_exists = |_id| true;
        let blob_exists = |_id| true;
        let context = ValidationContext::new(&record_exists, &blob_exists);
        layer.validate(&context)
    }

    fn style() -> ShapeStyle {
        ShapeStyle {
            fill: Some(ShapeFill::Solid([0, 0, 0, 255])),
            stroke: None,
        }
    }

    fn shape(geometry: ShapeGeometry) -> ShapeLayer {
        ShapeLayer {
            origin: Origin::User,
            name: "Rectangle 1".to_owned(),
            geometry,
            style: style(),
            angle_degrees: None,
            rasterized: false,
        }
    }

    #[test]
    fn a_named_shape_validates() {
        let layer = shape(ShapeGeometry::Rectangle { corner_radius: 4.0 });
        assert!(validate(&layer).is_ok());
    }

    #[test]
    fn a_negative_corner_radius_is_rejected() {
        let layer = shape(ShapeGeometry::Rectangle {
            corner_radius: -1.0,
        });
        assert!(validate(&layer).is_err());
    }

    #[test]
    fn a_non_finite_angle_is_rejected() {
        let mut layer = shape(ShapeGeometry::Ellipse);
        layer.angle_degrees = Some(f32::NAN);
        assert!(validate(&layer).is_err());
    }

    #[test]
    fn a_zero_dash_element_is_rejected() {
        let mut layer = shape(ShapeGeometry::Polygon);
        layer.style.stroke = Some(StrokeStyle {
            color: [0, 0, 0, 255],
            width: 2.0,
            dash: Some(vec![8.0, 0.0]),
            cap: LineCap::Butt,
            join: LineJoin::Miter,
        });
        assert!(validate(&layer).is_err());
    }

    #[test]
    fn an_empty_name_is_rejected() {
        let mut layer = shape(ShapeGeometry::Freeform);
        layer.name = String::new();
        assert!(validate(&layer).is_err());
    }

    #[test]
    fn a_negative_stroke_width_is_rejected() {
        let mut layer = shape(ShapeGeometry::Polygon);
        layer.style.stroke = Some(StrokeStyle {
            color: [0, 0, 0, 255],
            width: -1.0,
            dash: None,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
        });
        assert!(validate(&layer).is_err());
    }

    #[test]
    fn a_negative_arrow_head_is_rejected() {
        let layer = shape(ShapeGeometry::Line {
            start: [0.0, 0.0],
            end: [10.0, 0.0],
            head: Some(ArrowHead {
                length: -1.0,
                width: 4.0,
            }),
        });
        assert!(validate(&layer).is_err());
    }
}
