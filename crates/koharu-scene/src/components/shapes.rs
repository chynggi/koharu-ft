use revision::revisioned;
use serde::{Deserialize, Serialize};
use specta::Type;

use crate::{
    Error, Result,
    component::{Component, ValidationContext},
};

use super::{Origin, Point};

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

impl ShapeGeometry {
    /// Realigns a `Line`'s endpoints with the polygon, which is authoritative.
    ///
    /// `Line` is the only variant whose drawing parameters are stored here as
    /// well, so a polygon update must carry them along: `line_path` strokes
    /// `start`/`end`, while `element_frame` reads the polygon, so a stale pair
    /// makes the drawn line and its selection box separate. The first polygon
    /// point is `start` and the last is `end`, because a line's polygon is the
    /// segment sampled into three or more points. Every other variant derives
    /// nothing from the polygon and is left alone.
    pub fn resync_with_polygon(&mut self, points: &[Point]) {
        if let Self::Line { start, end, .. } = self
            && let (Some(first), Some(last)) = (points.first(), points.last())
        {
            *start = [first.x, first.y];
            *end = [last.x, last.y];
        }
    }
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct ArrowHead {
    pub length: f32,
    pub width: f32,
}

/// A persisted gradient fill.
///
/// Coordinates are frame-local normalized values: the frame's top-left is
/// `(0,0)` and its bottom-right is `(1,1)`. They are never page-absolute, so
/// moving or resizing the shape's polygon does not invalidate them.
///
/// Stops may be stored in any order. The renderer sorts them, and validation
/// deliberately does not require sorted offsets.
#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct GradientSpec {
    pub kind: GradientKind,
    pub stops: Vec<GradientStop>,
    pub extend: GradientExtend,
}

impl GradientSpec {
    /// Rejects only shapes the renderer cannot resolve.
    ///
    /// Unsorted stops are *allowed*: the renderer sorts them, and forcing order
    /// here would make the editor re-sort the whole list on every mid-list
    /// insert. Only duplicate offsets are rejected, because which color wins
    /// would otherwise be undefined.
    ///
    /// `-0.0` passes the `0.0..=1.0` range check and compares equal to `0.0`,
    /// so it counts as a duplicate of an existing `0.0` stop. That is correct
    /// IEEE-754 behavior, not a gap.
    pub fn validate(&self, _context: &ValidationContext<'_>) -> Result<()> {
        if !(2..=32).contains(&self.stops.len()) {
            return Err(Error::invalid("gradient must have between 2 and 32 stops"));
        }
        for stop in &self.stops {
            if !stop.offset.is_finite() || !(0.0..=1.0).contains(&stop.offset) {
                return Err(Error::invalid(
                    "gradient stop offset must be finite and within 0.0..=1.0",
                ));
            }
        }
        for (index, stop) in self.stops.iter().enumerate() {
            if self.stops[index + 1..]
                .iter()
                .any(|other| other.offset == stop.offset)
            {
                return Err(Error::invalid("gradient stop offsets must be unique"));
            }
        }
        match &self.kind {
            GradientKind::Linear { start, end } => {
                if start.iter().chain(end).any(|value| !value.is_finite()) {
                    return Err(Error::invalid("gradient endpoints must be finite"));
                }
                if start == end {
                    return Err(Error::invalid("gradient linear endpoints must differ"));
                }
            }
            GradientKind::Radial { radius, .. } => {
                if !radius.is_finite() || *radius <= 0.0 {
                    return Err(Error::invalid(
                        "gradient radial radius must be finite and positive",
                    ));
                }
            }
            GradientKind::Sweep {
                start_degrees,
                end_degrees,
                ..
            } => {
                if !start_degrees.is_finite() || !end_degrees.is_finite() {
                    return Err(Error::invalid("gradient sweep angles must be finite"));
                }
            }
        }
        Ok(())
    }
}

#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum GradientKind {
    /// Interpolates along the straight line from `start` to `end`; both points
    /// are normalized frame-local coordinates.
    Linear { start: [f32; 2], end: [f32; 2] },
    /// Radiates from `center` by `radius`, expressed as a fraction of the
    /// frame's width.
    Radial { center: [f32; 2], radius: f32 },
    /// Sweeps from `start_degrees` to `end_degrees` around `center`.
    Sweep {
        center: [f32; 2],
        start_degrees: f32,
        end_degrees: f32,
    },
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct GradientStop {
    /// `0.0..=1.0`. Stops need not be sorted; the renderer sorts them.
    pub offset: f32,
    pub color: [u8; 4],
}

#[revisioned(revision = 1)]
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum GradientExtend {
    /// Fills beyond the stops with the last stop's color. The editor picks this
    /// when it creates a gradient; there is deliberately no `Default` impl.
    Pad,
    Repeat,
    Reflect,
}

#[revisioned(revision = 1)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ShapeFill {
    Solid([u8; 4]),
    /// 1b. `Gradient` stays last: the revision enum tag carries the variant id
    /// in bits 0..=4, so appending keeps the existing `Solid` at id 0 and old
    /// data decodes unchanged.
    Gradient(GradientSpec),
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
    fn validate(&self, context: &ValidationContext<'_>) -> Result<()> {
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
        if let Some(fill) = &self.style.fill
            && let ShapeFill::Gradient(gradient) = fill
        {
            gradient.validate(context)?;
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

    fn spec(kind: GradientKind) -> GradientSpec {
        GradientSpec {
            kind,
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
        }
    }

    fn linear() -> GradientKind {
        GradientKind::Linear {
            start: [0.0, 0.0],
            end: [1.0, 1.0],
        }
    }

    fn validate_spec(spec: &GradientSpec) -> Result<()> {
        let record_exists = |_id| true;
        let blob_exists = |_id| true;
        let context = ValidationContext::new(&record_exists, &blob_exists);
        spec.validate(&context)
    }

    /// Compile-checked proof that all four types are re-exported at the crate
    /// root, which is the path an external crate such as `koharu-renderer`
    /// must use. Every annotation here fails to compile if its re-export is
    /// dropped from `components.rs` or `lib.rs`.
    #[test]
    fn the_gradient_types_are_reachable_from_the_crate_root() {
        let kind: crate::GradientKind = linear();
        let gradient: crate::GradientSpec = GradientSpec {
            kind,
            stops: vec![
                crate::GradientStop {
                    offset: 0.0,
                    color: [0, 0, 0, 255],
                },
                crate::GradientStop {
                    offset: 1.0,
                    color: [255, 255, 255, 255],
                },
            ],
            extend: crate::GradientExtend::Pad,
        };
        assert!(validate_spec(&gradient).is_ok());
    }

    #[test]
    fn a_two_stop_gradient_validates() {
        assert!(validate_spec(&spec(linear())).is_ok());
    }

    #[test]
    fn a_thirty_two_stop_gradient_validates() {
        let mut gradient = spec(linear());
        gradient.stops = (0..32)
            .map(|index| GradientStop {
                offset: index as f32 / 31.0,
                color: [index as u8, 0, 0, 255],
            })
            .collect();
        assert!(validate_spec(&gradient).is_ok());
    }

    #[test]
    fn a_single_stop_is_rejected() {
        let mut gradient = spec(linear());
        gradient.stops.truncate(1);
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn thirty_three_stops_are_rejected() {
        let mut gradient = spec(linear());
        gradient.stops = (0..33)
            .map(|_| GradientStop {
                offset: 0.5,
                color: [0, 0, 0, 255],
            })
            .collect();
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn an_out_of_range_stop_offset_is_rejected() {
        let mut gradient = spec(linear());
        gradient.stops[0].offset = 1.5;
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn a_non_finite_stop_offset_is_rejected() {
        let mut gradient = spec(linear());
        gradient.stops[0].offset = f32::NAN;
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn duplicate_stop_offsets_are_rejected() {
        let mut gradient = spec(linear());
        gradient.stops[1].offset = 0.0;
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn unsorted_stop_offsets_are_allowed() {
        let mut gradient = spec(linear());
        gradient.stops = vec![
            GradientStop {
                offset: 0.8,
                color: [255, 0, 0, 255],
            },
            GradientStop {
                offset: 0.2,
                color: [0, 0, 255, 255],
            },
        ];
        assert!(validate_spec(&gradient).is_ok());
    }

    #[test]
    fn a_zero_length_linear_gradient_is_rejected() {
        let gradient = spec(GradientKind::Linear {
            start: [0.5, 0.5],
            end: [0.5, 0.5],
        });
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn a_positive_radius_radial_gradient_validates() {
        let gradient = spec(GradientKind::Radial {
            center: [0.5, 0.5],
            radius: 0.5,
        });
        assert!(validate_spec(&gradient).is_ok());
    }

    #[test]
    fn a_finite_sweep_gradient_validates() {
        let gradient = spec(GradientKind::Sweep {
            center: [0.5, 0.5],
            start_degrees: 0.0,
            end_degrees: 360.0,
        });
        assert!(validate_spec(&gradient).is_ok());
    }

    #[test]
    fn a_zero_radius_radial_gradient_is_rejected() {
        let gradient = spec(GradientKind::Radial {
            center: [0.5, 0.5],
            radius: 0.0,
        });
        assert!(validate_spec(&gradient).is_err());
    }

    #[test]
    fn a_non_finite_sweep_angle_is_rejected() {
        let gradient = spec(GradientKind::Sweep {
            center: [0.5, 0.5],
            start_degrees: f32::INFINITY,
            end_degrees: 360.0,
        });
        assert!(validate_spec(&gradient).is_err());
    }

    /// `ShapeFill` gains a variant; a `Solid` written before it existed must
    /// still decode as `Solid`. This is the direct evidence the spec requires
    /// instead of trusting that appending a variant preserves ids.
    #[test]
    fn a_solid_fill_round_trips_with_the_gradient_variant_present() {
        let solid = ShapeFill::Solid([1, 2, 3, 255]);
        let bytes = revision::to_vec(&solid).unwrap();
        // `[revision varint = 1][enum tag = 0][r, g, b, a]`. Pinning the whole
        // vector fails if `Solid` stops being variant id 0 (tag becomes 1) or if
        // `ShapeFill`'s revision is bumped (the varint changes), so this is real
        // evidence rather than a round-trip that any ordering would satisfy.
        assert_eq!(bytes, [1, 0, 1, 2, 3, 255]);
        let decoded: ShapeFill = revision::from_slice(&bytes).unwrap();
        assert_eq!(decoded, solid);
    }
}
