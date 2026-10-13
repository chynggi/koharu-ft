//! Shape parameters plus a frame become one prepared drawable path.
//!
//! This module is the single implementation of per-shape-kind geometry. It maps
//! a [`ShapeGeometry`] and its frame onto [`PreparedPath`] elements in the
//! frame's local coordinates, where `(0, 0)` is the frame box's top-left; the
//! returned `transform` rotates and places that box on the page. Rectangle and
//! ellipse geometry comes from the frame, while polygon and freeform geometry
//! comes from the entity's polygon points, because the parameters are lossless
//! and the polygon is only the hit-testing approximation
//! (`specs/2026-10-13-shape-tools-design.md`, "무엇이 그리는가").

use koharu_rasterizer::{
    FillRule, PathElement, PreparedFill, PreparedLineCap, PreparedLineJoin, PreparedPath,
    PreparedStroke,
};
use koharu_scene::{
    ArrowHead, Geometry, LineCap, LineJoin, ShapeFill, ShapeGeometry, ShapeStyle, StrokeStyle,
};

/// Cubic approximation constant for a quarter circle.
const KAPPA: f64 = 0.552_284_749_830_793_6;

/// A shape's authored frame: an axis-aligned box in the shape's own axes plus
/// its rotation about that box's centre.
///
/// This mirrors `crate::bubble::GeometryFrame`, which is `pub(crate)`.
/// `koharu-app` rasterizes shapes and needs the frame, so this public type and
/// [`shape_frame`] are the conversion boundary into it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeFrame {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub angle_degrees: f32,
}

/// Measures `geometry` in its authored axes when an angle is known.
///
/// With `Some(angle)` the polygon is un-rotated before its bounds are taken, so
/// a rotated ellipse keeps its true diameter instead of growing to its
/// axis-aligned box. `None` falls back to axis-aligned bounds. `None` is
/// returned when `crate::bubble::geometry_frame` cannot measure the geometry,
/// which callers treat as "skip this shape".
#[must_use]
pub fn shape_frame(geometry: &Geometry, angle_degrees: Option<f32>) -> Option<ShapeFrame> {
    crate::bubble::geometry_frame(geometry, angle_degrees).map(|frame| ShapeFrame {
        x: frame.bounds.x,
        y: frame.bounds.y,
        width: frame.bounds.width,
        height: frame.bounds.height,
        angle_degrees: frame.angle_degrees,
    })
}

/// Builds the path for `Rectangle`, `Ellipse`, and `Line`.
///
/// `Polygon` and `Freeform` have no parameters to draw from, so this returns
/// the empty path for them; [`polygon_path`] builds theirs from the entity's
/// points and [`shape_layer_path`] chooses between the two.
#[must_use]
pub fn shape_path(geometry: &ShapeGeometry, style: &ShapeStyle, frame: ShapeFrame) -> PreparedPath {
    let width = f64::from(frame.width);
    let height = f64::from(frame.height);
    match geometry {
        ShapeGeometry::Rectangle { corner_radius } => prepared(
            rectangle_path(width, height, *corner_radius),
            style,
            frame,
            false,
        ),
        ShapeGeometry::Ellipse => prepared(ellipse_path(width, height), style, frame, false),
        ShapeGeometry::Line { start, end, head } => {
            let (elements, has_head) = line_path(frame, *start, *end, *head);
            prepared(elements, style, frame, has_head)
        }
        ShapeGeometry::Polygon | ShapeGeometry::Freeform => {
            prepared(empty_path(), style, frame, false)
        }
    }
}

/// Builds the path for `Polygon` and `Freeform` from the entity's points.
#[must_use]
pub fn polygon_path(
    points: &[koharu_scene::Point],
    style: &ShapeStyle,
    frame: ShapeFrame,
) -> PreparedPath {
    let mut elements = Vec::with_capacity(points.len() + 2);
    let mut points = points.iter().map(|point| [point.x, point.y]);
    if let Some(first) = points.next() {
        elements.push(PathElement::MoveTo(to_local(frame, first)));
        elements.extend(points.map(|point| PathElement::LineTo(to_local(frame, point))));
        elements.push(PathElement::Close);
    } else {
        elements.push(PathElement::MoveTo([0.0, 0.0]));
    }
    prepared(elements, style, frame, false)
}

/// Builds the one path a shape layer draws, delegating polygon geometry to the
/// entity's points and every other kind to its parameters.
#[must_use]
pub fn shape_layer_path(
    shape: &ShapeGeometry,
    points: &[koharu_scene::Point],
    style: &ShapeStyle,
    frame: ShapeFrame,
) -> PreparedPath {
    match shape {
        ShapeGeometry::Polygon | ShapeGeometry::Freeform => polygon_path(points, style, frame),
        _ => shape_path(shape, style, frame),
    }
}

/// The affine that rotates the local frame box about its centre and places it on
/// the page: Kurbo coefficients `[a, b, c, d, e, f]`.
fn transform(frame: ShapeFrame) -> [f64; 6] {
    let (sin, cos) = f64::from(frame.angle_degrees).to_radians().sin_cos();
    let center_x = f64::from(frame.x + frame.width * 0.5);
    let center_y = f64::from(frame.y + frame.height * 0.5);
    [
        cos,
        sin,
        -sin,
        cos,
        center_x - (cos * f64::from(frame.width) * 0.5 - sin * f64::from(frame.height) * 0.5),
        center_y - (sin * f64::from(frame.width) * 0.5 + cos * f64::from(frame.height) * 0.5),
    ]
}

/// Maps a page point into the frame's local coordinates: un-rotate about the
/// frame centre, then place relative to the box's top-left.
fn to_local(frame: ShapeFrame, point: [f64; 2]) -> [f64; 2] {
    let (sin, cos) = (-f64::from(frame.angle_degrees)).to_radians().sin_cos();
    let center_x = f64::from(frame.x + frame.width * 0.5);
    let center_y = f64::from(frame.y + frame.height * 0.5);
    let x = point[0] - center_x;
    let y = point[1] - center_y;
    [
        x * cos - y * sin + f64::from(frame.width) * 0.5,
        x * sin + y * cos + f64::from(frame.height) * 0.5,
    ]
}

fn prepared(
    elements: Vec<PathElement>,
    style: &ShapeStyle,
    frame: ShapeFrame,
    has_head: bool,
) -> PreparedPath {
    PreparedPath {
        transform: transform(frame),
        elements,
        fill: fill(style, has_head),
        stroke: style.stroke.as_ref().map(stroke),
    }
}

/// The path a shape draws with no geometry of its own.
fn empty_path() -> Vec<PathElement> {
    vec![PathElement::MoveTo([0.0, 0.0])]
}

fn rectangle_path(width: f64, height: f64, corner_radius: f32) -> Vec<PathElement> {
    let radius = f64::from(corner_radius).min(width * 0.5).min(height * 0.5);
    if radius <= 0.0 {
        return vec![
            PathElement::MoveTo([0.0, 0.0]),
            PathElement::LineTo([width, 0.0]),
            PathElement::LineTo([width, height]),
            PathElement::LineTo([0.0, height]),
            PathElement::Close,
        ];
    }
    vec![
        PathElement::MoveTo([radius, 0.0]),
        PathElement::LineTo([width - radius, 0.0]),
        PathElement::QuadTo([width, 0.0, width, radius]),
        PathElement::LineTo([width, height - radius]),
        PathElement::QuadTo([width, height, width - radius, height]),
        PathElement::LineTo([radius, height]),
        PathElement::QuadTo([0.0, height, 0.0, height - radius]),
        PathElement::LineTo([0.0, radius]),
        PathElement::QuadTo([0.0, 0.0, radius, 0.0]),
        PathElement::Close,
    ]
}

/// The ellipse inscribed in the frame, as four cubic quarter arcs.
fn ellipse_path(width: f64, height: f64) -> Vec<PathElement> {
    let (radius_x, radius_y) = (width * 0.5, height * 0.5);
    let (control_x, control_y) = (radius_x * KAPPA, radius_y * KAPPA);
    vec![
        PathElement::MoveTo([radius_x, 0.0]),
        PathElement::CurveTo([
            radius_x + control_x,
            0.0,
            width,
            radius_y - control_y,
            width,
            radius_y,
        ]),
        PathElement::CurveTo([
            width,
            radius_y + control_y,
            radius_x + control_x,
            height,
            radius_x,
            height,
        ]),
        PathElement::CurveTo([
            radius_x - control_x,
            height,
            0.0,
            radius_y + control_y,
            0.0,
            radius_y,
        ]),
        PathElement::CurveTo([
            0.0,
            radius_y - control_y,
            radius_x - control_x,
            0.0,
            radius_x,
            0.0,
        ]),
        PathElement::Close,
    ]
}

/// The line's own two points plus, when requested, the arrow head triangle.
fn line_path(
    frame: ShapeFrame,
    start: [f64; 2],
    end: [f64; 2],
    head: Option<ArrowHead>,
) -> (Vec<PathElement>, bool) {
    let start = to_local(frame, start);
    let end = to_local(frame, end);
    let mut elements = vec![
        PathElement::MoveTo(start),
        PathElement::LineTo(end),
        PathElement::Close,
    ];
    // A zero-length line has no direction, so there is no perpendicular along
    // which a head could sit. The degenerate MoveTo/LineTo stays, keeping the
    // path finite and valid, and the head is dropped instead of dividing by
    // zero.
    let delta = [end[0] - start[0], end[1] - start[1]];
    let length = delta[0].hypot(delta[1]);
    let has_head = head.is_some() && length > 0.0 && length.is_finite();
    if let Some(head) = head
        && has_head
    {
        let direction = [delta[0] / length, delta[1] / length];
        let base = [
            end[0] - direction[0] * f64::from(head.length),
            end[1] - direction[1] * f64::from(head.length),
        ];
        let perpendicular = [
            -direction[1] * f64::from(head.width) * 0.5,
            direction[0] * f64::from(head.width) * 0.5,
        ];
        // The head shares this path's single fill slot; see `fill`.
        elements.push(PathElement::MoveTo(end));
        elements.push(PathElement::LineTo([
            base[0] + perpendicular[0],
            base[1] + perpendicular[1],
        ]));
        elements.push(PathElement::LineTo([
            base[0] - perpendicular[0],
            base[1] - perpendicular[1],
        ]));
        elements.push(PathElement::Close);
    }
    (elements, has_head)
}

/// `PreparedSceneCommand::FillPath` carries exactly one path, so an arrow head
/// becomes a subpath of the line's path rather than a second command. The head
/// is a filled triangle, which forces the path's single `fill` to be `Some` even
/// when the author set no fill. Such a head has no colour of its own, so it
/// takes the stroke's colour — the head is the business end of a stroked arrow
/// — and is dropped entirely when neither a fill nor a stroke supplies one.
fn fill(style: &ShapeStyle, has_head: bool) -> Option<PreparedFill> {
    let color = match &style.fill {
        Some(ShapeFill::Solid(color)) => Some(*color),
        None if has_head => style.stroke.as_ref().map(|stroke| stroke.color),
        None => None,
    };
    color.map(|color| PreparedFill::Solid {
        color,
        rule: FillRule::NonZero,
    })
}

fn stroke(style: &StrokeStyle) -> PreparedStroke {
    PreparedStroke {
        color: style.color,
        width: style.width,
        // Scene validation accepts `Some(vec![])` but the prepared path rejects
        // an empty dash list, so an empty-or-filtered-out list means "solid".
        dash: style
            .dash
            .as_ref()
            .map(|dash| {
                dash.iter()
                    .copied()
                    .filter(|length| length.is_finite() && *length > 0.0)
                    .collect::<Vec<_>>()
            })
            .filter(|dash| !dash.is_empty()),
        cap: match style.cap {
            LineCap::Butt => PreparedLineCap::Butt,
            LineCap::Round => PreparedLineCap::Round,
            LineCap::Square => PreparedLineCap::Square,
        },
        join: match style.join {
            LineJoin::Miter => PreparedLineJoin::Miter,
            LineJoin::Round => PreparedLineJoin::Round,
            LineJoin::Bevel => PreparedLineJoin::Bevel,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: f32, height: f32, angle_degrees: f32) -> ShapeFrame {
        ShapeFrame {
            x: 0.0,
            y: 0.0,
            width,
            height,
            angle_degrees,
        }
    }

    fn filled() -> ShapeStyle {
        ShapeStyle {
            fill: Some(ShapeFill::Solid([10, 20, 30, 255])),
            stroke: None,
        }
    }

    /// Applies a Kurbo affine to a local point: `x' = a·x + c·y + e`,
    /// `y' = b·x + d·y + f`.
    fn to_page(affine: [f64; 6], local: [f64; 2]) -> [f64; 2] {
        let [a, b, c, d, e, f] = affine;
        [
            a * local[0] + c * local[1] + e,
            b * local[0] + d * local[1] + f,
        ]
    }

    fn frame_box(x: f32, y: f32, width: f32, height: f32, angle_degrees: f32) -> ShapeFrame {
        ShapeFrame {
            x,
            y,
            width,
            height,
            angle_degrees,
        }
    }

    /// `assert_eq!` on rotated coordinates is too strict: `90°`'s cosine is
    /// `6.1e-17`, not exactly zero, so output carries ~1e-15 of rounding.
    fn assert_close(actual: &[f64; 2], expected: [f64; 2]) {
        assert!(
            (actual[0] - expected[0]).abs() < 1e-9 && (actual[1] - expected[1]).abs() < 1e-9,
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn a_rectangle_becomes_a_closed_four_point_path() {
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &filled(),
            frame(40.0, 20.0, 0.0),
        );
        assert!(matches!(path.fill, Some(PreparedFill::Solid { .. })));
        assert_eq!(path.elements.last(), Some(&PathElement::Close));
    }

    #[test]
    fn a_rectangle_keeps_its_rotation_in_the_transform() {
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &filled(),
            frame(40.0, 20.0, 90.0),
        );
        // 90도 회전은 [cos, sin, -sin, cos, ..] = [0, 1, -1, 0, ..]
        assert!((path.transform[0] - 0.0).abs() < 1e-9);
        assert!((path.transform[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_rotated_frame_maps_its_centre_and_round_trips_local_points() {
        // 90° about (120, 60). The centre is the one point both spaces agree on,
        // and it must land at the centre of the local box, not at the origin.
        let frame = frame_box(100.0, 50.0, 40.0, 20.0, 90.0);
        let centre = to_local(frame, [120.0, 60.0]);
        assert!((centre[0] - 20.0).abs() < 1e-9, "x was {}", centre[0]);
        assert!((centre[1] - 10.0).abs() < 1e-9, "y was {}", centre[1]);

        // local -> page -> local is the identity for every box corner, which no
        // axis-aligned frame can demonstrate.
        let affine = transform(frame);
        for local in [[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]] {
            let round_trip = to_local(frame, to_page(affine, local));
            assert!(
                (round_trip[0] - local[0]).abs() < 1e-9,
                "{round_trip:?} != {local:?}"
            );
            assert!(
                (round_trip[1] - local[1]).abs() < 1e-9,
                "{round_trip:?} != {local:?}"
            );
        }

        // A 90° turn sends the local top-left corner to page (130, 40), so the
        // mapping is a real rotation rather than a translation.
        let corner = to_page(affine, [0.0, 0.0]);
        assert!((corner[0] - 130.0).abs() < 1e-9, "x was {}", corner[0]);
        assert!((corner[1] - 40.0).abs() < 1e-9, "y was {}", corner[1]);
    }

    #[test]
    fn a_thirty_degree_frame_round_trips_its_corners() {
        // 30° exercises both sin and cos non-trivially, so a sign error in
        // `to_local` that 90° happens to mask cannot survive this.
        let frame = frame_box(7.0, 11.0, 30.0, 15.0, 30.0);
        let affine = transform(frame);
        for local in [[0.0, 0.0], [30.0, 0.0], [30.0, 15.0], [0.0, 15.0]] {
            let round_trip = to_local(frame, to_page(affine, local));
            assert!(
                (round_trip[0] - local[0]).abs() < 1e-9,
                "{round_trip:?} != {local:?}"
            );
            assert!(
                (round_trip[1] - local[1]).abs() < 1e-9,
                "{round_trip:?} != {local:?}"
            );
        }
    }

    #[test]
    fn a_rotated_polygon_maps_page_points_into_the_frame() {
        // A 90° frame about (120, 60) has the page mapping `(130 - y, 40 + x)`
        // for local `(x, y)`, so these page points are exactly the local corners.
        let points = vec![
            koharu_scene::Point { x: 130.0, y: 40.0 },
            koharu_scene::Point { x: 130.0, y: 80.0 },
            koharu_scene::Point { x: 110.0, y: 80.0 },
            koharu_scene::Point { x: 110.0, y: 40.0 },
        ];
        let path = polygon_path(&points, &filled(), frame_box(100.0, 50.0, 40.0, 20.0, 90.0));
        let coordinates: Vec<[f64; 2]> = path
            .elements
            .iter()
            .filter_map(|element| match element {
                PathElement::MoveTo(values) | PathElement::LineTo(values) => Some(*values),
                PathElement::Close => None,
                other => panic!("unexpected element {other:?}"),
            })
            .collect();
        assert_eq!(coordinates.len(), 4);
        let expected = [[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]];
        for (actual, wanted) in coordinates.iter().zip(expected) {
            assert!(
                (actual[0] - wanted[0]).abs() < 1e-9,
                "{actual:?} != {wanted:?}"
            );
            assert!(
                (actual[1] - wanted[1]).abs() < 1e-9,
                "{actual:?} != {wanted:?}"
            );
        }
        assert_eq!(path.elements.last(), Some(&PathElement::Close));
    }

    #[test]
    fn an_unfilled_rectangle_has_no_fill() {
        let style = ShapeStyle {
            fill: None,
            stroke: None,
        };
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &style,
            frame(40.0, 20.0, 0.0),
        );
        assert!(path.fill.is_none());
    }

    #[test]
    fn a_zero_length_line_produces_finite_coordinates() {
        let style = ShapeStyle {
            fill: None,
            stroke: Some(StrokeStyle {
                color: [0, 0, 0, 255],
                width: 2.0,
                dash: None,
                cap: LineCap::Round,
                join: LineJoin::Round,
            }),
        };
        let path = shape_path(
            &ShapeGeometry::Line {
                start: [5.0, 5.0],
                end: [5.0, 5.0],
                head: None,
            },
            &style,
            frame(40.0, 20.0, 0.0),
        );
        for element in &path.elements {
            let values: &[f64] = match element {
                PathElement::MoveTo(v) | PathElement::LineTo(v) => v,
                PathElement::QuadTo(v) => v,
                PathElement::CurveTo(v) => v,
                PathElement::Close => continue,
            };
            assert!(values.iter().all(|value| value.is_finite()));
        }
        assert!(path.validate().is_ok());
    }

    #[test]
    fn an_arrow_head_adds_a_filled_triangle_element() {
        let style = ShapeStyle {
            fill: None,
            stroke: Some(StrokeStyle {
                color: [0, 0, 0, 255],
                width: 2.0,
                dash: None,
                cap: LineCap::Butt,
                join: LineJoin::Miter,
            }),
        };
        let path = shape_path(
            &ShapeGeometry::Line {
                start: [0.0, 0.0],
                end: [50.0, 0.0],
                head: Some(ArrowHead {
                    length: 10.0,
                    width: 8.0,
                }),
            },
            &style,
            frame(100.0, 100.0, 0.0),
        );
        // 화살촉은 채워진 삼각형이므로 두 번째 FillPath가 필요하다.
        // 이 함수는 하나의 PreparedPath만 돌려주므로, 화살촉은 같은
        // 경로에 별도 subpath로 들어가고 fill이 함께 켜진다.
        assert!(matches!(path.fill, Some(PreparedFill::Solid { .. })));
        // The line runs along +x in local space, so the head is the triangle
        // (end), (end - length·dir ± width/2·perp) = (50, 0), (40, 4), (40, -4).
        assert_eq!(
            &path.elements[3..],
            [
                PathElement::MoveTo([50.0, 0.0]),
                PathElement::LineTo([40.0, 4.0]),
                PathElement::LineTo([40.0, -4.0]),
                PathElement::Close,
            ]
        );
    }

    #[test]
    fn an_arrow_head_keeps_its_base_and_half_width_on_a_rotated_line() {
        // A 90° frame about (50, 50) maps local (x, y) to page (100 - y, x), so
        // this horizontal page line is local (50, 0) -> (50, 100): straight
        // down. The base therefore sits 10 above the tip and the half-width of 4
        // falls on the local x axis.
        let style = ShapeStyle {
            fill: None,
            stroke: Some(StrokeStyle {
                color: [0, 0, 0, 255],
                width: 2.0,
                dash: None,
                cap: LineCap::Butt,
                join: LineJoin::Miter,
            }),
        };
        let path = shape_path(
            &ShapeGeometry::Line {
                start: [100.0, 50.0],
                end: [0.0, 50.0],
                head: Some(ArrowHead {
                    length: 10.0,
                    width: 8.0,
                }),
            },
            &style,
            frame_box(0.0, 0.0, 100.0, 100.0, 90.0),
        );
        assert_eq!(path.elements.len(), 7);
        assert!(matches!(path.elements[0], PathElement::MoveTo(_)));
        assert!(matches!(path.elements[1], PathElement::LineTo(_)));
        assert!(matches!(path.elements[2], PathElement::Close));
        let PathElement::MoveTo(tip) = path.elements[3] else {
            panic!("the head starts with a MoveTo");
        };
        let PathElement::LineTo(base_a) = path.elements[4] else {
            panic!("the head side is a LineTo");
        };
        let PathElement::LineTo(base_b) = path.elements[5] else {
            panic!("the head side is a LineTo");
        };
        assert_eq!(path.elements[6], PathElement::Close);
        assert_close(&tip, [50.0, 100.0]);
        assert_close(&base_a, [46.0, 90.0]);
        assert_close(&base_b, [54.0, 90.0]);
    }

    #[test]
    fn shape_layer_path_delegates_polygons_to_the_points() {
        let points = vec![
            koharu_scene::Point { x: 0.0, y: 0.0 },
            koharu_scene::Point { x: 10.0, y: 0.0 },
            koharu_scene::Point { x: 5.0, y: 10.0 },
        ];
        let path = shape_layer_path(
            &ShapeGeometry::Polygon,
            &points,
            &filled(),
            frame(20.0, 20.0, 0.0),
        );
        assert!(path.elements.len() >= 4); // 3 MoveTo/LineTo + Close
    }

    #[test]
    fn a_polygon_without_points_produces_an_empty_path() {
        let path = shape_layer_path(
            &ShapeGeometry::Freeform,
            &[],
            &filled(),
            frame(20.0, 20.0, 0.0),
        );
        assert_eq!(path.elements, vec![PathElement::MoveTo([0.0, 0.0])]);
    }

    #[test]
    fn an_empty_dash_list_becomes_a_solid_stroke() {
        let style = ShapeStyle {
            fill: None,
            stroke: Some(StrokeStyle {
                color: [0, 0, 0, 255],
                width: 2.0,
                dash: Some(Vec::new()),
                cap: LineCap::Butt,
                join: LineJoin::Miter,
            }),
        };
        let path = shape_path(&ShapeGeometry::Ellipse, &style, frame(40.0, 20.0, 0.0));
        let stroke = path.stroke.as_ref().expect("the stroke is carried over");
        assert!(stroke.dash.is_none());
        assert!(path.validate().is_ok());
    }
}
