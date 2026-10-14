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
    FillRule, PathElement, PreparedFill, PreparedGradient, PreparedGradientExtend,
    PreparedGradientKind, PreparedGradientStop, PreparedLineCap, PreparedLineJoin, PreparedPath,
    PreparedPattern, PreparedPatternImage, PreparedPatternRepeat, PreparedStroke,
};
use koharu_scene::{
    ArrowHead, Geometry, GradientExtend, GradientKind, LineCap, LineJoin, PatternRepeat, ShapeFill,
    ShapeGeometry, ShapeStyle, StrokeStyle,
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
        fill: fill(style, frame, has_head),
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
///
/// This is the **only** place a persisted gradient's normalized frame-local
/// coordinates (`0..=1`) become the path-local coordinates that
/// [`PreparedPath::elements`] and the prepared gradient share. Every path — the
/// parameterized kinds and the polygon/freeform delegate alike — reaches the
/// prepared fill through here, so a second spread site would give the control
/// points two authorities and let them drift, the same class of bug as 1a's
/// `Line` double-store. Do not repeat the multiplication anywhere downstream.
///
/// The radius is a fraction of the frame's **width**, while `Sweep` angles are
/// already absolute degrees and pass through untouched. The result stays in the
/// path's local frame: rotation lives in [`PreparedPath::transform`] and the
/// renderer applies it to the brush automatically.
fn fill(style: &ShapeStyle, frame: ShapeFrame, has_head: bool) -> Option<PreparedFill> {
    match &style.fill {
        Some(ShapeFill::Solid(color)) => Some(PreparedFill::Solid {
            color: *color,
            rule: FillRule::NonZero,
        }),
        Some(ShapeFill::Gradient(spec)) => {
            // Normalized frame-local (0..=1) -> path-local. Both control points
            // and the width-relative radius are scaled here and nowhere else.
            let local = |point: [f32; 2]| [point[0] * frame.width, point[1] * frame.height];
            let kind = match &spec.kind {
                GradientKind::Linear { start, end } => PreparedGradientKind::Linear {
                    start: local(*start),
                    end: local(*end),
                },
                GradientKind::Radial { center, radius } => PreparedGradientKind::Radial {
                    center: local(*center),
                    // The radius is a fraction of the frame's width, not of
                    // its height or its smaller or larger extent.
                    radius: *radius * frame.width,
                },
                GradientKind::Sweep {
                    center,
                    start_degrees,
                    end_degrees,
                } => PreparedGradientKind::Sweep {
                    center: local(*center),
                    // Angles are absolute, not normalized: no frame factor.
                    start_degrees: *start_degrees,
                    end_degrees: *end_degrees,
                },
            };
            Some(PreparedFill::Gradient {
                gradient: PreparedGradient {
                    kind,
                    // Left in the author's order; the renderer sorts stably by
                    // offset.
                    stops: spec
                        .stops
                        .iter()
                        .map(|stop| PreparedGradientStop {
                            offset: stop.offset,
                            color: stop.color,
                        })
                        .collect(),
                    extend: match spec.extend {
                        GradientExtend::Pad => PreparedGradientExtend::Pad,
                        GradientExtend::Repeat => PreparedGradientExtend::Repeat,
                        GradientExtend::Reflect => PreparedGradientExtend::Reflect,
                    },
                },
                rule: FillRule::NonZero,
            })
        }
        Some(ShapeFill::Pattern(spec)) => Some(PreparedFill::Pattern {
            pattern: PreparedPattern {
                repeat: match spec.repeat {
                    PatternRepeat::Tile => PreparedPatternRepeat::Tile,
                    PatternRepeat::Stretch => PreparedPatternRepeat::Stretch,
                },
                // Normalized frame-local (0..=1) -> path-local, the same spread
                // rule and the same single site as the gradient above. The frame
                // box's top-left *is* the path-local origin, so `frame.x` /
                // `frame.y` must not be added here — the gradient adds neither,
                // and adding them would offset the pattern twice.
                origin: [spec.origin[0] * frame.width, spec.origin[1] * frame.height],
                // The tile pitch is a fraction of the frame's **width**, the
                // same convention as the gradient's radial radius.
                scale: spec.scale * frame.width,
                // The pattern's own rotation, carried for
                // `PreparedPattern::brush_transform`. It is deliberately not the
                // shape's `angle_degrees`: this path's transform already rotates
                // by that angle, and folding it in would rotate the pattern
                // twice. The conversion is exact — `rotation_degrees` is an
                // integer in the saved spec.
                rotation_degrees: spec.rotation_degrees as f32,
                // P-D6's stand-in colour. `append_to` applies it when the asset
                // is missing, undecodable, or empty.
                fallback: spec.fallback,
                // `Stretch` fits one image to this box.
                extent: [frame.width, frame.height],
                // `PatternSpec` references no asset, so the caller that resolves
                // the owning layer's `pattern` asset fills this in through
                // `attach_pattern_image`.
                image: None,
            },
            rule: FillRule::NonZero,
        }),
        None if has_head => style.stroke.as_ref().map(|stroke| PreparedFill::Solid {
            color: stroke.color,
            rule: FillRule::NonZero,
        }),
        None => None,
    }
}

/// Attaches decoded pattern pixels to a prepared path whose fill is a pattern.
///
/// Returns whether an image was attached. A caller that gets `false` must leave
/// the fill alone rather than substituting a different one: the fallback colour
/// is drawn by `PreparedPath::append_to`, which is the single place that decides
/// what "no image" looks like, so the committed render and the browser preview
/// cannot diverge.
///
/// `PatternSpec` deliberately references no asset, so the layer that owns the
/// shape owns the `pattern` asset and the caller resolving it is the only party
/// that can supply the pixels. Every renderer — the native frame builder, shape
/// rasterization, and the WASM preview — funnels through this one function
/// rather than growing its own attachment path.
pub fn attach_pattern_image(path: &mut PreparedPath, image: PreparedPatternImage) -> bool {
    match &mut path.fill {
        Some(PreparedFill::Pattern { pattern, .. }) => {
            pattern.image = Some(image);
            true
        }
        _ => false,
    }
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
    use koharu_scene::{GradientSpec, GradientStop, PatternSpec};

    fn frame(width: f32, height: f32, angle_degrees: f32) -> ShapeFrame {
        ShapeFrame {
            x: 0.0,
            y: 0.0,
            width,
            height,
            angle_degrees,
        }
    }

    fn pattern_style(repeat: PatternRepeat) -> ShapeStyle {
        ShapeStyle {
            fill: Some(ShapeFill::Pattern(PatternSpec {
                repeat,
                origin: [0.5, 0.25],
                scale: 0.5,
                rotation_degrees: 30,
                fallback: [1, 2, 3, 255],
            })),
            stroke: None,
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

    fn gradient_style() -> ShapeStyle {
        ShapeStyle {
            fill: Some(ShapeFill::Gradient(GradientSpec {
                kind: GradientKind::Linear {
                    start: [0.0, 0.0],
                    end: [1.0, 1.0],
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
                extend: GradientExtend::Pad,
            })),
            stroke: None,
        }
    }

    #[test]
    fn a_normalized_gradient_is_spread_onto_the_frame() {
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &gradient_style(),
            frame(200.0, 100.0, 0.0),
        );
        let Some(PreparedFill::Gradient { gradient, .. }) = path.fill else {
            panic!("a gradient fill must survive preparation");
        };
        let PreparedGradientKind::Linear { start, end } = gradient.kind else {
            panic!("the linear kind must survive preparation");
        };
        // 정규화 (0,0)->(1,1) 이 200x100 프레임에서 (0,0)->(200,100) 이 된다.
        assert!((start[0] - 0.0).abs() < 1e-6);
        assert!((end[0] - 200.0).abs() < 1e-6);
        assert!((end[1] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn a_rotated_shape_keeps_its_gradient_in_local_coordinates() {
        // 회전은 경로 변환(`transform`)이 담당하고 vello 가 그것을 브러시에
        // 자동 적용하므로, 준비된 그라디언트는 회전한 도형에서도 로컬 좌표에
        // 남아 있어야 한다.
        //
        // Rectangle 을 쓰는 이유: Line 의 끝점은 `to_local` 로 각도에 따라
        // 바뀌므로 `Line` 으로는 이 단언이 성립하지 않는다. 그라디언트 좌표는
        // 프레임의 폭·높이에만 의존하고 각도에는 의존하지 않는다.
        let upright = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &gradient_style(),
            frame(200.0, 100.0, 0.0),
        );
        let rotated = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &gradient_style(),
            frame(200.0, 100.0, 90.0),
        );
        assert_eq!(upright.fill, rotated.fill);
        assert_ne!(upright.transform, rotated.transform);
    }

    #[test]
    fn a_gradient_fill_with_a_none_fill_stays_none() {
        let style = ShapeStyle {
            fill: None,
            stroke: None,
        };
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &style,
            frame(200.0, 100.0, 0.0),
        );
        assert!(path.fill.is_none());
    }

    #[test]
    fn an_arrow_head_gradient_uses_the_stroke_colour_when_fill_is_absent() {
        let style = ShapeStyle {
            fill: None,
            stroke: Some(StrokeStyle {
                color: [9, 9, 9, 255],
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
        assert_eq!(
            path.fill,
            Some(PreparedFill::Solid {
                color: [9, 9, 9, 255],
                rule: FillRule::NonZero
            })
        );
    }

    #[test]
    fn a_radial_gradient_radius_scales_by_the_frame_width() {
        // 반지름은 프레임 폭 기준이다. 높이 기준으로 잘못 펴면 이 테스트가
        // 잡는다 (200x100 프레임에서 0.25 -> 50).
        let mut style = gradient_style();
        style.fill = Some(ShapeFill::Gradient(GradientSpec {
            kind: GradientKind::Radial {
                center: [0.5, 0.5],
                radius: 0.25,
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
            extend: GradientExtend::Repeat,
        }));
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &style,
            frame(200.0, 100.0, 0.0),
        );
        let Some(PreparedFill::Gradient { gradient, .. }) = path.fill else {
            panic!("a radial gradient must survive preparation");
        };
        let PreparedGradientKind::Radial { center, radius } = gradient.kind else {
            panic!("the radial kind must survive preparation");
        };
        assert!((center[0] - 100.0).abs() < 1e-6);
        assert!((center[1] - 50.0).abs() < 1e-6);
        assert!((radius - 50.0).abs() < 1e-6);
        assert_eq!(gradient.extend, PreparedGradientExtend::Repeat);
    }

    #[test]
    fn a_sweep_gradient_passes_its_angles_through_unchanged() {
        // 각도는 정규화 대상이 아니다. 프레임 크기로 곱하면 0..360 스윕이
        // 터무니없이 커진다.
        let mut style = gradient_style();
        style.fill = Some(ShapeFill::Gradient(GradientSpec {
            kind: GradientKind::Sweep {
                center: [0.5, 0.5],
                start_degrees: 90.0,
                end_degrees: 270.0,
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
        }));
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &style,
            frame(200.0, 100.0, 0.0),
        );
        let Some(PreparedFill::Gradient { gradient, .. }) = path.fill else {
            panic!("a sweep gradient must survive preparation");
        };
        let PreparedGradientKind::Sweep {
            center,
            start_degrees,
            end_degrees,
        } = gradient.kind
        else {
            panic!("the sweep kind must survive preparation");
        };
        assert!((center[0] - 100.0).abs() < 1e-6);
        assert!((center[1] - 50.0).abs() < 1e-6);
        assert!((start_degrees - 90.0).abs() < 1e-6);
        assert!((end_degrees - 270.0).abs() < 1e-6);
        assert_eq!(gradient.extend, PreparedGradientExtend::Reflect);
    }

    #[test]
    fn a_gradient_carries_its_stops_through_unsorted() {
        // 정렬은 렌더러의 몫이므로 준비 단계는 순서를 건드리지 않는다.
        let mut style = gradient_style();
        style.fill = Some(ShapeFill::Gradient(GradientSpec {
            kind: GradientKind::Linear {
                start: [0.0, 0.0],
                end: [1.0, 0.0],
            },
            stops: vec![
                GradientStop {
                    offset: 0.75,
                    color: [1, 2, 3, 255],
                },
                GradientStop {
                    offset: 0.0,
                    color: [4, 5, 6, 255],
                },
            ],
            extend: GradientExtend::Pad,
        }));
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &style,
            frame(200.0, 100.0, 0.0),
        );
        let Some(PreparedFill::Gradient { gradient, .. }) = path.fill else {
            panic!("a gradient must survive preparation");
        };
        assert_eq!(
            gradient.stops,
            vec![
                PreparedGradientStop {
                    offset: 0.75,
                    color: [1, 2, 3, 255],
                },
                PreparedGradientStop {
                    offset: 0.0,
                    color: [4, 5, 6, 255],
                },
            ]
        );
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

    /// The pattern's coordinates are **path-local**, exactly like the gradient's:
    /// the frame box's top-left is the path-local origin, so `frame.x` and
    /// `frame.y` must not be added. A frame at `(10, 20)` therefore still yields
    /// an origin of `(100, 25)`, not `(110, 45)`. Adding the offset here would
    /// apply it a second time through the path transform.
    #[test]
    fn a_pattern_origin_is_spread_onto_the_frame() {
        let style = pattern_style(PatternRepeat::Tile);
        let Some(PreparedFill::Pattern { pattern, .. }) =
            fill(&style, frame_box(10.0, 20.0, 200.0, 100.0, 0.0), false)
        else {
            panic!("expected a pattern fill");
        };
        assert_eq!(pattern.origin, [0.5 * 200.0, 0.25 * 100.0]);
        assert_eq!(pattern.scale, 0.5 * 200.0);
        assert_eq!(pattern.extent, [200.0, 100.0]);
        assert_eq!(pattern.fallback, [1, 2, 3, 255]);
    }

    /// The shape's angle lives in the path transform, so the pattern's own
    /// coordinates must be identical at any angle. Folding the angle in here
    /// would rotate the pattern twice.
    #[test]
    fn a_pattern_keeps_its_coordinates_local_when_rotated() {
        let style = pattern_style(PatternRepeat::Tile);
        let upright = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &style,
            frame(200.0, 100.0, 0.0),
        );
        let rotated = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &style,
            frame(200.0, 100.0, 90.0),
        );
        assert_eq!(upright.fill, rotated.fill);
        assert_ne!(upright.transform, rotated.transform);
    }

    #[test]
    fn a_stretch_pattern_reports_the_stretch_repeat() {
        let style = pattern_style(PatternRepeat::Stretch);
        let Some(PreparedFill::Pattern { pattern, .. }) = fill(&style, frame(40.0, 20.0, 0.0), false)
        else {
            panic!("expected a pattern fill");
        };
        assert_eq!(pattern.repeat, PreparedPatternRepeat::Stretch);
    }

    /// `PatternSpec` references no asset (the owning layer does), so `fill()` has
    /// nothing to put in `image` and must not invent one. The pixels arrive
    /// through `attach_pattern_image`, and until then `append_to` draws the
    /// fallback.
    #[test]
    fn a_pattern_carries_no_image_until_its_asset_is_attached() {
        let style = pattern_style(PatternRepeat::Tile);
        let path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &style,
            frame(40.0, 20.0, 0.0),
        );
        let Some(PreparedFill::Pattern { pattern, .. }) = path.fill else {
            panic!("expected a pattern fill");
        };
        assert!(pattern.image.is_none());
    }

    #[test]
    fn attaching_a_pattern_image_fills_the_pattern_fill() {
        let style = pattern_style(PatternRepeat::Tile);
        let mut path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &style,
            frame(40.0, 20.0, 0.0),
        );
        let image = PreparedPatternImage {
            width: 1,
            height: 1,
            pixels: std::sync::Arc::from(&[9u8, 8, 7, 255][..]),
        };
        assert!(attach_pattern_image(&mut path, image.clone()));
        let Some(PreparedFill::Pattern { pattern, .. }) = path.fill else {
            panic!("expected a pattern fill");
        };
        assert_eq!(pattern.image, Some(image));
    }

    /// The attachment point is the pattern fill and nothing else, so a caller
    /// cannot accidentally turn a solid or gradient fill into a pattern.
    #[test]
    fn attaching_a_pattern_image_rejects_a_non_pattern_fill() {
        let mut path = shape_path(
            &ShapeGeometry::Rectangle { corner_radius: 0.0 },
            &filled(),
            frame(40.0, 20.0, 0.0),
        );
        let before = path.fill.clone();
        let image = PreparedPatternImage {
            width: 1,
            height: 1,
            pixels: std::sync::Arc::from(&[9u8, 8, 7, 255][..]),
        };
        assert!(!attach_pattern_image(&mut path, image));
        assert_eq!(path.fill, before);
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
