//! Text layout and portable glyph recording.

use std::sync::Arc;

use anyhow::Result;
use i_overlay::{
    core::{fill_rule::FillRule as OverlayFillRule, overlay_rule::OverlayRule},
    float::{simplify::SimplifyShape, single::SingleFloatOverlay},
    mesh::float::{
        outline::offset::OutlineOffset,
        style::{LineJoin as OverlayJoin, OutlineStyle},
    },
};
use koharu_rasterizer::{
    FillRule, PathElement, PreparedFill, PreparedGlyph, PreparedGlyphRun, PreparedPath,
    PreparedResource, PreparedScene, PreparedSceneCommand, ResourceId,
};
use koharu_scene::{LineJoin, StrokeMethod, StrokePosition};
use skrifa::{
    GlyphId, MetadataProvider,
    instance::Size,
    outline::{DrawSettings, OutlinePen},
};
use vello::kurbo::{Affine, BezPath, Diagonal2, Join, PathEl, Rect, Shape, expand_path, flatten};

use crate::{
    Error, FontStyle, HyphenationPolicy, LayoutRun, RenderBounds, RenderDiagnostic,
    Result as RenderResult, TextAlign, TextLayout, WritingMode,
    bubble::LayoutBox,
    fonts::{Fonts, font_key},
    script::is_chinese_or_japanese_text,
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TextNodeDescriptor {
    pub(crate) entity: koharu_scene::EntityId,
    pub(crate) text: String,
    pub(crate) language: Option<koharu_scene::LanguageTag>,
    pub(crate) width: f32,
    pub(crate) height: f32,
    pub(crate) balloon_contour: Option<Vec<(f32, f32)>>,
    pub(crate) preferred_font: Option<String>,
    pub(crate) font_families: Vec<String>,
    pub(crate) font_weight: Option<u16>,
    pub(crate) font_style: Option<FontStyle>,
    pub(crate) font_size: Option<f32>,
    pub(crate) minimum_font_size: f32,
    pub(crate) auto_fit: bool,
    pub(crate) alignment: TextAlign,
    pub(crate) writing_mode: WritingMode,
    pub(crate) foreground_color: [u8; 4],
    pub(crate) stroke: Option<StrokeOptions>,
    pub(crate) line_height: f32,
    pub(crate) letter_spacing: f32,
    pub(crate) word_spacing: f32,
    pub(crate) text_inset: [f32; 4],
    pub(crate) point_text: bool,
}

pub(crate) struct RenderedTextNode {
    pub(crate) scene: Arc<PreparedScene>,
    pub(crate) resources: Arc<[PreparedResource]>,
    pub(crate) local_bounds: RenderBounds,
    pub(crate) metadata: RenderedTextMetadata,
    pub(crate) diagnostics: Vec<RenderDiagnostic>,
}

pub(crate) struct RenderedTextMetadata {
    pub(crate) rendered_bounds: RenderBounds,
    pub(crate) layout_bounds: RenderBounds,
    pub(crate) post_script_fonts: Vec<String>,
    pub(crate) font_size: f32,
    pub(crate) color: [u8; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StrokeOptions {
    pub color: [u8; 4],
    pub width_px: f32,
    pub method: StrokeMethod,
    pub position: StrokePosition,
    pub join: LineJoin,
    /// A second outline drawn immediately outside the first one's outer edge.
    pub outer: Option<OuterStroke>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OuterStroke {
    pub color: [u8; 4],
    pub width_px: f32,
}

impl StrokeOptions {
    /// The second outline, when it is wide enough and visible.
    fn visible_outer(&self) -> Option<OuterStroke> {
        self.outer
            .filter(|outer| outer.width_px > 0.0 && outer.color[3] > 0)
    }

    /// How far the first outline reaches beyond the glyph edge.
    fn outward_extent(&self) -> f64 {
        let width = f64::from(self.width_px);
        match (self.method, self.position) {
            (StrokeMethod::Expansion, _) | (_, StrokePosition::Outside) => width,
            (_, StrokePosition::Center) => width * 0.5,
            (_, StrokePosition::Inside) => 0.0,
        }
    }
}

/// Paint options used when recording one laid-out text run into a Vello scene.
#[derive(Debug, Clone)]
pub(crate) struct TextRenderOptions {
    pub color: [u8; 4],
    pub hint_glyphs: bool,
    pub padding: f32,
    pub baseline_shift: f32,
    pub stroke: Option<StrokeOptions>,
}

impl Default for TextRenderOptions {
    fn default() -> Self {
        Self {
            color: [0, 0, 0, 255],
            hint_glyphs: true,
            padding: 0.0,
            baseline_shift: 0.0,
            stroke: None,
        }
    }
}

/// Shapes text and records the resulting glyphs into vector scenes.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TextRenderer;

impl TextRenderer {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    pub(crate) fn layout<'a>(&self, builder: &TextLayout<'a>, text: &str) -> Result<LayoutRun<'a>> {
        builder.run(text)
    }

    pub(crate) fn render(
        &self,
        scene: &mut PreparedScene,
        resources: &mut Vec<PreparedResource>,
        layout: &LayoutRun<'_>,
        writing_mode: WritingMode,
        options: &TextRenderOptions,
        transform: Affine,
    ) -> Result<Option<Rect>> {
        if let Some(stroke) = options.stroke.filter(|stroke| {
            stroke.method == StrokeMethod::Contour && stroke.width_px > 0.0 && stroke.color[3] > 0
        }) {
            let contours = text_contours(layout, options)?;
            let mut fill_options = options.clone();
            // The contour and fill must use the same unhinted geometry at every zoom level.
            fill_options.hint_glyphs = false;
            if contours.is_empty() {
                draw_layout(
                    scene,
                    resources,
                    layout,
                    writing_mode,
                    &fill_options,
                    transform,
                    GlyphPaint {
                        color: options.color,
                        dilation_px: 0.0,
                    },
                );
                return Ok(None);
            }
            // Opaque letters cover the expanded region's interior. Subtracting the letters first
            // would antialias their shared edge twice and expose the background. Translucent
            // letters need an outside-only region to preserve their alpha instead.
            let behind_fill = stroke.position == StrokePosition::Outside && options.color[3] == 255;
            let border = contour_border(&contours, stroke, behind_fill);
            let mut bounds =
                (!border.is_empty()).then(|| transform.transform_rect_bbox(border.bounding_box()));
            if let Some(outer) = stroke.visible_outer() {
                // Everything drawn over a solid outer region covers its interior, so it can be
                // filled whole without a shared antialiased seam against the first outline.
                let solid = options.color[3] == 255 && stroke.color[3] == 255;
                let region = outer_border(&contours, stroke, outer, solid);
                if !region.is_empty() {
                    let outer_bounds = transform.transform_rect_bbox(region.bounding_box());
                    bounds = Some(bounds.map_or(outer_bounds, |inner| inner.union(outer_bounds)));
                    scene
                        .commands
                        .push(fill_command(&region, transform, outer.color));
                }
            }
            let command = fill_command(&border, transform, stroke.color);
            if behind_fill && !border.is_empty() {
                scene.commands.push(command.clone());
            }
            draw_layout(
                scene,
                resources,
                layout,
                writing_mode,
                &fill_options,
                transform,
                GlyphPaint {
                    color: options.color,
                    dilation_px: 0.0,
                },
            );
            if !behind_fill && !border.is_empty() {
                scene.commands.push(command);
            }
            return Ok(bounds);
        }
        // The border is drawn first as an outward-only dilation of the glyph outline (see
        // `draw_layout`), then the ordinary fill is drawn on top in the foreground color. Unlike a
        // centered stroke, a dilation never grows inward, so it can't punch a hole through small
        // features (e.g. dots) once the border width exceeds their radius.
        if let Some(stroke) = options
            .stroke
            .filter(|stroke| stroke.width_px > 0.0 && stroke.color[3] > 0)
        {
            if let Some(outer) = stroke.visible_outer() {
                draw_layout(
                    scene,
                    resources,
                    layout,
                    writing_mode,
                    options,
                    transform,
                    GlyphPaint {
                        color: outer.color,
                        dilation_px: stroke.width_px + outer.width_px,
                    },
                );
            }
            draw_layout(
                scene,
                resources,
                layout,
                writing_mode,
                options,
                transform,
                GlyphPaint {
                    color: stroke.color,
                    dilation_px: stroke.width_px,
                },
            );
        }
        draw_layout(
            scene,
            resources,
            layout,
            writing_mode,
            options,
            transform,
            GlyphPaint {
                color: options.color,
                dilation_px: 0.0,
            },
        );
        Ok(None)
    }

    pub(crate) fn render_descriptor(
        &self,
        descriptor: &TextNodeDescriptor,
        fonts: &Fonts,
    ) -> RenderResult<RenderedTextNode> {
        let is_bubble_text = descriptor.balloon_contour.is_some();
        let frame = LayoutBox {
            x: 0.0,
            y: 0.0,
            width: descriptor.width,
            height: descriptor.height,
        };
        let bounds = if is_bubble_text {
            inset(frame, descriptor.text_inset)
        } else {
            frame
        };
        if bounds.width <= 0.0 || bounds.height <= 0.0 {
            return Err(Error::invalid(format!(
                "text inset leaves no layout area for entity {}",
                descriptor.entity
            )));
        }
        let fonts = fonts
            .resolve(
                descriptor.preferred_font.as_deref(),
                descriptor.font_weight,
                descriptor.font_style,
                &descriptor.font_families,
                &descriptor.text,
                descriptor
                    .language
                    .as_ref()
                    .map(koharu_scene::LanguageTag::as_str),
            )
            .map_err(|source| Error::Font {
                entity: descriptor.entity,
                source,
            })?;
        let automatic_maximum = || automatic_maximum(descriptor, bounds, is_bubble_text);
        let maximum = if descriptor.auto_fit {
            automatic_maximum()
        } else {
            descriptor.font_size.unwrap_or_else(automatic_maximum)
        };
        let minimum = descriptor.minimum_font_size.min(maximum);
        let mut layout = TextLayout::new(&fonts[0])
            .with_fallback_fonts(&fonts[1..])
            .with_writing_mode(descriptor.writing_mode)
            .with_alignment(descriptor.alignment)
            .with_line_height(descriptor.line_height)
            .with_spacing(descriptor.letter_spacing, descriptor.word_spacing)
            .with_cjk_punctuation_layout(
                is_chinese_or_japanese_text(&descriptor.text)
                    || descriptor
                        .language
                        .as_ref()
                        .is_some_and(|language| is_chinese_or_japanese_language(language.as_str())),
            );
        if !descriptor.point_text {
            layout = layout
                .with_max_width(bounds.width)
                .with_max_height(bounds.height);
        }
        if let Some(contour) = &descriptor.balloon_contour {
            let [top, _, _, left] = descriptor.text_inset;
            let contour = contour.iter().map(|&(x, y)| (x - left, y - top)).collect();
            layout = layout.with_comic_balloon(
                bounds.width,
                bounds.height,
                contour,
                descriptor.text_inset.into_iter().fold(0.0, f32::max),
            );
        }
        if let Some(language) = &descriptor.language {
            layout = layout.with_hyphenation_language_tag(language.as_str());
        }
        if is_bubble_text && descriptor.writing_mode == WritingMode::Horizontal {
            layout = layout.with_hyphenation_policy(HyphenationPolicy::LastResort);
        }
        let layout = if descriptor.auto_fit && !descriptor.point_text {
            layout
                .with_max_font_size(maximum)
                .with_min_font_size(minimum)
        } else {
            layout.with_font_size(descriptor.font_size.unwrap_or(maximum))
        };
        let layout = self
            .layout(&layout, &descriptor.text)
            .map_err(|source| Error::Layout {
                entity: descriptor.entity,
                source,
            })?;
        let (mut x, mut y) = if descriptor.point_text {
            (bounds.x, bounds.y)
        } else {
            placement(bounds, layout.width, layout.height)
        };
        x += layout.placement_offset_x();
        y += layout.placement_offset_y();
        let transform = Affine::translate((f64::from(x), f64::from(y)));
        let color = descriptor.foreground_color;
        let mut options = TextRenderOptions {
            color,
            stroke: None,
            ..TextRenderOptions::default()
        };
        let mut scene = PreparedScene::default();
        let mut resources = Vec::new();
        if let Some(stroke) = descriptor.stroke {
            options.stroke = Some(stroke);
        }
        let contour_bounds = self
            .render(
                &mut scene,
                &mut resources,
                &layout,
                descriptor.writing_mode,
                &options,
                transform,
            )
            .map_err(|source| Error::Layout {
                entity: descriptor.entity,
                source,
            })?;
        let mut diagnostics = Vec::new();
        if layout.font_size + f32::EPSILON < descriptor.minimum_font_size {
            diagnostics.push(RenderDiagnostic::TextBelowReadableSize {
                entity: descriptor.entity,
                font_size: layout.font_size,
                minimum_font_size: descriptor.minimum_font_size,
            });
        }
        if layout.overflowed() {
            diagnostics.push(RenderDiagnostic::TextOverflow {
                entity: descriptor.entity,
                available: bounds.into(),
                actual_width: layout.width,
                actual_height: layout.height,
                font_size: layout.font_size,
            });
        }
        let rendered_bounds = RenderBounds {
            x,
            y,
            width: layout.width,
            height: layout.height,
        };
        let stroke_padding = descriptor
            .stroke
            .filter(|stroke| stroke.method == StrokeMethod::Expansion)
            .map_or(0.0, |stroke| {
                stroke.width_px.max(0.0)
                    + stroke
                        .visible_outer()
                        .map_or(0.0, |outer| outer.width_px.max(0.0))
            });
        let mut local_bounds = Rect::new(
            f64::from(rendered_bounds.x - stroke_padding),
            f64::from(rendered_bounds.y - stroke_padding),
            f64::from(rendered_bounds.x + rendered_bounds.width + stroke_padding),
            f64::from(rendered_bounds.y + rendered_bounds.height + stroke_padding),
        );
        if let Some(contour_bounds) = contour_bounds {
            local_bounds = local_bounds.union(contour_bounds);
        }
        Ok(RenderedTextNode {
            scene: Arc::new(scene),
            resources: resources.into(),
            local_bounds: RenderBounds {
                x: local_bounds.x0 as f32,
                y: local_bounds.y0 as f32,
                width: local_bounds.width() as f32,
                height: local_bounds.height() as f32,
            },
            metadata: RenderedTextMetadata {
                rendered_bounds,
                layout_bounds: if descriptor.point_text {
                    rendered_bounds
                } else {
                    bounds.into()
                },
                post_script_fonts: fonts
                    .iter()
                    .map(|font| font.post_script_name().to_owned())
                    .collect(),
                font_size: layout.font_size,
                color,
            },
            diagnostics,
        })
    }
}

type ContourRegion = Vec<Vec<Vec<[f64; 2]>>>;

fn text_contours(layout: &LayoutRun<'_>, options: &TextRenderOptions) -> Result<ContourRegion> {
    let mut contours = BezPath::new();
    for line in &layout.lines {
        let (baseline_x, baseline_y) = line.baseline;
        let (mut pen_x, mut pen_y) = (0.0, 0.0);
        for glyph in &line.glyphs {
            let font = glyph.font.skrifa_ref()?;
            if let Some(outline) = font.outline_glyphs().get(GlyphId::new(glyph.glyph_id)) {
                let mut path = BezPath::new();
                outline.draw(
                    DrawSettings::unhinted(Size::new(layout.font_size), glyph.font.location()),
                    &mut ContourPen(&mut path),
                )?;
                if glyph.font.synthetic_bold() {
                    // Match Vello's synthetic weight for fallback faces. The border itself is
                    // always a conventional stroke, never a large glyph expansion.
                    path = expand_path(&path, Diagonal2::new(1.0, 1.0), Join::Round, 4.0, 0.1);
                }
                let skew = glyph
                    .font
                    .synthetic_skew()
                    .map_or(Affine::IDENTITY, |angle| {
                        Affine::skew(-(f64::from(angle).to_radians().tan()), 0.0)
                    });
                path.apply_affine(
                    Affine::translate((
                        f64::from(options.padding + baseline_x + pen_x + glyph.x_offset),
                        f64::from(
                            options.padding + baseline_y + pen_y
                                - glyph.y_offset
                                - options.baseline_shift,
                        ),
                    )) * skew
                        * Affine::scale_non_uniform(1.0, -1.0),
                );
                // Normalize face winding before combining fallback fonts into one clipping path.
                if path.area() < 0.0 {
                    path = path.reverse_subpaths();
                }
                contours.extend(path.elements().iter().copied());
            }
            pen_x += glyph.x_advance;
            pen_y -= glyph.y_advance;
        }
    }
    Ok(union_contours(&contours))
}

/// Stroke the visible boundary of the complete text, including touching glyphs and overlaps
/// within variable/CJK glyphs. Nonzero winding preserves counters and self-intersections.
fn union_contours(path: &BezPath) -> ContourRegion {
    let mut contours: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut contour = Vec::new();
    // The boolean library consumes polygons. At 0.02 layout pixels, the flattening error stays
    // below one tenth of a pixel at 4x scale; the original glyph curves still paint the fill.
    flatten(
        path.elements().iter().copied(),
        0.02,
        |element| match element {
            PathEl::MoveTo(point) => {
                if contour.len() >= 3 {
                    contours.push(std::mem::take(&mut contour));
                }
                contour.clear();
                contour.push([point.x, point.y]);
            }
            PathEl::LineTo(point) => contour.push([point.x, point.y]),
            PathEl::ClosePath => {
                if contour.len() >= 3 {
                    contours.push(std::mem::take(&mut contour));
                }
                contour.clear();
            }
            _ => unreachable!("flatten emits only straight segments"),
        },
    );
    if contour.len() >= 3 {
        contours.push(contour);
    }
    contours.simplify_shape(OverlayFillRule::NonZero)
}

/// Offset the merged glyph region by `distance` (negative shrinks it) with the stroke's joins.
fn offset_region(contours: &ContourRegion, distance: f64, join: LineJoin) -> ContourRegion {
    if distance == 0.0 {
        return contours.clone();
    }
    let join = match join {
        // Bound arc chord error to about 0.01 layout pixels. The offset library also
        // clamps the angular step to its supported range.
        LineJoin::Round => OverlayJoin::Round((8.0 * 0.01 / distance.abs()).sqrt()),
        // i_overlay takes a minimum interior angle, rather than a miter-length ratio.
        LineJoin::Miter => OverlayJoin::Miter(2.0 * (1.0_f64 / 4.0).asin()),
        LineJoin::Bevel => OverlayJoin::Bevel,
    };
    contours.outline(
        &OutlineStyle::new(distance)
            .line_join(join)
            .miter_min_turn(0.0),
    )
}

/// The second outline, from the first outline's outer edge outward by `outer.width_px`.
///
/// With `solid` set, every layer drawn above it is opaque and covers the inner part, so the
/// whole expanded region is returned to be painted underneath. Otherwise only the ring is
/// returned, so translucent text or a translucent first outline doesn't reveal it.
fn outer_border(
    contours: &ContourRegion,
    stroke: StrokeOptions,
    outer: OuterStroke,
    solid: bool,
) -> BezPath {
    let inner = stroke.outward_extent();
    let expanded = offset_region(contours, inner + f64::from(outer.width_px), stroke.join);
    if solid {
        return contours_path(&expanded);
    }
    let edge = offset_region(contours, inner, stroke.join);
    contours_path(&expanded.overlay(&edge, OverlayRule::Difference, OverlayFillRule::NonZero))
}

fn fill_command(region: &BezPath, transform: Affine, color: [u8; 4]) -> PreparedSceneCommand {
    PreparedSceneCommand::FillPath(PreparedPath {
        transform: transform.as_coeffs(),
        elements: region
            .elements()
            .iter()
            .map(|element| match *element {
                PathEl::MoveTo(p) => PathElement::MoveTo([p.x, p.y]),
                PathEl::LineTo(p) => PathElement::LineTo([p.x, p.y]),
                PathEl::QuadTo(..) | PathEl::CurveTo(..) => {
                    unreachable!("polygon offsets contain only straight segments")
                }
                PathEl::ClosePath => PathElement::Close,
            })
            .collect(),
        fill: Some(PreparedFill::Solid {
            color,
            rule: FillRule::NonZero,
        }),
        stroke: None,
    })
}

fn contour_border(contours: &ContourRegion, stroke: StrokeOptions, behind_fill: bool) -> BezPath {
    let width = f64::from(stroke.width_px);
    let offset = |distance: f64| offset_region(contours, distance, stroke.join);
    // Resolve offset self-intersections before rasterization. A single-winding filled region
    // avoids coverage accumulation from overlapping wide strokes on the flattened glyphs.
    let region = match stroke.position {
        StrokePosition::Outside if behind_fill => offset(width),
        StrokePosition::Outside => {
            offset(width).overlay(contours, OverlayRule::Difference, OverlayFillRule::NonZero)
        }
        StrokePosition::Center => offset(width * 0.5).overlay(
            &offset(-width * 0.5),
            OverlayRule::Difference,
            OverlayFillRule::NonZero,
        ),
        StrokePosition::Inside => contours.overlay(
            &offset(-width),
            OverlayRule::Difference,
            OverlayFillRule::NonZero,
        ),
    };
    contours_path(&region)
}

fn contours_path(region: &ContourRegion) -> BezPath {
    let mut result = BezPath::new();
    for shape in region {
        for contour in shape {
            if let Some(first) = contour.first() {
                result.move_to((first[0], first[1]));
                for point in &contour[1..] {
                    result.line_to((point[0], point[1]));
                }
                result.close_path();
            }
        }
    }
    result
}

struct ContourPen<'a>(&'a mut BezPath);

impl OutlinePen for ContourPen<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to((f64::from(x), f64::from(y)));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to((f64::from(x), f64::from(y)));
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.0
            .quad_to((f64::from(cx), f64::from(cy)), (f64::from(x), f64::from(y)));
    }
    fn curve_to(&mut self, c1x: f32, c1y: f32, c2x: f32, c2y: f32, x: f32, y: f32) {
        self.0.curve_to(
            (f64::from(c1x), f64::from(c1y)),
            (f64::from(c2x), f64::from(c2y)),
            (f64::from(x), f64::from(y)),
        );
    }
    fn close(&mut self) {
        self.0.close_path();
    }
}

fn automatic_maximum(
    descriptor: &TextNodeDescriptor,
    bounds: LayoutBox,
    is_bubble_text: bool,
) -> f32 {
    if is_bubble_text {
        if descriptor.writing_mode.is_vertical() {
            bounds.height
        } else {
            bounds.width
        }
    } else {
        24.0
    }
}

fn is_chinese_or_japanese_language(language: &str) -> bool {
    language
        .split(['-', '_'])
        .next()
        .is_some_and(|primary| matches!(primary.to_ascii_lowercase().as_str(), "ja" | "zh"))
}

fn inset(rect: LayoutBox, [top, right, bottom, left]: [f32; 4]) -> LayoutBox {
    // Automatic balloon flow can produce cells only a few pixels wide. A fixed inset would leave
    // no layout area and abort the whole export, so an inset that would consume an axis entirely
    // shrinks to keep half of it. Boxes the inset fits are laid out exactly as before.
    let (left, right) = fit_inset(rect.width, left, right);
    let (top, bottom) = fit_inset(rect.height, top, bottom);
    LayoutBox {
        x: rect.x + left,
        y: rect.y + top,
        width: (rect.width - left - right).max(0.0),
        height: (rect.height - top - bottom).max(0.0),
    }
}

fn fit_inset(extent: f32, start: f32, end: f32) -> (f32, f32) {
    let total = start + end;
    if total < extent || total <= 0.0 {
        return (start, end);
    }
    let scale = extent.max(0.0) * 0.5 / total;
    (start * scale, end * scale)
}

fn placement(rect: LayoutBox, width: f32, height: f32) -> (f32, f32) {
    let x = rect.x + (rect.width - width) * 0.5;
    let remaining = rect.height - height;
    let y = rect.y + remaining * 0.5;
    (x, y)
}

/// Color and outward dilation for one glyph draw pass (border or ordinary fill).
struct GlyphPaint {
    color: [u8; 4],
    dilation_px: f32,
}

fn draw_layout(
    scene: &mut PreparedScene,
    resources: &mut Vec<PreparedResource>,
    layout: &LayoutRun<'_>,
    writing_mode: WritingMode,
    options: &TextRenderOptions,
    transform: Affine,
    paint: GlyphPaint,
) {
    for line in &layout.lines {
        let (baseline_x, baseline_y) = match writing_mode {
            WritingMode::Horizontal | WritingMode::VerticalRl => line.baseline,
        };
        let mut pen_x = 0.0;
        let mut pen_y = 0.0;
        let mut start = 0;

        while start < line.glyphs.len() {
            let font = line.glyphs[start].font;
            let key = font_key(font);
            let mut end = start + 1;
            while end < line.glyphs.len() && font_key(line.glyphs[end].font) == key {
                end += 1;
            }

            let mut glyphs = Vec::with_capacity(end - start);
            for glyph in &line.glyphs[start..end] {
                glyphs.push(PreparedGlyph {
                    id: glyph.glyph_id,
                    x: options.padding + baseline_x + pen_x + glyph.x_offset,
                    y: options.padding + baseline_y + pen_y
                        - glyph.y_offset
                        - options.baseline_shift,
                });
                pen_x += glyph.x_advance;
                pen_y -= glyph.y_advance;
            }

            let font_id = ResourceId::for_font(font.bytes());
            if !resources.iter().any(|candidate| candidate.id() == font_id) {
                resources.push(PreparedResource::font_shared(font.shared_bytes()));
            }
            let glyph_transform = font
                .synthetic_skew()
                .map(|angle| Affine::skew(-(angle.to_radians().tan() as f64), 0.0).as_coeffs());
            let synthetic_bold = if font.synthetic_bold() { 1.0 } else { 0.0 };
            scene
                .commands
                .push(PreparedSceneCommand::GlyphRun(PreparedGlyphRun {
                    font: font_id,
                    font_index: font.index(),
                    font_size: layout.font_size,
                    normalized_coords: font.normalized_coords().to_vec(),
                    transform: transform.as_coeffs(),
                    glyph_transform,
                    // Hinting folds a uniform zoom into `font_size` for crisp fill text, but a
                    // border needs the real transform so its dilation amount stays proportional
                    // to that same zoom instead of being rendered at a fixed pixel size.
                    hint: options.hint_glyphs && paint.dilation_px == 0.0,
                    embolden: [synthetic_bold + paint.dilation_px; 2],
                    color: paint.color,
                    glyphs,
                }));
            start = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use koharu_rasterizer::{
        Bounds, CompositionCommand, LayerId, LayerKind as PreparedLayerKind, Point,
        PreparedContent, PreparedFrame, PreparedFrameBundle, PreparedFrameManifest, PreparedLayer,
        PreparedResourcePacket, PreparedResourceStore, Presentation, Revision,
    };
    use koharu_scene::EntityId;

    use super::*;
    use crate::fonts::{FontRequest, FontSystem, Fonts};

    #[test]
    fn inset_keeps_layout_area_in_cells_narrower_than_the_inset() {
        let cell = LayoutBox {
            x: 10.0,
            y: 20.0,
            width: 7.0,
            height: 43.0,
        };
        let bounds = inset(cell, [4.0; 4]);
        assert!(bounds.width > 0.0 && bounds.height > 0.0);
        assert_eq!(bounds.width, 3.5);
        assert_eq!(bounds.x, 11.75);
        // An axis the inset fits keeps its exact inset.
        assert_eq!((bounds.y, bounds.height), (24.0, 35.0));

        // An inset that exactly consumes an axis would leave zero width, so it shrinks too.
        let exact = inset(
            LayoutBox {
                x: 0.0,
                y: 0.0,
                width: 8.0,
                height: 30.0,
            },
            [4.0; 4],
        );
        assert_eq!((exact.x, exact.width), (2.0, 4.0));

        // The same applies to height, independently of width.
        let short = inset(
            LayoutBox {
                x: 0.0,
                y: 0.0,
                width: 30.0,
                height: 5.0,
            },
            [4.0; 4],
        );
        assert_eq!((short.x, short.width), (4.0, 22.0));
        assert_eq!((short.y, short.height), (1.25, 2.5));

        // A genuinely empty cell stays empty, so the renderer still rejects it.
        let empty = inset(
            LayoutBox {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 30.0,
            },
            [4.0; 4],
        );
        assert_eq!(empty.width, 0.0);
    }

    /// The cell from #1127: automatic balloon flow produced a 7 x 43 px cell, the 4 px inset
    /// left no layout width, and rendering (and with it the whole export) failed.
    #[tokio::test]
    async fn balloon_cell_narrower_than_the_inset_still_renders() {
        let family = FontSystem::new()
            .first_font()
            .unwrap()
            .family_name()
            .to_owned();
        let descriptor = TextNodeDescriptor {
            entity: EntityId::new(),
            text: "Test".to_owned(),
            language: None,
            width: 7.0,
            height: 43.0,
            balloon_contour: Some(vec![(0.0, 0.0), (7.0, 0.0), (7.0, 43.0), (0.0, 43.0)]),
            preferred_font: Some(family.clone()),
            font_families: vec![family],
            font_weight: None,
            font_style: None,
            font_size: None,
            minimum_font_size: 9.0,
            auto_fit: true,
            alignment: TextAlign::Center,
            writing_mode: WritingMode::Horizontal,
            foreground_color: [0, 0, 0, 255],
            stroke: None,
            line_height: 1.2,
            letter_spacing: 0.0,
            word_spacing: 0.0,
            text_inset: [4.0; 4],
            point_text: false,
        };
        let fonts = Fonts::new();
        fonts
            .prepare(&[FontRequest {
                family: descriptor.font_families[0].clone(),
                weight: None,
                style: None,
            }])
            .await
            .unwrap();
        let rendered = TextRenderer::new().render_descriptor(&descriptor, &fonts);
        assert!(rendered.is_ok(), "{:?}", rendered.err());
    }

    #[test]
    fn contour_union_removes_shared_edges_and_preserves_counters() {
        use vello::kurbo::{ParamCurveNearest, Point, Rect, Shape};
        let mut raw = Rect::new(0.0, 0.0, 20.0, 30.0).to_path(0.01);
        raw.extend(
            Rect::new(10.0, 0.0, 30.0, 30.0)
                .to_path(0.01)
                .elements()
                .iter()
                .copied(),
        );
        let hole = Rect::new(12.0, 10.0, 18.0, 20.0)
            .to_path(0.01)
            .reverse_subpaths();
        // Twice the opposite winding cuts the counter through both overlapping glyph shapes.
        raw.extend(hole.elements().iter().copied());
        raw.extend(hole.elements().iter().copied());
        let united = contours_path(&union_contours(&raw));
        assert_eq!(
            united.winding(Point::new(15.0, 15.0)),
            0,
            "counter stays open"
        );
        assert_ne!(
            united.winding(Point::new(15.0, 5.0)),
            0,
            "glyph overlap stays filled"
        );
        assert!((united.area().abs() - 840.0).abs() < 0.001);
        // The old path had interior vertical edges at x=10 and x=20 even above the counter.
        for segment in united.segments() {
            assert!(segment.nearest(Point::new(10.0, 5.0), 0.001).distance_sq > 1.0);
            assert!(segment.nearest(Point::new(20.0, 5.0), 0.001).distance_sq > 1.0);
        }
    }

    #[test]
    fn contour_outside_covers_antialiased_seams_and_keeps_transparent_fill() {
        use koharu_rasterizer::{RasterOptions, Rasterizer};
        let font = FontSystem::new().first_font().unwrap();
        let layout = TextLayout::new(&font)
            .with_font_size(40.0)
            .run("Ho")
            .unwrap();
        let bounds = contours_path(&text_contours(&layout, &TextRenderOptions::default()).unwrap())
            .bounding_box();
        let transform = Affine::translate((8.37 - bounds.x0, 5.23 - bounds.y0));
        let rasterizer = Rasterizer::new().unwrap();
        let render = |alpha: u8, outlined: bool| {
            let mut scene = PreparedScene::default();
            let mut resources = Vec::new();
            TextRenderer::new()
                .render(
                    &mut scene,
                    &mut resources,
                    &layout,
                    WritingMode::Horizontal,
                    &TextRenderOptions {
                        color: [255, 255, 255, alpha],
                        hint_glyphs: false,
                        stroke: outlined.then_some(StrokeOptions {
                            color: [255, 255, 255, 255],
                            width_px: 4.0,
                            method: StrokeMethod::Contour,
                            position: StrokePosition::Outside,
                            join: LineJoin::Round,
                            outer: None,
                        }),
                        ..TextRenderOptions::default()
                    },
                    transform,
                )
                .unwrap();
            let frame = text_bundle(scene, resources, 160, 48)
                .into_frame_with_raster_sources(&Default::default())
                .unwrap();
            rasterizer
                .rasterize(&frame, RasterOptions::default())
                .unwrap()
                .image
        };
        let fill = render(255, false);
        let opaque = render(255, true);
        let clear = render(0, true);
        let translucent = render(96, true);
        let mut edges = 0;
        let mut interiors = 0;
        for (x, y, pixel) in fill.enumerate_pixels() {
            if (32..=223).contains(&pixel.0[3]) {
                edges += 1;
                assert!(
                    opaque.get_pixel(x, y).0[3] >= 254,
                    "no seam at glyph edge ({x},{y})"
                );
            }
            if pixel.0[3] == 255
                && x > 0
                && y > 0
                && x + 1 < fill.width()
                && y + 1 < fill.height()
                && [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                    .iter()
                    .all(|&(x, y)| fill.get_pixel(x, y).0[3] == 255)
            {
                interiors += 1;
                assert!(
                    clear.get_pixel(x, y).0[3] <= 1,
                    "outside-only stroke preserves clear interior ({x},{y})"
                );
                assert!(
                    translucent.get_pixel(x, y).0[3].abs_diff(96) <= 1,
                    "outside-only stroke preserves translucent interior ({x},{y})"
                );
            }
        }
        assert!(
            edges > 10 && interiors > 10,
            "exercise both antialiased edges and solid glyph interiors: edges={edges}, interiors={interiors}"
        );
    }

    #[test]
    fn second_outline_starts_at_the_first_outlines_outer_edge() {
        use vello::kurbo::Point;
        let mut raw = Rect::new(20.0, 20.0, 80.0, 80.0).to_path(0.01);
        raw.extend(
            Rect::new(40.0, 40.0, 60.0, 60.0)
                .to_path(0.01)
                .reverse_subpaths()
                .elements()
                .iter()
                .copied(),
        );
        let contours = union_contours(&raw);
        let outer = OuterStroke {
            color: [0, 0, 0, 255],
            width_px: 4.0,
        };
        // (position, the first outline's outward reach in px) for a 6 px first outline.
        for (position, reach) in [
            (StrokePosition::Outside, 6.0),
            (StrokePosition::Center, 3.0),
            (StrokePosition::Inside, 0.0),
        ] {
            let stroke = StrokeOptions {
                color: [255; 4],
                width_px: 6.0,
                method: StrokeMethod::Contour,
                position,
                join: LineJoin::Round,
                outer: Some(outer),
            };
            let ring = outer_border(&contours, stroke, outer, false);
            let left = 20.0 - reach;
            // Inside the second outline, on the outer and the hole side of the shape.
            assert_ne!(
                ring.winding(Point::new(left - 2.0, 50.0)),
                0,
                "{position:?}"
            );
            assert_ne!(
                ring.winding(Point::new(40.0 + reach + 2.0, 50.0)),
                0,
                "{position:?}"
            );
            // Beyond it, and inside the first outline: not part of the ring.
            assert_eq!(
                ring.winding(Point::new(left - 4.5, 50.0)),
                0,
                "{position:?}"
            );
            assert_eq!(
                ring.winding(Point::new(left + 0.5, 50.0)),
                0,
                "{position:?}"
            );
            assert_eq!(ring.winding(Point::new(30.0, 50.0)), 0, "{position:?}");
            // The solid variant also covers everything inside, for painting underneath.
            let solid = outer_border(&contours, stroke, outer, true);
            assert_ne!(
                solid.winding(Point::new(left - 2.0, 50.0)),
                0,
                "{position:?}"
            );
            assert_ne!(solid.winding(Point::new(30.0, 50.0)), 0, "{position:?}");
            assert!(
                (solid.bounding_box().x0 - (left - 4.0)).abs() < 0.02,
                "{position:?}"
            );
        }
    }

    #[test]
    fn second_outline_is_drawn_beneath_the_first_and_widens_bounds() {
        let font = FontSystem::new().first_font().unwrap();
        let layout = TextLayout::new(&font)
            .with_font_size(40.0)
            .run("Ho")
            .unwrap();
        for method in [StrokeMethod::Contour, StrokeMethod::Expansion] {
            let single = StrokeOptions {
                color: [255; 4],
                width_px: 3.0,
                method,
                position: StrokePosition::Outside,
                join: LineJoin::Round,
                outer: None,
            };
            let double = StrokeOptions {
                outer: Some(OuterStroke {
                    color: [255, 0, 0, 255],
                    width_px: 5.0,
                }),
                ..single
            };
            let render = |stroke: StrokeOptions| {
                let mut scene = PreparedScene::default();
                let mut resources = Vec::new();
                let bounds = TextRenderer::new()
                    .render(
                        &mut scene,
                        &mut resources,
                        &layout,
                        WritingMode::Horizontal,
                        &TextRenderOptions {
                            stroke: Some(stroke),
                            ..TextRenderOptions::default()
                        },
                        Affine::IDENTITY,
                    )
                    .unwrap();
                (scene, bounds)
            };
            let (single_scene, single_bounds) = render(single);
            let (double_scene, double_bounds) = render(double);
            assert!(double_scene.commands.len() > single_scene.commands.len());
            // The outer color is painted before anything else, so the rest covers its inside.
            let first_color = match &double_scene.commands[0] {
                PreparedSceneCommand::FillPath(PreparedPath {
                    fill: Some(PreparedFill::Solid { color, .. }),
                    ..
                }) => *color,
                PreparedSceneCommand::GlyphRun(run) => run.color,
                other => panic!("text renders no other command: {other:?}"),
            };
            assert_eq!(first_color, [255, 0, 0, 255], "{method:?}");
            if method == StrokeMethod::Contour {
                let (single, double) = (single_bounds.unwrap(), double_bounds.unwrap());
                assert!(
                    (single.x0 - double.x0 - 5.0).abs() < 0.05,
                    "{single:?} {double:?}"
                );
            }
        }
    }

    #[test]
    fn contour_regions_preserve_positions_counters_and_corner_styles() {
        use vello::kurbo::Point;
        let mut raw = Rect::new(20.0, 20.0, 80.0, 80.0).to_path(0.01);
        raw.extend(
            Rect::new(40.0, 40.0, 60.0, 60.0)
                .to_path(0.01)
                .reverse_subpaths()
                .elements()
                .iter()
                .copied(),
        );
        let contours = union_contours(&raw);
        for join in [LineJoin::Round, LineJoin::Miter, LineJoin::Bevel] {
            let stroke = StrokeOptions {
                color: [255; 4],
                width_px: 6.0,
                method: StrokeMethod::Contour,
                position: StrokePosition::Outside,
                join,
                outer: None,
            };
            for (position, outward, inward) in [
                (StrokePosition::Outside, true, false),
                (StrokePosition::Center, true, true),
                (StrokePosition::Inside, false, true),
            ] {
                let border = contour_border(&contours, StrokeOptions { position, ..stroke }, false);
                assert_eq!(border.winding(Point::new(18.0, 50.0)) != 0, outward);
                assert_eq!(border.winding(Point::new(22.0, 50.0)) != 0, inward);
                assert_eq!(border.winding(Point::new(42.0, 50.0)) != 0, outward);
                assert_eq!(border.winding(Point::new(38.0, 50.0)) != 0, inward);
                assert_eq!(
                    border.winding(Point::new(50.0, 50.0)),
                    0,
                    "counter stays open"
                );
                // Overlapping offset pieces must resolve to one winding before area AA.
                for y in 10..90 {
                    for x in 10..90 {
                        assert!(
                            border
                                .winding(Point::new(x as f64 + 0.37, y as f64 + 0.23))
                                .abs()
                                <= 1
                        );
                    }
                }
            }
            let border = contour_border(&contours, stroke, true);
            assert_ne!(
                border.winding(Point::new(30.0, 50.0)),
                0,
                "opaque fill has underlap"
            );
            assert_eq!(
                border.winding(Point::new(15.0, 15.0)) != 0,
                join == LineJoin::Miter
            );
            assert_eq!(
                border.winding(Point::new(16.5, 16.5)) != 0,
                join != LineJoin::Bevel
            );
            assert!((border.bounding_box().x0 - 14.0).abs() < 0.02);
        }
        let collapsed = contour_border(
            &contours,
            StrokeOptions {
                color: [255; 4],
                width_px: 40.0,
                method: StrokeMethod::Contour,
                position: StrokePosition::Inside,
                join: LineJoin::Round,
                outer: None,
            },
            false,
        );
        assert!((collapsed.area().abs() - contours_path(&contours).area().abs()).abs() < 0.01);
    }

    #[test]
    fn contour_outside_outer_edge_keeps_antialiasing_at_large_sizes() {
        use koharu_rasterizer::{RasterOptions, Rasterizer};
        let font = FontSystem::new().first_font().unwrap();
        let rasterizer = Rasterizer::new().unwrap();
        for size in [64.0, 300.0] {
            let layout = TextLayout::new(&font)
                .with_font_size(size)
                .run("O")
                .unwrap();
            let stroke_width = size * 24.5 / 300.0;
            let options = TextRenderOptions {
                stroke: Some(StrokeOptions {
                    color: [255; 4],
                    width_px: stroke_width,
                    method: StrokeMethod::Contour,
                    position: StrokePosition::Outside,
                    join: LineJoin::Round,
                    outer: None,
                }),
                ..TextRenderOptions::default()
            };
            let bounds = contours_path(&text_contours(&layout, &options).unwrap()).bounding_box();
            let margin = f64::from(stroke_width) + 4.0;
            let width = (bounds.width() + 2.0 * margin).ceil() as u32;
            let height = (bounds.height() + 2.0 * margin).ceil() as u32;
            let mut scene = PreparedScene::default();
            let mut resources = Vec::new();
            TextRenderer::new()
                .render(
                    &mut scene,
                    &mut resources,
                    &layout,
                    WritingMode::Horizontal,
                    &options,
                    Affine::translate((margin + 0.37 - bounds.x0, margin + 0.23 - bounds.y0)),
                )
                .unwrap();
            let frame = text_bundle(scene, resources, width, height)
                .into_frame_with_raster_sources(&Default::default())
                .unwrap();
            let image = rasterizer
                .rasterize(&frame, RasterOptions::default())
                .unwrap()
                .image;
            let mut edge_pixels = 0;
            let mut partial_pixels = 0;
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let alpha = image.get_pixel(x, y).0[3];
                    if alpha > 0
                        && [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                            .iter()
                            .any(|&(x, y)| image.get_pixel(x, y).0[3] == 0)
                    {
                        edge_pixels += 1;
                        partial_pixels += usize::from(alpha < 255);
                    }
                }
            }
            assert!(edge_pixels > 30);
            assert!(
                partial_pixels * 4 > edge_pixels,
                "curved outer border must retain partial coverage at size {size}: {partial_pixels}/{edge_pixels}"
            );
        }
    }

    fn text_bundle(
        scene: PreparedScene,
        resources: Vec<PreparedResource>,
        width: u32,
        height: u32,
    ) -> PreparedFrameBundle {
        PreparedFrameBundle {
            frame: PreparedFrame {
                revision: Revision::new(7),
                page: LayerId::from_bytes([1; 16]),
                width,
                height,
                origin: (0, 0),
                normalization: Affine::IDENTITY.as_coeffs(),
                layers: vec![PreparedLayer {
                    id: LayerId::from_bytes([2; 16]),
                    geometry: vec![
                        Point { x: 0.0, y: 0.0 },
                        Point {
                            x: width as f64,
                            y: 0.0,
                        },
                        Point {
                            x: width as f64,
                            y: height as f64,
                        },
                        Point {
                            x: 0.0,
                            y: height as f64,
                        },
                    ],
                    bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: width as f32,
                        height: height as f32,
                    },
                    local_bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: width as f32,
                        height: height as f32,
                    },
                    presentation: Presentation {
                        visible: true,
                        opacity: 1.0,
                    },
                    kind: PreparedLayerKind::Text,
                    placement: Affine::IDENTITY.as_coeffs(),
                    content: PreparedContent::Vector(scene),
                    element_frame: None,
                }],
            },
            resources,
        }
    }

    #[test]
    fn automatic_size_preserves_free_text_default_and_balloon_extent() {
        let descriptor = TextNodeDescriptor {
            entity: EntityId::new(),
            text: "Hi".to_owned(),
            language: None,
            width: 240.0,
            height: 120.0,
            balloon_contour: None,
            preferred_font: Some("Arial".to_owned()),
            font_families: vec!["Arial".to_owned()],
            font_weight: None,
            font_style: None,
            font_size: Some(6.0),
            minimum_font_size: 9.0,
            auto_fit: true,
            alignment: TextAlign::Center,
            writing_mode: WritingMode::Horizontal,
            foreground_color: [0, 0, 0, 255],
            stroke: None,
            line_height: 1.2,
            letter_spacing: 0.0,
            word_spacing: 0.0,
            text_inset: [0.0; 4],
            point_text: false,
        };
        let bounds = LayoutBox {
            x: 0.0,
            y: 0.0,
            width: 240.0,
            height: 120.0,
        };

        assert_eq!(automatic_maximum(&descriptor, bounds, false), 24.0);
        assert_eq!(automatic_maximum(&descriptor, bounds, true), 240.0);
    }

    #[test]
    fn rendered_text_survives_prepared_packet_round_trip() {
        let font = FontSystem::new().first_font().unwrap();
        let layout = TextLayout::new(&font)
            .with_font_size(24.0)
            .run("Koharu")
            .unwrap();
        let mut scene = PreparedScene::default();
        let mut resources = Vec::new();
        TextRenderer::new()
            .render(
                &mut scene,
                &mut resources,
                &layout,
                WritingMode::Horizontal,
                &TextRenderOptions {
                    stroke: Some(StrokeOptions {
                        color: [255, 255, 255, 255],
                        width_px: 6.0,
                        method: StrokeMethod::Contour,
                        position: StrokePosition::Outside,
                        join: LineJoin::Round,
                        outer: None,
                    }),
                    ..TextRenderOptions::default()
                },
                Affine::IDENTITY,
            )
            .unwrap();
        assert!(
            scene
                .commands
                .iter()
                .any(|command| matches!(command, PreparedSceneCommand::FillPath(_)))
        );
        assert_eq!(resources.len(), 1);
        assert!(matches!(
            &resources[0],
            PreparedResource::Font { bytes, .. } if !bytes.is_empty()
        ));

        let layer = LayerId::from_bytes([2; 16]);
        let bundle = text_bundle(scene, resources, 160, 48);

        let encoded = bundle.manifest().unwrap().encode().unwrap();
        let manifest = PreparedFrameManifest::decode(&encoded).unwrap();
        let mut resources = PreparedResourceStore::default();
        for reference in manifest.required_resources() {
            let encoded = bundle
                .resource_packet(reference.id)
                .unwrap()
                .encode()
                .unwrap();
            resources.insert(PreparedResourcePacket::decode(&encoded).unwrap());
        }
        let compiled = manifest.compile(&resources).unwrap();
        assert_eq!(compiled.revision(), Revision::new(7));
        assert_eq!(compiled.layers()[0].id(), layer);
        assert!(matches!(
            compiled.composition_commands(1).as_slice(),
            [CompositionCommand::Vector(_)]
        ));
    }
}
