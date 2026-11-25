//! The brush tools, draw, eraser, and remove: a stroke's outline, which the
//! preview and the coverage both fill, and paint and erasure on a raster's
//! straight-alpha pixels.

use image::{RgbaImage, imageops};
use koharu_scene::{Content, Edit, LayerId, Op, PixelFormat, Raster, Ref, Rgba};
use kurbo::{
    Affine, BezPath, Cap, Circle, Join, Point, Shape as _, Size, Stroke as Style, StrokeOpts,
    stroke,
};
use vello_cpu::{Pixmap, RenderContext, Resources, color::palette::css::WHITE};

use super::added;
use crate::{
    Effect, Notice, RampAlong, Tool,
    draw::overlay::Overlay,
    gesture::{Context, Gesture, GestureLook, Press},
    view::Sample,
};

/// How closely a stroke's outline follows its round caps and joins, in page
/// pixels.
const TOLERANCE: f64 = 0.1;

/// A brush tool's press starts a stroke: the draw tool's and the remove
/// tool's onto the one selected raster or a new one, the eraser's onto the one
/// selected.
pub(crate) fn press(cx: &Context, sample: Sample) -> Press {
    if sample.modifiers.ctrl() && sample.modifiers.alt() {
        return Press::starts(Gesture::ResizeBrush(Resize {
            origin: sample.screen,
            start: cx.brush.diameter,
            diameter: cx.brush.diameter,
        }));
    }
    let target = cx.selection.single().filter(|&layer| {
        // A mask takes no colours, which the remove tool's result is.
        cx.snapshot
            .writable(layer)
            .is_some_and(|(_, format)| cx.tool != Tool::Remove || format == PixelFormat::Rgba8)
    });
    if cx.tool == Tool::Eraser && target.is_none() {
        return Press::answers(vec![Effect::Notice(Notice::EraserNeedsRaster)]);
    }
    let points = vec![sample.page];
    Press::starts(Gesture::Stroke(Stroke {
        tool: cx.tool,
        outline: outline(&points, cx.brush.diameter),
        points,
        target,
    }))
}

/// A diameter preview anchored to the press. Only release changes the brush.
pub(crate) struct Resize {
    origin: Point,
    start: f64,
    diameter: f64,
}

impl Resize {
    pub fn moved(&mut self, cx: &Context, samples: &[Sample]) {
        if let Some(last) = samples.last() {
            let travel = (last.screen.x - self.origin.x) / cx.view.camera().scale;
            let size = cx.snapshot.size();
            self.diameter = (self.start + 2.0 * travel)
                .round()
                .clamp(1.0, size.width.max(size.height).max(1.0));
        }
    }

    pub fn finish(self) -> Vec<Effect> {
        (self.diameter != self.start)
            .then_some(Effect::BrushDiameter(self.diameter))
            .into_iter()
            .collect()
    }

    pub fn preview(&self, overlay: &mut Overlay) {
        overlay.ring(self.origin, self.diameter);
    }

    pub fn look(&self) -> GestureLook {
        GestureLook::BrushRing(self.origin, self.diameter)
    }
}

/// A brush stroke through page points, onto `target` or, for the draw tool
/// and the remove tool with none, a new raster.
pub(crate) struct Stroke {
    tool: Tool,
    points: Vec<Point>,
    target: Option<LayerId>,
    /// The points' outline, expanded again only when a point arrives.
    outline: BezPath,
}

impl Stroke {
    pub fn moved(&mut self, cx: &Context, samples: &[Sample]) -> Vec<Effect> {
        let before = self.points.len();
        for sample in samples {
            if self.points.last() != Some(&sample.page) {
                self.points.push(sample.page);
            }
        }
        if self.points.len() != before {
            self.outline = outline(&self.points, cx.brush.diameter);
        }
        Vec::new()
    }

    /// The remove tool's inpaint request, with the raster it adds when it has
    /// none, or the draw tool's or the eraser's edit.
    pub fn finish(self, cx: &Context) -> anyhow::Result<Vec<Effect>> {
        let page = cx.snapshot.page().id;
        let Some((rect, coverage)) = coverage(&self.outline, cx.snapshot.size()) else {
            return Ok(Vec::new());
        };
        if self.tool == Tool::Remove {
            let mut effects = Vec::new();
            let layer = self.target.unwrap_or_else(|| {
                // Added at once, so a stroke made while this one is redrawn
                // goes onto it too.
                let raster = Content::Raster(Raster::default());
                let (id, add) = added(cx.snapshot.page(), None, raster);
                let edit = Edit {
                    label: "add_layer".to_owned(),
                    ops: vec![add],
                };
                effects = vec![Effect::Edit(edit), Effect::Selection(vec![id])];
                id
            });
            effects.push(Effect::Inpaint {
                page,
                layer,
                rect,
                mask: coverage,
            });
            return Ok(effects);
        }
        self.stroked(cx, rect, &coverage)
    }

    pub fn preview(&self, cx: &Context, overlay: &mut Overlay) {
        // A mask takes coverage, not colour.
        let colored = self
            .target
            .and_then(|layer| cx.snapshot.writable(layer))
            .is_none_or(|(_, format)| format == PixelFormat::Rgba8);
        let tint = match self.tool {
            Tool::Draw if colored => {
                // A ramp previews as a straight one between its ends.
                if let Some((from, to)) = self.ramp_ends(cx) {
                    let brush = cx.brush;
                    overlay.fill_ramp(
                        &self.outline,
                        (from, to),
                        (brush.color, brush.color_at(1.0)),
                    );
                    return;
                }
                cx.brush.color
            }
            _ => {
                let Rgba([r, g, b, _]) = cx.colors.selection;
                Rgba([r, g, b, 0.4])
            }
        };
        overlay.fill(&self.outline, tint);
    }

    /// Where the brush's ramp starts and ends on the page, `None` without one.
    fn ramp_ends(&self, cx: &Context) -> Option<(Point, Point)> {
        let size = cx.snapshot.size();
        Some(match cx.brush.ramp?.along {
            RampAlong::Stroke => (*self.points.first()?, *self.points.last()?),
            RampAlong::Horizontal => (Point::ZERO, Point::new(size.width, 0.0)),
            RampAlong::Vertical => (Point::ZERO, Point::new(0.0, size.height)),
        })
    }

    /// The brush's colour over the page pixels `rect`, as opaque as
    /// `coverage` says of it: one colour, or each pixel's along the ramp,
    /// whose stroke position is that of the nearest point of the stroke's
    /// line, measured along it.
    fn paint(&self, cx: &Context, rect: koharu_scene::Rect, coverage: &[u8]) -> RgbaImage {
        let brush = cx.brush;
        let size = cx.snapshot.size();
        let lengths: Vec<f64> = self
            .points
            .windows(2)
            .map(|pair| pair[0].distance(pair[1]))
            .collect();
        let total: f64 = lengths.iter().sum();
        let along_stroke = |at: Point| {
            let (mut best, mut walked, mut position) = (f64::INFINITY, 0.0, 0.0);
            for (pair, length) in self.points.windows(2).zip(&lengths) {
                let t = match *length > 0.0 {
                    true => {
                        ((at - pair[0]).dot(pair[1] - pair[0]) / (length * length)).clamp(0.0, 1.0)
                    }
                    false => 0.0,
                };
                let distance = pair[0].lerp(pair[1], t).distance(at);
                if distance < best {
                    (best, position) = (distance, walked + t * length);
                }
                walked += length;
            }
            if total > 0.0 { position / total } else { 0.0 }
        };
        let pixels = coverage
            .iter()
            .enumerate()
            .flat_map(|(index, &covered)| {
                let color = match brush.ramp {
                    None => brush.color,
                    Some(ramp) => {
                        let index = index as u32;
                        let at = Point::new(
                            f64::from(rect.x) + f64::from(index % rect.width) + 0.5,
                            f64::from(rect.y) + f64::from(index / rect.width) + 0.5,
                        );
                        brush.color_at(match ramp.along {
                            _ if covered == 0 => 0.0,
                            RampAlong::Stroke => along_stroke(at),
                            RampAlong::Horizontal => at.x / size.width,
                            RampAlong::Vertical => at.y / size.height,
                        })
                    }
                };
                let Rgba([r, g, b, a]) = color;
                let [r, g, b] = [r, g, b].map(|channel| (channel * 255.0).round() as u8);
                [r, g, b, (f32::from(covered) * a).round() as u8]
            })
            .collect();
        RgbaImage::from_raw(rect.width, rect.height, pixels).expect("coverage covers `rect`")
    }

    pub fn look(&self) -> GestureLook {
        GestureLook::Stroke {
            points: self.points.len(),
            last: self.points.last().copied().unwrap_or_default(),
        }
    }

    /// A paint or erase stroke's coverage over the page pixels `rect` as one
    /// write to its raster, clean-up, or mask, read and composited in its own
    /// coordinates; painting with no target adds a raster at the top of the
    /// page, selected. A mask takes coverage on an alpha scale, as Clip Studio
    /// Paint's does: a stroke of any colour uncovers the clean-up, and the
    /// eraser hides it.
    fn stroked(
        &self,
        cx: &Context,
        rect: koharu_scene::Rect,
        coverage: &[u8],
    ) -> anyhow::Result<Vec<Effect>> {
        let label = match self.tool {
            Tool::Draw => "paint",
            _ => "erase",
        };
        let target = self
            .target
            .and_then(|layer| Some((layer, cx.snapshot.writable(layer)?)));
        let (ops, created) = match target {
            Some((layer, (offset, format))) => {
                let own = koharu_scene::Rect {
                    x: rect.x - offset.x,
                    y: rect.y - offset.y,
                    ..rect
                };
                let mut pixels = cx.session.lock().pixels(layer, own)?;
                match (self.tool, format) {
                    (Tool::Draw, PixelFormat::Rgba8) => {
                        let mut under = RgbaImage::from_raw(own.width, own.height, pixels)
                            .ok_or_else(|| anyhow::anyhow!("a raster's pixels are RGBA"))?;
                        imageops::overlay(&mut under, &self.paint(cx, rect, coverage), 0, 0);
                        pixels = under.into_raw();
                    }
                    (Tool::Draw, PixelFormat::Gray8) => {
                        uncovered(&mut pixels, coverage, cx.brush.color.0[3]);
                    }
                    (_, PixelFormat::Rgba8) => erased(&mut pixels, coverage, 4),
                    (_, PixelFormat::Gray8) => erased(&mut pixels, coverage, 1),
                }
                let write = Op::WritePixels {
                    layer: Ref::from(layer),
                    rect: own,
                    pixels,
                };
                (vec![write], None)
            }
            None if self.tool == Tool::Draw => {
                let pixels = self.paint(cx, rect, coverage).into_raw();
                let raster = Content::Raster(Raster::default());
                let (id, add) = added(cx.snapshot.page(), None, raster);
                let write = Op::WritePixels {
                    layer: Ref::from(id),
                    rect,
                    pixels,
                };
                (vec![add, write], Some(id))
            }
            None => return Ok(Vec::new()),
        };
        let edit = Effect::Edit(Edit {
            label: label.to_owned(),
            ops,
        });
        Ok(std::iter::once(edit)
            .chain(created.map(|id| Effect::Selection(vec![id])))
            .collect())
    }
}

/// A round stroke through `points`, `diameter` wide, as the outline it
/// fills; a single point is a dot.
fn outline(points: &[Point], diameter: f64) -> BezPath {
    match points {
        [] => BezPath::new(),
        [point] => Circle::new(*point, diameter * 0.5).to_path(TOLERANCE),
        [first, rest @ ..] => {
            let mut line = BezPath::new();
            line.move_to(*first);
            for point in rest {
                line.line_to(*point);
            }
            let style = Style::new(diameter)
                .with_caps(Cap::Round)
                .with_join(Join::Round);
            stroke(line.iter(), &style, &StrokeOpts::default(), TOLERANCE)
        }
    }
}

/// `outline`'s antialiased 8-bit coverage over the page pixels it touches.
pub(crate) fn coverage(outline: &BezPath, page: Size) -> Option<(koharu_scene::Rect, Vec<u8>)> {
    let rect = koharu_scene::Rect::covering(
        outline.bounding_box(),
        page.width as u32,
        page.height as u32,
    )?;
    let (width, height) = (
        u16::try_from(rect.width).ok()?,
        u16::try_from(rect.height).ok()?,
    );
    let mut context = RenderContext::new(width, height);
    context.set_transform(Affine::translate((-f64::from(rect.x), -f64::from(rect.y))));
    context.set_paint(WHITE);
    context.fill_path(outline);
    context.flush();
    let mut pixmap = Pixmap::new(width, height);
    context.render(&mut pixmap, &mut Resources::new());
    Some((rect, pixmap.data().iter().map(|pixel| pixel.a).collect()))
}

/// `pixels`, `bytes` to a pixel, with their last channel, a raster's alpha
/// or a mask's grey, taken away where `coverage` says.
fn erased(pixels: &mut [u8], coverage: &[u8], bytes: usize) {
    for (pixel, &covered) in pixels.chunks_exact_mut(bytes).zip(coverage) {
        let kept = 1.0 - f32::from(covered) / 255.0;
        pixel[bytes - 1] = (f32::from(pixel[bytes - 1]) * kept).round() as u8;
    }
}

/// A mask's grey `pixels` raised toward white where `coverage` says, as
/// opaque as the brush's `alpha`.
fn uncovered(pixels: &mut [u8], coverage: &[u8], alpha: f32) {
    for (pixel, &covered) in pixels.iter_mut().zip(coverage) {
        let added = f32::from(covered) / 255.0 * alpha;
        let grey = f32::from(*pixel);
        *pixel = (grey + (255.0 - grey) * added).round() as u8;
    }
}
