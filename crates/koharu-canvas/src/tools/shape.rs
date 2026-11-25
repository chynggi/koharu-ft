//! The rectangle and ellipse tools: a shape over a drag, filled with the
//! brush colour, as Photoshop's shape tools fill with the foreground colour.
//! Shift keeps the sides equal and Alt draws from the centre, as there.

use koharu_scene::{Content, Edit, FillRule, Path, Shape, ShapeFill};
use kurbo::{BezPath, Ellipse, Point, Rect, Shape as _, Vec2};

use super::added;
use crate::{Effect, Tool, draw::overlay::Overlay, gesture::Context, view::Sample};

/// A drag shorter than this on either side, in device-independent pixels,
/// is a click, which draws nothing.
const CLICK: f64 = 4.0;

/// Flattening tolerance of an ellipse's path, in page pixels.
const TOLERANCE: f64 = 0.1;

/// A shape being dragged out, in page points.
pub(crate) struct ShapeDraft {
    ellipse: bool,
    start: Point,
    current: Sample,
}

impl ShapeDraft {
    pub fn new(tool: Tool, sample: Sample) -> Self {
        Self {
            ellipse: tool == Tool::Ellipse,
            start: sample.page,
            current: sample,
        }
    }

    /// The box the shape fills: from the press to the pointer, squared with
    /// Shift and about the press with Alt.
    pub fn rect(&self) -> Rect {
        let modifiers = self.current.modifiers;
        let mut by = self.current.page - self.start;
        if modifiers.shift() {
            let side = by.x.abs().max(by.y.abs());
            by = Vec2::new(side.copysign(by.x), side.copysign(by.y));
        }
        match modifiers.alt() {
            true => Rect::from_points(self.start - by, self.start + by),
            false => Rect::from_points(self.start, self.start + by),
        }
    }

    pub fn moved(&mut self, samples: &[Sample]) {
        if let Some(last) = samples.last() {
            self.current = *last;
        }
    }

    fn path(&self) -> BezPath {
        let rect = self.rect();
        match self.ellipse {
            true => Ellipse::from_rect(rect).to_path(TOLERANCE),
            false => rect.to_path(TOLERANCE),
        }
    }

    /// The shape where the drag went, at the top of the page, selected.
    pub fn finish(self, cx: &Context) -> Vec<Effect> {
        let rect = self.rect();
        let click = cx.view.page_length(CLICK);
        if rect.width() < click || rect.height() < click {
            return Vec::new();
        }
        let Ok(path) = Path::try_from(self.path().to_svg()) else {
            return Vec::new();
        };
        let shape = Shape {
            path,
            fill_rule: FillRule::default(),
            fill: Some(ShapeFill::Solid(cx.brush.color)),
            stroke: None,
        };
        let (id, add) = added(cx.snapshot.page(), None, Content::Shape(shape));
        vec![
            Effect::Edit(Edit {
                label: "add_shape".to_owned(),
                ops: vec![add],
            }),
            Effect::Selection(vec![id]),
        ]
    }

    pub fn preview(&self, cx: &Context, overlay: &mut Overlay) {
        let path = self.path();
        overlay.fill(&path, cx.brush.color);
        overlay.outline(&path);
    }
}
