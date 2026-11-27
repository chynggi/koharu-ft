//! Retained content: pixels, shapes and text built from the scene, kept by
//! what changes them, so a render after an edit rebuilds only what it
//! touched.

use std::{io::Cursor, sync::Arc};

use anyhow::{Context as _, Result, anyhow};
use image::{ImageReader, RgbaImage};
use koharu_scene::{
    FillRule, Gradient, GradientKind, Hash, LayerId, Offset, PaintExtend, Project, Session, Shape,
    ShapeFill, Text,
};
use kurbo::{Affine, BezPath, Cap, Join, Point, Rect, Shape as _, Stroke};
use vello_gpu::{
    PaintType,
    peniko::{self, Color, ColorStop, Extend, Fill},
};

use super::Renderer;
use crate::{
    fonts,
    frame::{Content, ShapePaint, ShapeVector},
    pixels::{Pixels, assemble, check_size},
    text,
};

/// Each source's content, by the hash of what it's built from: an image's
/// file, a picture's tiles, or a text's or shape's canonical JSON (RFC
/// 8785), which stays the same for equal values whatever order a map's keys
/// come in. No two kinds hash the same bytes: a file is an image, the tiles
/// hash binary, a text's JSON is an array, and a shape's an object. `None`
/// for a source that draws nothing.
pub(super) type Cache = quick_cache::sync::Cache<Hash, Option<Content>, Weight>;

/// What retained content may take, in bytes: pixels count with their levels;
/// a text's layout or a shape counts as a kilobyte.
const CACHE_BYTES: u64 = 2 << 30;

#[derive(Clone)]
pub(super) struct Weight;

impl quick_cache::Weighter<Hash, Option<Content>> for Weight {
    fn weight(&self, _: &Hash, content: &Option<Content>) -> u64 {
        match content {
            Some(Content::Pixels(pixels)) => pixels.bytes(),
            Some(Content::Shape(shape)) => match &shape.fill {
                Some(ShapePaint::Pattern { pixels, .. }) => pixels.bytes(),
                _ => 1024,
            },
            _ => 1024,
        }
    }
}

pub(super) fn cache() -> Cache {
    // A shard keeps nothing heavier than its share of the bytes, so one
    // shard holds them all and a large page's pixels stay.
    quick_cache::sync::Cache::with_options(
        quick_cache::OptionsBuilder::new()
            .estimated_items_capacity(1024)
            .weight_capacity(CACHE_BYTES)
            .shards(1)
            .build()
            .expect("both capacities are set"),
        Weight,
        Default::default(),
        Default::default(),
    )
}

/// Imported file `file` decoded, its size checked before any pixel is.
fn decode(session: &Session, file: Hash) -> Result<RgbaImage> {
    let bytes = session.file(&file)?;
    let unsupported = || format!("file {file} is not a supported image");
    let reader = || {
        ImageReader::new(Cursor::new(&bytes))
            .with_guessed_format()
            .with_context(unsupported)
    };
    let (width, height) = reader()?.into_dimensions().with_context(unsupported)?;
    check_size(width, height)?;
    Ok(reader()?.decode().with_context(unsupported)?.into_rgba8())
}

/// A point normalized to `area`, in the shape's coordinates.
fn at(area: Rect, [x, y]: [f64; 2]) -> Point {
    Point::new(area.x0 + x * area.width(), area.y0 + y * area.height())
}

fn extend(extend: PaintExtend) -> Extend {
    match extend {
        PaintExtend::Pad => Extend::Pad,
        PaintExtend::Repeat => Extend::Repeat,
        PaintExtend::Reflect => Extend::Reflect,
    }
}

/// `gradient` across `area`, its stops in order.
fn paint_gradient(gradient: &Gradient, area: Rect) -> peniko::Gradient {
    let painted = match gradient.kind {
        GradientKind::Linear { start, end } => {
            peniko::Gradient::new_linear(at(area, start), at(area, end))
        }
        GradientKind::Radial { center, radius } => {
            peniko::Gradient::new_radial(at(area, center), (radius * area.width()) as f32)
        }
        GradientKind::Sweep {
            center,
            start_degrees,
            end_degrees,
        } => peniko::Gradient::new_sweep(
            at(area, center),
            start_degrees.to_radians() as f32,
            end_degrees.to_radians() as f32,
        ),
    };
    let mut stops: Vec<ColorStop> = gradient
        .stops
        .iter()
        .map(|stop| ColorStop::from((stop.offset, Color::from(stop.color))))
        .collect();
    stops.sort_by(|a, b| a.offset.total_cmp(&b.offset));
    painted
        .with_extend(extend(gradient.extend))
        .with_stops(stops.as_slice())
}

fn canonical(value: &impl serde::Serialize) -> Result<Hash> {
    let bytes = serde_json_canonicalizer::to_vec(value)
        .map_err(|error| anyhow!("a value has no canonical JSON: {error}"))?;
    Ok(Hash::of(&bytes))
}

/// Text layer `id`'s text laid out, falling back on `families`; `None` when
/// it has no characters.
pub(super) fn laid(
    id: LayerId,
    layer: &Text,
    families: &[String],
) -> Result<Option<text::TextLayout>> {
    text::layout(layer, |style, characters| {
        fonts::chain(style, families, characters, &layer.language)
    })
    .with_context(|| format!("text layer {id} couldn't be laid out"))
}

impl Renderer {
    /// What `key` holds, `build` building it when nothing does. Two renders
    /// may both build it: one waiting on the other's build inside rayon could
    /// deadlock.
    fn retain(
        &self,
        key: Hash,
        build: impl FnOnce() -> Result<Option<Content>>,
    ) -> Result<Option<Content>> {
        if let Some(retained) = self.0.cache.get(&key) {
            return Ok(retained);
        }
        let retained = build()?;
        self.0.cache.insert(key, retained.clone());
        Ok(retained)
    }

    /// Text layer `id`'s text laid out in its kernel's face, as `project`
    /// sets it, retained; `None` when it has no
    /// characters.
    pub(super) fn text(
        &self,
        project: &Project,
        id: LayerId,
        layer: &Text,
    ) -> Result<Option<Content>> {
        let (layer, families) = super::lettered(project, layer);
        // The transform places the layout when it's drawn, so moving a text
        // keeps its layout. Naming every field fails to build when one is
        // added, until the key says whether it counts.
        let Text {
            content,
            style,
            frame,
            direction,
            language,
            transform: _,
        } = &*layer;
        let key = canonical(&(content, style, frame, direction, language, &families))?;
        self.retain(key, || {
            let layout = laid(id, &layer, &families)?;
            Ok(layout.map(|layout| Content::Text(Arc::new(layout))))
        })
    }

    /// A shape layer's path, fill, and stroke: centred on the path, with
    /// miter joins and butt caps. A gradient or pattern spans the path's
    /// bounds.
    pub(super) fn shape(&self, session: &Session, shape: &Shape) -> Result<Option<Content>> {
        self.retain(canonical(shape)?, || {
            let path = BezPath::from(&shape.path);
            let area = path.bounding_box();
            let mut bounds = area;
            let stroke = shape.stroke.as_ref().map(|stroke| {
                bounds = bounds.inflate(stroke.width * 2.0, stroke.width * 2.0);
                (
                    Stroke::new(stroke.width)
                        .with_join(Join::Miter)
                        .with_caps(Cap::Butt),
                    Color::from(stroke.color),
                )
            });
            let fill = match &shape.fill {
                None => None,
                Some(ShapeFill::Solid(color)) => {
                    Some(ShapePaint::Brush(PaintType::Solid(Color::from(*color))))
                }
                Some(ShapeFill::Gradient { gradient }) => Some(ShapePaint::Brush(
                    PaintType::Gradient(paint_gradient(gradient, area)),
                )),
                Some(ShapeFill::Pattern { pattern }) => {
                    let image = decode(session, pattern.file)?;
                    let tile = pattern.scale * area.width() / f64::from(image.width().max(1));
                    Some(ShapePaint::Pattern {
                        transform: Affine::translate(at(area, pattern.origin).to_vec2())
                            * Affine::rotate(pattern.rotation_degrees.to_radians())
                            * Affine::scale(tile),
                        pixels: Arc::new(Pixels::new(Offset::default(), image)),
                        extend: extend(pattern.extend),
                    })
                }
            };
            Ok(Some(Content::Shape(Arc::new(ShapeVector {
                path,
                fill_rule: match shape.fill_rule {
                    FillRule::NonZero => Fill::NonZero,
                    FillRule::EvenOdd => Fill::EvenOdd,
                },
                fill,
                stroke,
                bounds,
            }))))
        })
    }

    /// Imported file `file` decoded, its size checked before any pixel is.
    pub(super) fn image(&self, session: &Session, file: Hash) -> Result<Option<Content>> {
        self.retain(file, || {
            Ok(Some(Content::Pixels(Arc::new(Pixels::new(
                Offset::default(),
                decode(session, file)?,
            )))))
        })
    }

    /// The picture of the layer or block part `layer`, cropped to what it
    /// paints; `None` when it paints nothing.
    pub(super) fn tiles(&self, session: &Session, layer: LayerId) -> Result<Option<Content>> {
        self.retain(session.tiles_hash(layer)?, || {
            Ok(assemble(session.tiles(layer, None)?)?
                .map(|pixels| Content::Pixels(Arc::new(pixels))))
        })
    }
}
