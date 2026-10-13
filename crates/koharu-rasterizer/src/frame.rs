//! Retained Vello frame compiled from portable display data.

use std::{collections::HashMap, sync::Arc};

use vello::{
    FontEmbolden, Glyph, Scene,
    kurbo::{Affine, BezPath, Cap, Diagonal2, Join, Rect, Stroke},
    peniko::{Blob, Color, ColorStop, Extend, Fill, FontData, Gradient, Mix},
};

use crate::{
    Bounds, CompositionCommand, Error, FillRule, LayerId, LayerKind, PathElement, PreparedContent,
    PreparedElementFrame, PreparedFill, PreparedFrame, PreparedFrameBundle, PreparedFrameManifest,
    PreparedGradient, PreparedGradientExtend, PreparedGradientKind, PreparedLineCap,
    PreparedLineJoin, PreparedPath, PreparedResource, PreparedResourceStore, PreparedScene,
    PreparedSceneCommand, PreparedStroke, Presentation, RasterDraw, ResourceId, Result, Revision,
};

#[derive(Clone)]
pub struct Frame(Arc<FrameData>);

struct FrameData {
    revision: Revision,
    page: LayerId,
    width: u32,
    height: u32,
    origin: (i32, i32),
    normalization: Affine,
    layers: Arc<[Layer]>,
    layer_index: HashMap<LayerId, usize>,
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Frame")
            .field("revision", &self.revision())
            .field("page", &self.page())
            .field("size", &self.size())
            .field("origin", &self.origin())
            .field("layers", &self.layers())
            .finish_non_exhaustive()
    }
}

impl Frame {
    #[must_use]
    pub fn revision(&self) -> Revision {
        self.0.revision
    }

    #[must_use]
    pub fn page(&self) -> LayerId {
        self.0.page
    }

    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.0.width, self.0.height)
    }

    #[must_use]
    pub fn origin(&self) -> (i32, i32) {
        self.0.origin
    }

    #[must_use]
    pub fn normalization(&self) -> Affine {
        self.0.normalization
    }

    #[must_use]
    pub fn layers(&self) -> &[Layer] {
        &self.0.layers
    }

    #[must_use]
    pub fn layer(&self, id: LayerId) -> Option<&Layer> {
        self.0
            .layer_index
            .get(&id)
            .map(|index| &self.0.layers[*index])
    }

    /// Builds the ordered composition used by native headless rasterization.
    #[must_use]
    pub fn composition_commands(&self, scale: u32) -> Vec<CompositionCommand> {
        let outer = Affine::scale(f64::from(scale.max(1))) * self.normalization();
        let mut commands = Vec::new();
        let mut vectors = Scene::new();
        let mut vectors_pending = false;
        for layer in self.layers() {
            let presentation = layer.presentation();
            if let Some(image) = layer.raster_image() {
                if vectors_pending {
                    commands.push(CompositionCommand::Vector(Box::new(std::mem::replace(
                        &mut vectors,
                        Scene::new(),
                    ))));
                    vectors_pending = false;
                }
                if presentation.visible
                    && presentation.opacity.is_finite()
                    && presentation.opacity > 0.0
                {
                    commands.push(CompositionCommand::Raster(RasterDraw {
                        image: image.clone(),
                        transform: outer * layer.placement(),
                        opacity: presentation.opacity.clamp(0.0, 1.0),
                        erase: false,
                    }));
                }
            } else {
                layer.append_vector_with_presentation(&mut vectors, Some(outer), presentation);
                vectors_pending |= presentation.visible
                    && presentation.opacity.is_finite()
                    && presentation.opacity > 0.0;
            }
        }
        if vectors_pending {
            commands.push(CompositionCommand::Vector(Box::new(vectors)));
        }
        commands
    }
}

#[derive(Clone)]
pub struct Layer(Arc<LayerData>);

struct LayerData {
    id: LayerId,
    geometry: Arc<[crate::Point]>,
    bounds: Bounds,
    local_bounds: Bounds,
    presentation: Presentation,
    kind: LayerKind,
    placement: Affine,
    scene: Arc<Scene>,
    image: Option<RasterImage>,
    element_frame: Option<PreparedElementFrame>,
}

impl std::fmt::Debug for Layer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Layer")
            .field("id", &self.id())
            .field("bounds", &self.bounds())
            .field("presentation", &self.presentation())
            .field("kind", &self.kind())
            .finish_non_exhaustive()
    }
}

impl Layer {
    #[must_use]
    pub fn id(&self) -> LayerId {
        self.0.id
    }

    #[must_use]
    pub fn geometry(&self) -> &[crate::Point] {
        &self.0.geometry
    }

    #[must_use]
    pub fn bounds(&self) -> Bounds {
        self.0.bounds
    }

    #[must_use]
    pub fn presentation(&self) -> Presentation {
        self.0.presentation
    }

    #[must_use]
    pub fn kind(&self) -> LayerKind {
        self.0.kind
    }

    #[must_use]
    pub fn placement(&self) -> Affine {
        self.0.placement
    }

    #[must_use]
    pub fn raster_image(&self) -> Option<&RasterImage> {
        self.0.image.as_ref()
    }

    #[must_use]
    pub fn element_frame(&self) -> Option<PreparedElementFrame> {
        self.0.element_frame
    }

    pub fn append_vector_with_presentation(
        &self,
        scene: &mut Scene,
        transform: Option<Affine>,
        presentation: Presentation,
    ) {
        if self.raster_image().is_some()
            || !presentation.visible
            || !presentation.opacity.is_finite()
            || presentation.opacity <= 0.0
        {
            return;
        }
        let opacity = presentation.opacity.clamp(0.0, 1.0);
        let placement = transform.map_or(self.0.placement, |outer| outer * self.0.placement);
        if opacity < 1.0 {
            let bounds = self.0.local_bounds;
            scene.push_layer(
                Fill::NonZero,
                Mix::Normal,
                opacity,
                placement,
                &Rect::new(
                    f64::from(bounds.x),
                    f64::from(bounds.y),
                    f64::from(bounds.x + bounds.width),
                    f64::from(bounds.y + bounds.height),
                ),
            );
        }
        scene.append(&self.0.scene, Some(placement));
        if opacity < 1.0 {
            scene.pop_layer();
        }
    }
}

#[derive(Clone)]
pub struct RasterImage {
    source: ResourceId,
    width: u32,
    height: u32,
    tiles: Arc<[RasterTile]>,
    pixels: Option<Arc<[u8]>>,
}

#[derive(Clone)]
pub struct RasterTile {
    id: ResourceId,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    gutter: [u32; 4],
}

impl RasterImage {
    #[must_use]
    pub fn source(&self) -> ResourceId {
        self.source
    }

    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[must_use]
    pub fn tiles(&self) -> &[RasterTile] {
        &self.tiles
    }

    pub(crate) fn pixels(&self) -> Option<&Arc<[u8]>> {
        self.pixels.as_ref()
    }
}

impl RasterTile {
    #[must_use]
    pub fn id(&self) -> ResourceId {
        self.id
    }

    #[must_use]
    pub fn origin(&self) -> (u32, u32) {
        (self.x, self.y)
    }

    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[must_use]
    pub fn gutter(&self) -> [u32; 4] {
        self.gutter
    }

    #[must_use]
    pub const fn source_origin(&self) -> (u32, u32) {
        (self.x - self.gutter[0], self.y - self.gutter[1])
    }

    #[must_use]
    pub fn resource_size(&self) -> (u32, u32) {
        (
            self.width + self.gutter[0] + self.gutter[2],
            self.height + self.gutter[1] + self.gutter[3],
        )
    }
}

impl PreparedFrameBundle {
    pub fn into_frame_with_raster_sources(
        self,
        raster_sources: &HashMap<ResourceId, Arc<[u8]>>,
    ) -> Result<Frame> {
        self.validate()?;
        let Self { frame, resources } = self;
        let resources = resources
            .iter()
            .map(|resource| (resource.id(), resource))
            .collect::<HashMap<_, _>>();
        compile_frame(frame, &resources, Some(raster_sources))
    }
}

pub(crate) fn compile_manifest(
    manifest: &PreparedFrameManifest,
    store: &PreparedResourceStore,
) -> Result<Frame> {
    let mut resources = HashMap::with_capacity(manifest.resources.len());
    for expected in manifest.required_resources() {
        let packet = store.get(expected.id).ok_or_else(|| {
            Error::invalid(format!(
                "prepared resource {} is not installed",
                expected.id
            ))
        })?;
        if packet.resource().reference() != *expected {
            return Err(Error::invalid(format!(
                "installed prepared resource {} does not match its manifest reference",
                expected.id
            )));
        }
        resources.insert(expected.id, packet.resource());
    }
    compile_frame(manifest.frame.clone(), &resources, None)
}

fn compile_frame(
    frame: PreparedFrame,
    resources: &HashMap<ResourceId, &PreparedResource>,
    raster_sources: Option<&HashMap<ResourceId, Arc<[u8]>>>,
) -> Result<Frame> {
    let mut font_blobs = HashMap::new();
    let mut layers = Vec::with_capacity(frame.layers.len());
    let mut layer_index = HashMap::with_capacity(frame.layers.len());
    for prepared in frame.layers {
        let (scene, image) = match &prepared.content {
            PreparedContent::Raster(raster) => {
                let Some(PreparedResource::EncodedRaster { width, height, .. }) =
                    resources.get(&raster.source).copied()
                else {
                    return Err(Error::invalid(
                        "raster references a missing encoded source resource",
                    ));
                };
                if (*width, *height) != (raster.width, raster.height) {
                    return Err(Error::invalid(
                        "encoded source dimensions do not match the raster",
                    ));
                }
                let pixels = raster_sources
                    .map(|sources| {
                        sources.get(&raster.source).cloned().ok_or_else(|| {
                            Error::invalid(format!(
                                "native raster source {} is not installed",
                                raster.source
                            ))
                        })
                    })
                    .transpose()?;
                if let Some(pixels) = pixels.as_ref() {
                    let expected =
                        usize::try_from(u64::from(raster.width) * u64::from(raster.height) * 4)
                            .map_err(|_| {
                                Error::invalid("native raster byte length exceeds usize")
                            })?;
                    if pixels.len() != expected {
                        return Err(Error::invalid(
                            "native raster byte length does not match its dimensions",
                        ));
                    }
                }
                let mut tiles = Vec::with_capacity(raster.tiles.len());
                for prepared_tile in &raster.tiles {
                    tiles.push(RasterTile {
                        id: prepared_tile.id(raster.source),
                        x: prepared_tile.x,
                        y: prepared_tile.y,
                        width: prepared_tile.width,
                        height: prepared_tile.height,
                        gutter: prepared_tile.gutter,
                    });
                }
                (
                    Scene::new(),
                    Some(RasterImage {
                        source: raster.source,
                        width: raster.width,
                        height: raster.height,
                        tiles: tiles.into(),
                        pixels,
                    }),
                )
            }
            PreparedContent::Vector(prepared_scene) => (
                compile_scene(prepared_scene, resources, &mut font_blobs)?,
                None,
            ),
        };
        let layer = Layer(Arc::new(LayerData {
            id: prepared.id,
            geometry: prepared.geometry.into(),
            bounds: prepared.bounds,
            local_bounds: prepared.local_bounds,
            presentation: prepared.presentation,
            kind: prepared.kind,
            placement: Affine::new(prepared.placement),
            scene: Arc::new(scene),
            image,
            element_frame: prepared.element_frame,
        }));
        layer_index.insert(layer.id(), layers.len());
        layers.push(layer);
    }
    Ok(Frame(Arc::new(FrameData {
        revision: frame.revision,
        page: frame.page,
        width: frame.width,
        height: frame.height,
        origin: frame.origin,
        normalization: Affine::new(frame.normalization),
        layers: layers.into(),
        layer_index,
    })))
}

fn compile_scene(
    prepared: &PreparedScene,
    resources: &HashMap<ResourceId, &PreparedResource>,
    font_blobs: &mut HashMap<ResourceId, Blob<u8>>,
) -> Result<Scene> {
    let mut scene = Scene::new();
    for command in &prepared.commands {
        match command {
            PreparedSceneCommand::GlyphRun(prepared) => {
                let Some(PreparedResource::Font { bytes, .. }) =
                    resources.get(&prepared.font).copied()
                else {
                    return Err(Error::invalid(
                        "glyph run references a missing font resource",
                    ));
                };
                let blob = font_blobs
                    .entry(prepared.font)
                    .or_insert_with(|| Blob::new(Arc::new(Arc::clone(bytes))))
                    .clone();
                let font = FontData::new(blob, prepared.font_index);
                let mut run = scene
                    .draw_glyphs(&font)
                    .font_size(prepared.font_size)
                    .transform(Affine::new(prepared.transform))
                    .hint(prepared.hint);
                if !prepared.normalized_coords.is_empty() {
                    run = run.normalized_coords(&prepared.normalized_coords);
                }
                if let Some(transform) = prepared.glyph_transform {
                    run = run.glyph_transform(Some(Affine::new(transform)));
                }
                if prepared.embolden != [0.0, 0.0] {
                    // Round joins avoid the long miter spikes a dilated outline can otherwise
                    // produce at sharp glyph corners once the border width is large.
                    run = run.font_embolden(
                        FontEmbolden::new(Diagonal2::new(
                            f64::from(prepared.embolden[0]),
                            f64::from(prepared.embolden[1]),
                        ))
                        .with_join(Join::Round),
                    );
                }
                let glyphs = prepared.glyphs.iter().map(|glyph| Glyph {
                    id: glyph.id,
                    x: glyph.x,
                    y: glyph.y,
                });
                run.brush(rgba(prepared.color)).draw(Fill::NonZero, glyphs);
            }
            PreparedSceneCommand::FillPath(prepared) => prepared.append_to(&mut scene),
        }
    }
    Ok(scene)
}

impl PreparedPath {
    /// Draws this path into a Vello scene.
    ///
    /// Committed frames and the browser's live shape preview both draw through
    /// here, so a shape cannot look one way while it is being dragged and
    /// another way once it is committed.
    pub fn append_to(&self, scene: &mut Scene) {
        let path = compile_path(self);
        let affine = Affine::new(self.transform);
        match &self.fill {
            Some(PreparedFill::Solid { color, rule }) => {
                scene.fill(fill(rule), affine, rgba(*color), None, &path);
            }
            Some(PreparedFill::Gradient { gradient, rule }) => {
                // `brush_transform` must stay `None` here. Vello *composes* it
                // with the path transform (`Transform::from_kurbo(&(transform *
                // brush_transform))` at `vello-0.10.0/src/scene.rs:328-331`), so
                // handing it the path affine would apply that affine to the
                // brush twice: a rotated gradient would over-rotate or, at some
                // angles, collapse to a single colour. The control points are
                // already in path-local coordinates, so the path transform
                // rotates the brush and the outline together with no extra work.
                scene.fill(fill(rule), affine, &compile_gradient(gradient), None, &path);
            }
            None => {}
        }
        if let Some(stroke) = &self.stroke {
            scene.stroke(
                &compile_stroke(stroke),
                affine,
                rgba(stroke.color),
                None,
                &path,
            );
        }
    }
}

fn fill(rule: &FillRule) -> Fill {
    match rule {
        FillRule::NonZero => Fill::NonZero,
        FillRule::EvenOdd => Fill::EvenOdd,
    }
}

fn compile_gradient(prepared: &PreparedGradient) -> Gradient {
    let stops = crate::stops_sorted(&prepared.stops)
        .into_iter()
        .map(|stop| ColorStop {
            offset: stop.offset,
            color: rgba(stop.color).into(),
        })
        .collect::<Vec<_>>();
    let gradient = match &prepared.kind {
        PreparedGradientKind::Linear { start, end } => {
            Gradient::new_linear((start[0], start[1]), (end[0], end[1]))
        }
        PreparedGradientKind::Radial { center, radius } => {
            Gradient::new_radial((center[0], center[1]), *radius)
        }
        // Sweep angles are stored in degrees on the portable side; peniko
        // measures them in radians.
        PreparedGradientKind::Sweep {
            center,
            start_degrees,
            end_degrees,
        } => Gradient::new_sweep(
            (center[0], center[1]),
            start_degrees.to_radians(),
            end_degrees.to_radians(),
        ),
    };
    gradient
        .with_extend(match prepared.extend {
            PreparedGradientExtend::Pad => Extend::Pad,
            PreparedGradientExtend::Repeat => Extend::Repeat,
            PreparedGradientExtend::Reflect => Extend::Reflect,
        })
        .with_stops(&stops[..])
}

fn compile_path(prepared: &PreparedPath) -> BezPath {
    let mut path = BezPath::new();
    for element in &prepared.elements {
        match *element {
            PathElement::MoveTo([x, y]) => path.move_to((x, y)),
            PathElement::LineTo([x, y]) => path.line_to((x, y)),
            PathElement::QuadTo([x1, y1, x2, y2]) => path.quad_to((x1, y1), (x2, y2)),
            PathElement::CurveTo([x1, y1, x2, y2, x3, y3]) => {
                path.curve_to((x1, y1), (x2, y2), (x3, y3));
            }
            PathElement::Close => path.close_path(),
        }
    }
    path
}

fn compile_stroke(prepared: &PreparedStroke) -> Stroke {
    let mut stroke = Stroke::new(f64::from(prepared.width))
        .with_caps(match prepared.cap {
            PreparedLineCap::Butt => Cap::Butt,
            PreparedLineCap::Round => Cap::Round,
            PreparedLineCap::Square => Cap::Square,
        })
        .with_join(match prepared.join {
            PreparedLineJoin::Miter => Join::Miter,
            PreparedLineJoin::Round => Join::Round,
            PreparedLineJoin::Bevel => Join::Bevel,
        });
    if let Some(dash) = &prepared.dash {
        stroke = stroke.with_dashes(0.0, dash.iter().map(|length| f64::from(*length)));
    }
    stroke
}

fn rgba([r, g, b, a]: [u8; 4]) -> Color {
    Color::from_rgba8(r, g, b, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Point, PreparedFrame, PreparedGradient, PreparedGradientExtend, PreparedGradientKind,
        PreparedGradientStop, PreparedLayer, PreparedRaster, PreparedRasterTile, PreparedScene,
        PreparedSceneCommand,
    };
    use vello::peniko::{ColorStop, GradientKind};

    /// A gradient path whose kind, stops, extend, and transform are all
    /// non-trivial.
    ///
    /// The encoding comparison this feeds cannot see the brush payload, so stop
    /// order and `extend` are pinned by the direct `compile_gradient` tests
    /// below, not here. What it *can* see is the brush transform and the kind's
    /// numeric payload (including a missing degree→radian conversion), which is
    /// why the transform is non-identity and the kind is a sweep.
    fn gradient_path(kind: PreparedGradientKind) -> PreparedPath {
        PreparedPath {
            transform: [0.0, 1.0, -1.0, 0.0, 30.0, 40.0],
            elements: vec![
                PathElement::MoveTo([1.0, 2.0]),
                PathElement::LineTo([9.0, 2.0]),
                PathElement::Close,
            ],
            fill: Some(PreparedFill::Gradient {
                gradient: PreparedGradient {
                    kind,
                    stops: vec![
                        PreparedGradientStop {
                            offset: 0.75,
                            color: [255, 0, 0, 255],
                        },
                        PreparedGradientStop {
                            offset: 0.25,
                            color: [0, 0, 255, 255],
                        },
                    ],
                    extend: PreparedGradientExtend::Reflect,
                },
                rule: FillRule::NonZero,
            }),
            stroke: None,
        }
    }

    /// Paths whose transform, fill, and stroke are all non-trivial, so a preview
    /// that drops or reorders any of them changes the encoding.
    fn shape_path() -> PreparedPath {
        PreparedPath {
            transform: [0.0, 1.0, -1.0, 0.0, 30.0, 40.0],
            elements: vec![
                PathElement::MoveTo([1.0, 2.0]),
                PathElement::LineTo([9.0, 2.0]),
                PathElement::QuadTo([11.0, 6.0, 9.0, 10.0]),
                PathElement::CurveTo([7.0, 14.0, 3.0, 14.0, 1.0, 10.0]),
                PathElement::Close,
            ],
            fill: Some(PreparedFill::Solid {
                color: [10, 20, 30, 255],
                rule: FillRule::EvenOdd,
            }),
            stroke: Some(PreparedStroke {
                color: [1, 2, 3, 128],
                width: 3.0,
                dash: Some(vec![5.0, 2.0]),
                cap: PreparedLineCap::Square,
                join: PreparedLineJoin::Bevel,
            }),
        }
    }

    fn shape_layer_frame(path: PreparedPath) -> Frame {
        let bundle = PreparedFrameBundle {
            frame: PreparedFrame {
                revision: Revision::new(1),
                page: LayerId::from_bytes([7; 16]),
                width: 100,
                height: 100,
                origin: (0, 0),
                normalization: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                layers: vec![PreparedLayer {
                    id: LayerId::from_bytes([3; 16]),
                    geometry: vec![
                        Point { x: 0.0, y: 0.0 },
                        Point { x: 1.0, y: 0.0 },
                        Point { x: 0.0, y: 1.0 },
                    ],
                    bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                    },
                    local_bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                    },
                    presentation: Presentation {
                        visible: true,
                        opacity: 1.0,
                    },
                    kind: LayerKind::Shape,
                    placement: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    content: PreparedContent::Vector(PreparedScene {
                        commands: vec![PreparedSceneCommand::FillPath(path)],
                    }),
                    element_frame: None,
                }],
            },
            resources: Vec::new(),
        };
        bundle
            .into_frame_with_raster_sources(&HashMap::new())
            .unwrap()
    }

    /// The browser's live shape preview appends a prepared path straight into a
    /// scene instead of going through a committed frame. That path must issue
    /// the same draws the committed frame does, or a shape would change
    /// appearance when the drag ends.
    ///
    /// The expected scene is built with raw Vello calls rather than by reusing
    /// `append_to`, so dropping the fill, the stroke, or the transform fails
    /// this test instead of moving both sides together.
    #[test]
    fn appending_a_prepared_path_issues_the_same_draws_as_a_committed_frame() {
        let path = shape_path();

        let mut preview = Scene::new();
        path.append_to(&mut preview);

        let mut expected = Scene::new();
        let mut bez = BezPath::new();
        bez.move_to((1.0, 2.0));
        bez.line_to((9.0, 2.0));
        bez.quad_to((11.0, 6.0), (9.0, 10.0));
        bez.curve_to((7.0, 14.0), (3.0, 14.0), (1.0, 10.0));
        bez.close_path();
        let affine = Affine::new([0.0, 1.0, -1.0, 0.0, 30.0, 40.0]);
        expected.fill(
            Fill::EvenOdd,
            affine,
            Color::from_rgba8(10, 20, 30, 255),
            None,
            &bez,
        );
        expected.stroke(
            &Stroke::new(3.0)
                .with_caps(Cap::Square)
                .with_join(Join::Bevel)
                .with_dashes(0.0, [5.0, 2.0]),
            affine,
            Color::from_rgba8(1, 2, 3, 128),
            None,
            &bez,
        );

        let frame = shape_layer_frame(path);
        let mut committed = Scene::new();
        frame.layers()[0].append_vector_with_presentation(
            &mut committed,
            None,
            Presentation {
                visible: true,
                opacity: 1.0,
            },
        );

        assert_encoding_matches(&preview, &expected);
        assert_encoding_matches(&committed, &preview);
    }

    /// The browser's live shape preview appends a prepared path straight into a
    /// scene instead of going through a committed frame. A gradient path must
    /// issue the same draws there too, with the gradient control points left in
    /// path-local coordinates and **no** brush transform.
    ///
    /// The expected scene is built with `brush_transform: None`. Vello composes
    /// any brush transform with the path transform, so `Some(affine)` here would
    /// apply the path affine to the brush twice — a bug the committed frame and
    /// the preview would share, which is exactly why this test pins `None`
    /// rather than merely agreeing with `append_to`.
    #[test]
    fn appending_a_gradient_path_passes_no_brush_transform() {
        let path = gradient_path(PreparedGradientKind::Sweep {
            center: [5.0, 5.0],
            start_degrees: 0.0,
            end_degrees: 90.0,
        });

        let mut preview = Scene::new();
        path.append_to(&mut preview);

        let mut expected = Scene::new();
        let mut bez = BezPath::new();
        bez.move_to((1.0, 2.0));
        bez.line_to((9.0, 2.0));
        bez.close_path();
        let affine = Affine::new([0.0, 1.0, -1.0, 0.0, 30.0, 40.0]);
        expected.fill(
            Fill::NonZero,
            affine,
            &Gradient::new_sweep((5.0, 5.0), 0.0f32.to_radians(), 90.0f32.to_radians())
                .with_extend(Extend::Reflect)
                .with_stops([
                    ColorStop {
                        offset: 0.25,
                        color: Color::from_rgba8(0, 0, 255, 255).into(),
                    },
                    ColorStop {
                        offset: 0.75,
                        color: Color::from_rgba8(255, 0, 0, 255).into(),
                    },
                ]),
            None,
            &bez,
        );

        let frame = shape_layer_frame(path);
        let mut committed = Scene::new();
        frame.layers()[0].append_vector_with_presentation(
            &mut committed,
            None,
            Presentation {
                visible: true,
                opacity: 1.0,
            },
        );

        assert_encoding_matches(&preview, &expected);
        assert_encoding_matches(&committed, &preview);
    }

    /// A solid fill must keep passing `brush_transform: None`; a regression that
    /// started rotating solid colours would still paint identically, so only the
    /// encoding comparison can catch it.
    #[test]
    fn a_solid_fill_keeps_no_brush_transform() {
        let mut path = shape_path();
        path.stroke = None;

        let mut preview = Scene::new();
        path.append_to(&mut preview);

        let mut expected = Scene::new();
        let mut bez = BezPath::new();
        bez.move_to((1.0, 2.0));
        bez.line_to((9.0, 2.0));
        bez.quad_to((11.0, 6.0), (9.0, 10.0));
        bez.curve_to((7.0, 14.0), (3.0, 14.0), (1.0, 10.0));
        bez.close_path();
        let affine = Affine::new([0.0, 1.0, -1.0, 0.0, 30.0, 40.0]);
        expected.fill(
            Fill::EvenOdd,
            affine,
            Color::from_rgba8(10, 20, 30, 255),
            None,
            &bez,
        );

        assert_encoding_matches(&preview, &expected);
    }

    /// `Scene`'s encoding carries the brush *transform* as a path tag but not
    /// the brush payload, which lives in the scene's resource list. The
    /// encoding comparison above therefore cannot see stop order or `extend`,
    /// so the assembled gradient is compared directly here. A wrong kind or a
    /// missing degree-to-radian conversion *is* visible there, via `draw_data`.
    #[test]
    fn a_gradient_compiles_with_sorted_stops_and_mapped_extend() {
        let gradient = compile_gradient(&PreparedGradient {
            kind: PreparedGradientKind::Radial {
                center: [5.0, 6.0],
                radius: 25.0,
            },
            stops: vec![
                PreparedGradientStop {
                    offset: 0.75,
                    color: [255, 0, 0, 255],
                },
                PreparedGradientStop {
                    offset: 0.25,
                    color: [0, 0, 255, 255],
                },
            ],
            extend: PreparedGradientExtend::Repeat,
        });

        let expected = Gradient::new_radial((5.0f32, 6.0f32), 25.0)
            .with_extend(Extend::Repeat)
            .with_stops([
                ColorStop {
                    offset: 0.25,
                    color: Color::from_rgba8(0, 0, 255, 255).into(),
                },
                ColorStop {
                    offset: 0.75,
                    color: Color::from_rgba8(255, 0, 0, 255).into(),
                },
            ]);
        assert_eq!(gradient, expected);
    }

    /// Sweep is the only kind whose stored units differ from peniko's: degrees
    /// on the portable side, radians in the assembled gradient.
    #[test]
    fn sweep_angles_are_converted_from_degrees_to_radians() {
        let gradient = compile_gradient(&PreparedGradient {
            kind: PreparedGradientKind::Sweep {
                center: [1.0, 2.0],
                start_degrees: 0.0,
                end_degrees: 180.0,
            },
            stops: vec![
                PreparedGradientStop {
                    offset: 0.0,
                    color: [0, 0, 0, 255],
                },
                PreparedGradientStop {
                    offset: 1.0,
                    color: [255, 255, 255, 255],
                },
            ],
            extend: PreparedGradientExtend::Pad,
        });

        let expected =
            Gradient::new_sweep((1.0f32, 2.0f32), 0.0f32.to_radians(), 180.0f32.to_radians())
                .with_extend(Extend::Pad)
                .with_stops([
                    ColorStop {
                        offset: 0.0,
                        color: Color::from_rgba8(0, 0, 0, 255).into(),
                    },
                    ColorStop {
                        offset: 1.0,
                        color: Color::from_rgba8(255, 255, 255, 255).into(),
                    },
                ]);
        assert_eq!(gradient, expected);

        let GradientKind::Sweep(position) = gradient.kind else {
            panic!("a sweep gradient must assemble as a sweep");
        };
        assert_eq!(position.end_angle, std::f32::consts::PI);
    }

    /// Every `PreparedGradientExtend` value must reach peniko unchanged; a
    /// wildcard arm that fell back to `Pad` would otherwise go unnoticed.
    #[test]
    fn every_extend_mode_maps_across() {
        for (prepared, expected) in [
            (PreparedGradientExtend::Pad, Extend::Pad),
            (PreparedGradientExtend::Repeat, Extend::Repeat),
            (PreparedGradientExtend::Reflect, Extend::Reflect),
        ] {
            let gradient = compile_gradient(&PreparedGradient {
                kind: PreparedGradientKind::Linear {
                    start: [0.0, 0.0],
                    end: [1.0, 0.0],
                },
                stops: vec![
                    PreparedGradientStop {
                        offset: 0.0,
                        color: [0, 0, 0, 255],
                    },
                    PreparedGradientStop {
                        offset: 1.0,
                        color: [255, 255, 255, 255],
                    },
                ],
                extend: prepared,
            });
            assert_eq!(gradient.extend, expected, "{prepared:?}");
        }
    }

    /// Compares the encoded draw streams of two scenes.
    ///
    /// `PathTag`/`DrawTag` have no `Debug`, so the tag streams are compared by
    /// length and the data they index is compared exactly.
    fn assert_encoding_matches(left: &Scene, right: &Scene) {
        let left = left.encoding();
        let right = right.encoding();
        assert_eq!(left.n_paths, right.n_paths, "path count differs");
        assert_eq!(
            left.n_path_segments, right.n_path_segments,
            "path segment count differs"
        );
        assert_eq!(left.n_clips, right.n_clips, "clip count differs");
        assert_eq!(
            left.path_tags.len(),
            right.path_tags.len(),
            "path tag count differs"
        );
        assert_eq!(
            left.draw_tags.len(),
            right.draw_tags.len(),
            "draw tag count differs"
        );
        assert_eq!(left.path_data, right.path_data, "path data differs");
        assert_eq!(left.draw_data, right.draw_data, "draw data differs");
        assert_eq!(left.transforms, right.transforms, "transforms differ");
        assert_eq!(left.styles, right.styles, "styles differ");
    }

    #[test]
    fn compiled_frame_preserves_raster_identity_and_metadata() {
        let resource =
            PreparedResource::encoded_raster(1, 1, "image/png", Arc::from(&b"png"[..])).unwrap();
        let source_id = resource.id();
        let prepared_tile = PreparedRasterTile {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            gutter: [0; 4],
        };
        let bundle = PreparedFrameBundle {
            frame: PreparedFrame {
                revision: Revision::new(3),
                page: LayerId::from_bytes([9; 16]),
                width: 10,
                height: 20,
                origin: (-2, 4),
                normalization: [1.0, 0.0, 0.0, 1.0, 2.0, -4.0],
                layers: vec![PreparedLayer {
                    id: LayerId::from_bytes([4; 16]),
                    geometry: vec![
                        Point { x: 0.0, y: 0.0 },
                        Point { x: 1.0, y: 0.0 },
                        Point { x: 0.0, y: 1.0 },
                    ],
                    bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                    },
                    local_bounds: Bounds {
                        x: 0.0,
                        y: 0.0,
                        width: 1.0,
                        height: 1.0,
                    },
                    presentation: Presentation {
                        visible: true,
                        opacity: 0.5,
                    },
                    kind: LayerKind::Raster,
                    placement: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    content: PreparedContent::Raster(PreparedRaster {
                        source: source_id,
                        width: 1,
                        height: 1,
                        tiles: vec![prepared_tile],
                    }),
                    element_frame: None,
                }],
            },
            resources: vec![resource],
        };
        let frame = bundle
            .into_frame_with_raster_sources(&HashMap::from([(
                source_id,
                Arc::from(&[1, 2, 3, 4][..]),
            )]))
            .unwrap();
        assert_eq!(frame.revision(), Revision::new(3));
        assert_eq!(frame.origin(), (-2, 4));
        assert_eq!(
            frame.layers()[0].raster_image().unwrap().tiles()[0].id(),
            prepared_tile.id(source_id)
        );
        assert_eq!(
            frame.layers()[0]
                .raster_image()
                .unwrap()
                .pixels()
                .unwrap()
                .as_ref(),
            &[1, 2, 3, 4]
        );
    }
}
