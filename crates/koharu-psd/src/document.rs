//! Scene-to-PSD document projection.
//!
//! GIMP reference: layer traversal and reverse layer-record ordering:
//! https://github.com/GNOME/gimp/blob/758fb4ed995bbb339282d3f777089a33f0a391b8/plug-ins/file-psd/psd-export.c#L1520-L1880
//! GIMP reference: `TySh` transforms and descriptors consumed by the importer:
//! https://github.com/GNOME/gimp/blob/758fb4ed995bbb339282d3f777089a33f0a391b8/plug-ins/file-psd/psd-layer-res-load.c#L1345-L1438

use std::sync::Arc;

use image::{DynamicImage, GrayImage, Rgba, RgbaImage};
use koharu_rasterizer::{Raster, RasterOptions, Rasterizer};
use koharu_renderer::{
    Frame, ImageKind, LayerKind, TextAlign, TextMetadata as RenderedText, WritingMode,
};
use koharu_scene::{AssetRole, EntityId, Snapshot};

use crate::{
    engine_data::{TextJustification, TextOrientation},
    error::PsdExportError,
};

const SOURCE_ROLE: &str = "source";
const TEXT_MASK_ROLE: &str = "text-mask";
const COO_MASK_ROLE: &str = "coo-mask";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextLayerMode {
    Rasterized,
    Editable,
}

#[derive(Debug, Clone)]
pub struct PsdExportOptions {
    pub include_source: bool,
    pub include_raster_layers: bool,
    pub include_removal_mask: bool,
    pub text_layer_mode: TextLayerMode,
}

// GIMP reference: default layer visibility and normal-layer flags are serialized here:
// https://github.com/GNOME/gimp/blob/758fb4ed995bbb339282d3f777089a33f0a391b8/plug-ins/file-psd/psd-export.c#L1689-L1721
impl Default for PsdExportOptions {
    fn default() -> Self {
        Self {
            include_source: true,
            include_raster_layers: true,
            include_removal_mask: true,
            text_layer_mode: TextLayerMode::Editable,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Document {
    pub width: u32,
    pub height: u32,
    pub merged: RgbaImage,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Clone)]
pub(crate) struct Layer {
    pub id: i32,
    pub name: String,
    pub left: i32,
    pub top: i32,
    pub pixels: RgbaImage,
    pub hidden: bool,
    pub text: Option<TextMetadata>,
}

#[derive(Debug, Clone)]
pub(crate) struct TextMetadata {
    pub index: i32,
    pub text: String,
    pub bounds: [f64; 4],
    pub transform: [f64; 6],
    pub orientation: TextOrientation,
    pub justification: TextJustification,
    pub font_index: usize,
    pub font_set: Vec<String>,
    pub font_size: f64,
    pub color: [u8; 4],
    pub box_width: f64,
    pub box_height: f64,
}

pub(crate) async fn build(
    snapshot: &Snapshot,
    frame: &Frame,
    rasterizer: Arc<Rasterizer>,
    raster_options: RasterOptions,
    options: &PsdExportOptions,
) -> Result<Document, PsdExportError> {
    let merged = rasterize(Arc::clone(&rasterizer), frame, raster_options).await?;
    let width = merged.image.width();
    let height = merged.image.height();
    validate_dimensions(width, height)?;

    let source = read_image(snapshot, frame.page(), SOURCE_ROLE).await?;
    if options.include_source && source.is_none() {
        return Err(PsdExportError::MissingAsset {
            page: frame.page(),
            role: SOURCE_ROLE,
        });
    }
    let has_raster_layers = frame.layers().iter().any(|layer| {
        matches!(
            layer.kind(),
            LayerKind::Image(metadata)
                if matches!(metadata.kind, ImageKind::Cleanup | ImageKind::Paint)
        )
    });
    let mut layers = Vec::new();
    if let Some(source) = source.filter(|_| options.include_source) {
        push_raster_layer(
            &mut layers,
            "Original Image",
            source.to_rgba8(),
            options.include_raster_layers && has_raster_layers,
        )?;
    }
    if options.include_removal_mask {
        let mask = combine_masks(
            read_image(snapshot, frame.page(), TEXT_MASK_ROLE).await?,
            read_image(snapshot, frame.page(), COO_MASK_ROLE).await?,
        )?;
        if let Some(mask) = mask {
            push_raster_layer(
                &mut layers,
                "Text Removal Mask",
                grayscale_mask_rgba(&mask),
                true,
            )?;
        }
    }
    let text_entities = frame
        .layers()
        .iter()
        .filter_map(|layer| match layer.kind() {
            LayerKind::Text(text) => Some((layer.entity(), text)),
            LayerKind::Image(_) | LayerKind::Shape(_) => None,
        })
        .filter(|(_, text)| !text.text.trim().is_empty())
        .collect::<Vec<_>>();
    let font_set = collect_fonts(text_entities.iter().map(|(_, text)| *text));
    let shape_entities = frame
        .layers()
        .iter()
        .filter(|layer| matches!(layer.kind(), LayerKind::Shape(_)))
        .map(|layer| layer.entity())
        .collect::<Vec<_>>();
    // GIMP writes the application layer list in reverse so PSD records remain bottom-to-top.
    // Reversing the complete visual list preserves interleaving between text and raster layers.
    for visual in frame.layers().iter().rev() {
        let image = match visual.kind() {
            LayerKind::Image(image) => Some(image),
            LayerKind::Text(_) | LayerKind::Shape(_) => None,
        };
        let raster =
            image.is_some_and(|image| matches!(image.kind, ImageKind::Cleanup | ImageKind::Paint));
        let ordinary_image = image.is_some_and(|image| image.kind == ImageKind::Embedded);
        if (raster && options.include_raster_layers) || ordinary_image {
            let rendered =
                render_entity(frame, &rasterizer, raster_options, visual.entity()).await?;
            let image = image.expect("image branch was selected above");
            let name = image.name.clone().unwrap_or_else(|| match image.kind {
                ImageKind::Source | ImageKind::Embedded => "Image".to_owned(),
                ImageKind::Cleanup => "Cleanup".to_owned(),
                ImageKind::Paint => "Paint".to_owned(),
            });
            validate_pixels(&name, &rendered.image)?;
            layers.push(Layer {
                id: 0,
                name,
                left: rendered.left,
                top: rendered.top,
                pixels: rendered.image,
                hidden: false,
                text: None,
            });
        } else if let Some(offset) = shape_entities
            .iter()
            .position(|entity| *entity == visual.entity())
        {
            // A shape exports as an ordinary raster layer. PSD's vector shape
            // record (`SoCo`) is deliberately not written: the record's
            // descriptor format is far more intricate than anything else here,
            // `koharu-psd` already has a working crop-then-rasterize path, and
            // only text is kept editable (`TextLayerMode`), so a vector shape
            // would be the single exception to an otherwise uniform policy.
            //
            // The pixels come from the rendered frame, not from re-deriving the
            // path from `ShapeMetadata`, so a shape cannot export differently
            // from the way it draws on the canvas.
            let index =
                i32::try_from(offset + 1).map_err(|_| PsdExportError::TooManyLayers(offset + 1))?;
            let rendered =
                render_entity(frame, &rasterizer, raster_options, visual.entity()).await?;
            validate_pixels(&visual.entity().to_string(), &rendered.image)?;
            layers.push(Layer {
                id: 0,
                name: format!("Shape {index:03} {}", visual.entity()),
                left: rendered.left,
                top: rendered.top,
                pixels: rendered.image,
                hidden: false,
                text: None,
            });
        } else if let Some(text) = match visual.kind() {
            LayerKind::Text(text) if !text.text.trim().is_empty() => Some(text),
            LayerKind::Text(_) | LayerKind::Image(_) | LayerKind::Shape(_) => None,
        } {
            let offset = text_entities
                .iter()
                .position(|(entity, _)| *entity == visual.entity())
                .expect("nonempty visual text was collected before layer projection");
            let index =
                i32::try_from(offset + 1).map_err(|_| PsdExportError::TooManyLayers(offset + 1))?;
            let rendered =
                render_entity(frame, &rasterizer, raster_options, visual.entity()).await?;
            validate_pixels(&visual.entity().to_string(), &rendered.image)?;
            layers.push(Layer {
                id: 0,
                name: format!("TL {index:03} {}", visual.entity()),
                left: rendered.left,
                top: rendered.top,
                pixels: rendered.image,
                hidden: false,
                text: match options.text_layer_mode {
                    TextLayerMode::Rasterized => None,
                    TextLayerMode::Editable => Some(text_metadata(index, text, &font_set)),
                },
            });
        }
    }
    let layer_count = layers.len();
    if layer_count > i16::MAX as usize {
        return Err(PsdExportError::TooManyLayers(layer_count));
    }
    for (offset, layer) in layers.iter_mut().enumerate() {
        layer.id =
            i32::try_from(offset + 1).map_err(|_| PsdExportError::TooManyLayers(layer_count))?;
    }

    Ok(Document {
        width,
        height,
        merged: merged.image,
        layers,
    })
}

async fn rasterize(
    rasterizer: Arc<Rasterizer>,
    frame: &Frame,
    options: RasterOptions,
) -> Result<Raster, PsdExportError> {
    let frame = frame.raster_frame()?;
    tokio_rayon::spawn(move || rasterizer.rasterize(&frame, options))
        .await
        .map_err(PsdExportError::Rasterizer)
}

/// Rasterizes one visual entity alone, clipped to its own bounds.
///
/// `Frame::cropped` keeps the authored presentation out of the pixels so a
/// layer's visibility and opacity stay PSD metadata, and a shape's rotated
/// bounds are already the swept AABB, so a rotated shape is not clipped here.
async fn render_entity(
    frame: &Frame,
    rasterizer: &Arc<Rasterizer>,
    options: RasterOptions,
    entity: EntityId,
) -> Result<Raster, PsdExportError> {
    let cropped = frame
        .cropped(entity)?
        .ok_or(PsdExportError::MissingRenderedEntity(entity))?;
    rasterize(Arc::clone(rasterizer), &cropped, options).await
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), PsdExportError> {
    if width == 0 || height == 0 || width > 30_000 || height > 30_000 {
        return Err(PsdExportError::UnsupportedDimensions { width, height });
    }
    Ok(())
}

async fn read_image(
    snapshot: &Snapshot,
    page: koharu_scene::EntityId,
    role: &'static str,
) -> Result<Option<DynamicImage>, PsdExportError> {
    let Some(asset) = snapshot.asset(page, &AssetRole::new(role)?)? else {
        return Ok(None);
    };
    let bytes = snapshot.read_blob(asset.blob).await?;
    Ok(Some(
        tokio_rayon::spawn(move || image::load_from_memory(&bytes)).await?,
    ))
}

fn combine_masks(
    left: Option<DynamicImage>,
    right: Option<DynamicImage>,
) -> Result<Option<DynamicImage>, PsdExportError> {
    let result = match (left, right) {
        (None, None) => None,
        (Some(mask), None) | (None, Some(mask)) => Some(mask),
        (Some(left), Some(right)) => {
            let mut left = left.into_luma8();
            let right = right.into_luma8();
            if left.dimensions() != right.dimensions() {
                return Err(PsdExportError::MismatchedMaskDimensions {
                    left_width: left.width(),
                    left_height: left.height(),
                    right_width: right.width(),
                    right_height: right.height(),
                });
            }
            for (target, source) in left.pixels_mut().zip(right.pixels()) {
                target.0[0] = target.0[0].max(source.0[0]);
            }
            Some(DynamicImage::ImageLuma8(left))
        }
    };
    Ok(result)
}

fn grayscale_mask_rgba(image: &DynamicImage) -> RgbaImage {
    let mask: GrayImage = image.to_luma8();
    let mut rgba = RgbaImage::new(mask.width(), mask.height());
    for (x, y, pixel) in mask.enumerate_pixels() {
        rgba.put_pixel(x, y, Rgba([pixel[0], pixel[0], pixel[0], 255]));
    }
    rgba
}

fn push_raster_layer(
    layers: &mut Vec<Layer>,
    name: &str,
    pixels: RgbaImage,
    hidden: bool,
) -> Result<(), PsdExportError> {
    validate_pixels(name, &pixels)?;
    layers.push(Layer {
        id: 0,
        name: name.to_owned(),
        left: 0,
        top: 0,
        pixels,
        hidden,
        text: None,
    });
    Ok(())
}

fn validate_pixels(layer: &str, pixels: &RgbaImage) -> Result<(), PsdExportError> {
    let width = pixels.width() as i32;
    let height = pixels.height() as i32;
    if width <= 0 || height <= 0 {
        return Err(PsdExportError::InvalidLayerBounds {
            layer: layer.to_owned(),
            width,
            height,
        });
    }
    Ok(())
}

fn collect_fonts<'a>(texts: impl Iterator<Item = &'a RenderedText>) -> Vec<String> {
    let mut fonts = Vec::new();
    for font in texts.flat_map(|text| &text.post_script_fonts) {
        if !fonts.iter().any(|candidate| candidate == font) {
            fonts.push(font.clone());
        }
    }
    fonts
}

fn text_metadata(index: i32, text: &RenderedText, font_set: &[String]) -> TextMetadata {
    let angle = f64::from(text.angle_degrees).to_radians();
    let bounds = text.layout_bounds;
    let primary_font = text.post_script_fonts.first();
    let font_index = primary_font
        .and_then(|font| font_set.iter().position(|candidate| candidate == font))
        .unwrap_or(0);
    TextMetadata {
        index,
        text: text.text.clone(),
        bounds: [
            f64::from(bounds.x),
            f64::from(bounds.y),
            f64::from(bounds.x + bounds.width),
            f64::from(bounds.y + bounds.height),
        ],
        transform: [
            angle.cos(),
            angle.sin(),
            -angle.sin(),
            angle.cos(),
            f64::from(bounds.x),
            f64::from(bounds.y),
        ],
        orientation: match text.writing_mode {
            WritingMode::Horizontal => TextOrientation::Horizontal,
            WritingMode::VerticalRl => TextOrientation::Vertical,
        },
        justification: match text.alignment {
            TextAlign::Left | TextAlign::Justify => TextJustification::Left,
            TextAlign::Center => TextJustification::Center,
            TextAlign::Right => TextJustification::Right,
        },
        font_index,
        font_set: font_set.to_vec(),
        font_size: f64::from(text.font_size),
        color: text.color,
        box_width: f64::from(bounds.width.max(1.0)),
        box_height: f64::from(bounds.height.max(1.0)),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use koharu_renderer::{RenderBounds, Renderer, TextMetadata as RenderedText};
    use koharu_scene::{
        AssetInput, AssetMetadata, AssetRole, At, Edit, EntityId, Geometry, Origin, PageDraft,
        RasterLayer, RasterLayerKind, Session, ShapeFill, ShapeGeometry, ShapeLayer, ShapeStyle,
    };

    use super::*;

    fn rendered_text(fonts: &[&str]) -> RenderedText {
        RenderedText {
            text: "Hello".to_owned(),
            language: None,
            rendered_bounds: RenderBounds {
                x: 10.0,
                y: 20.0,
                width: 80.0,
                height: 24.0,
            },
            layout_bounds: RenderBounds {
                x: 10.0,
                y: 20.0,
                width: 100.0,
                height: 40.0,
            },
            post_script_fonts: fonts.iter().map(|font| (*font).to_owned()).collect(),
            font_size: 24.0,
            color: [1, 2, 3, 255],
            alignment: TextAlign::Center,
            writing_mode: WritingMode::Horizontal,
            angle_degrees: 0.0,
        }
    }

    #[test]
    fn fonts_keep_first_resolved_order_without_duplicates() {
        let first = rendered_text(&["Primary", "Fallback"]);
        let second = rendered_text(&["Fallback", "Other"]);
        assert_eq!(
            collect_fonts([&first, &second].into_iter()),
            ["Primary", "Fallback", "Other"]
        );
    }

    #[test]
    fn text_metadata_uses_renderer_resolved_presentation() {
        let mut text = rendered_text(&["Primary"]);
        text.writing_mode = WritingMode::VerticalRl;
        text.alignment = TextAlign::Right;
        text.angle_degrees = 90.0;
        let metadata = text_metadata(3, &text, &["Primary".to_owned()]);
        assert_eq!(metadata.orientation, TextOrientation::Vertical);
        assert_eq!(metadata.justification, TextJustification::Right);
        assert_eq!(metadata.font_size, 24.0);
        assert!((metadata.transform[0]).abs() < 1e-12);
        assert!((metadata.transform[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn combine_masks_rejects_different_scene_asset_sizes() {
        let error = combine_masks(
            Some(DynamicImage::ImageLuma8(GrayImage::new(2, 2))),
            Some(DynamicImage::ImageLuma8(GrayImage::new(3, 2))),
        )
        .expect_err("mismatched masks must fail");
        assert!(matches!(
            error,
            PsdExportError::MismatchedMaskDimensions { .. }
        ));
    }

    /// A 1-page scene whose source asset is a solid rectangle of `color`.
    ///
    /// Layers added later with `At::End` sit above the source, so the shape is
    /// the topmost visual layer and must come out first once the visual order is
    /// reversed for the PSD record list.
    async fn scene_with_a_page(width: u32, height: u32, color: [u8; 4]) -> (Session, EntityId) {
        let source = AssetRole::new("source").unwrap();
        let mut session = Session::memory().await.unwrap();
        let mut page = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let created = edit.add_page(
                    PageDraft::new("page", f64::from(width), f64::from(height)),
                    At::End,
                )?;
                edit.set_asset(
                    created,
                    &source,
                    AssetInput::new(
                        png(width, height, color),
                        "image/png",
                        AssetMetadata {
                            width: Some(width),
                            height: Some(height),
                            attributes: BTreeMap::new(),
                        },
                    ),
                )?;
                page = Some(created);
                Ok(())
            })
            .expect("build page patch");
        session.commit(patch).await.unwrap();
        (session, page.unwrap())
    }

    fn png(width: u32, height: u32, color: [u8; 4]) -> Arc<[u8]> {
        let image = RgbaImage::from_pixel(width, height, Rgba(color));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .expect("encode source png");
        bytes.into_inner().into()
    }

    fn add_ellipse(edit: &mut Edit, page: EntityId, name: &str, rect: Geometry) -> EntityId {
        let entity = edit.add_entity(page, At::End).expect("add shape entity");
        edit.set(
            entity,
            &ShapeLayer {
                origin: Origin::User,
                name: name.to_owned(),
                geometry: ShapeGeometry::Ellipse,
                style: ShapeStyle {
                    fill: Some(ShapeFill::Solid([255, 0, 0, 255])),
                    stroke: None,
                },
                angle_degrees: Some(0.0),
                rasterized: false,
            },
        )
        .expect("set shape layer");
        edit.set(entity, &rect).expect("set shape geometry");
        entity
    }

    /// Reads back the `luni` names of a serialized PSD in layer-record order.
    ///
    /// Scanning the whole file for the `luni` key would also match the byte
    /// sequence inside compressed channel data, so this walks the real section
    /// layout and only looks inside each layer's additional-info block.
    fn psd_layer_names(bytes: &[u8]) -> Vec<String> {
        let be_u32 = |offset: usize| {
            u32::from_be_bytes(bytes[offset..offset + 4].try_into().expect("be u32")) as usize
        };
        let mut extra = Vec::new();
        let mut offset = 26; // File header.
        offset += 4 + be_u32(offset); // Color mode data.
        offset += 4 + be_u32(offset); // Image resources.
        offset += 4; // Layer and mask info length.
        offset += 4; // Layer info length; the records start at the layer count.
        let count = i16::from_be_bytes(bytes[offset..offset + 2].try_into().expect("layer count"));
        offset += 2;
        for _ in 0..count.unsigned_abs() {
            offset += 16; // Bounds.
            let channels =
                u16::from_be_bytes(bytes[offset..offset + 2].try_into().expect("channels"))
                    as usize;
            offset += 2 + channels * 6;
            offset += 12; // Blend signature, key, opacity, clipping, flags, filler.
            let length = be_u32(offset);
            offset += 4;
            extra.push((offset, offset + length));
            offset += length;
        }
        let mut names = Vec::new();
        for (start, end) in extra {
            // Layer mask data, blending ranges, then the Pascal-string name whose
            // length byte and characters together are padded to a multiple of 4.
            let name_length = bytes[start + 8] as usize;
            let mut cursor = start + 8 + (name_length + 1).div_ceil(4) * 4;
            while cursor + 12 <= end {
                let key = &bytes[cursor + 4..cursor + 8];
                let length = be_u32(cursor + 8);
                let body = &bytes[cursor + 12..cursor + 12 + length];
                if key == b"luni" {
                    let units =
                        u32::from_be_bytes(body[..4].try_into().expect("luni count")) as usize;
                    names.push(String::from_utf16_lossy(
                        &(0..units)
                            .map(|index| {
                                u16::from_be_bytes(
                                    body[4 + index * 2..6 + index * 2]
                                        .try_into()
                                        .expect("luni unit"),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ));
                }
                cursor += 12 + length;
            }
        }
        names
    }

    #[tokio::test]
    async fn a_shape_is_exported_as_a_raster_layer_in_visual_order() {
        let (mut session, page) = scene_with_a_page(64, 64, [255, 255, 255, 255]).await;
        // A cleanup raster layer under the shape pins the emission order: the
        // PSD list is the reverse of the visual list, so the shape must come
        // out before (i.e. above) the cleanup layer, not appended last.
        let patch = session
            .snapshot()
            .patch(|edit| {
                let cleanup = edit.add_entity(page, At::End)?;
                edit.set(
                    cleanup,
                    &RasterLayer {
                        origin: Origin::User,
                        name: "Cleanup".to_owned(),
                        kind: RasterLayerKind::Cleanup,
                    },
                )?;
                edit.set_asset(
                    cleanup,
                    &AssetRole::new("source")?,
                    AssetInput::new(
                        png(64, 64, [0, 0, 0, 0]),
                        "image/png",
                        AssetMetadata {
                            width: Some(64),
                            height: Some(64),
                            attributes: BTreeMap::new(),
                        },
                    ),
                )?;
                add_ellipse(
                    edit,
                    page,
                    "Ellipse 1",
                    Geometry::rectangle(8.0, 8.0, 48.0, 32.0),
                );
                Ok(())
            })
            .expect("build shape patch");
        let snapshot = session.commit(patch).await.unwrap().snapshot;

        let frame = Renderer::default().render(&snapshot, page).await.unwrap();
        let rasterizer = Arc::new(Rasterizer::new().expect("a WGPU adapter is required"));
        let document = build(
            &snapshot,
            &frame,
            rasterizer,
            RasterOptions::default(),
            &PsdExportOptions::default(),
        )
        .await
        .expect("shape layers must not break export");

        // The rendered pixels of the ellipse, not a re-derivation from geometry.
        let shape_layer = document
            .layers
            .iter()
            .find(|layer| layer.name.starts_with("Shape"))
            .expect("a shape layer must be emitted");
        assert!(
            shape_layer.pixels.pixels().any(|pixel| pixel[3] > 0),
            "the shape layer must contain rendered pixels"
        );
        assert!(
            shape_layer.text.is_none(),
            "a shape is not an editable text layer"
        );
        let cleanup = document
            .layers
            .iter()
            .position(|layer| layer.name == "Cleanup")
            .expect("the cleanup layer must be emitted");
        let shape_index = document
            .layers
            .iter()
            .position(|layer| layer.name.starts_with("Shape"))
            .unwrap();
        assert!(
            shape_index < cleanup,
            "the topmost visual (shape) must precede the layer below it in the reversed PSD list"
        );
        // `Frame::cropped` must not clip the shape; the ellipse spans the whole
        // 48x32 rect, so a clipped crop would come out visibly smaller.
        assert!(
            shape_layer.pixels.width() >= 48 && shape_layer.pixels.height() >= 32,
            "a rotated-shape bounds regression would shrink the crop, was {}x{}",
            shape_layer.pixels.width(),
            shape_layer.pixels.height()
        );
        assert!(document.layers.iter().all(|layer| layer.id > 0));
    }

    /// A pattern-filled shape must export the **pattern**, not its fallback.
    ///
    /// Shapes are rasterized into the PSD, so the pattern reaches the pixels
    /// through the same `shape_renderer` path as everything else — but only if
    /// the layer's `pattern` asset was resolved and attached before drawing.
    /// `PatternSpec` references no asset (P-D2), so if the renderer's attachment
    /// were dropped the export would show a flat fallback colour instead, with
    /// every other PSD test still passing.
    #[tokio::test]
    async fn a_pattern_filled_shape_exports_its_pattern() {
        let (mut session, page) = scene_with_a_page(64, 64, [255, 255, 255, 255]).await;
        let fallback = [0, 0, 255, 255];
        let patch = session
            .snapshot()
            .patch(|edit| {
                let entity = add_ellipse(
                    edit,
                    page,
                    "Pattern 1",
                    Geometry::rectangle(8.0, 8.0, 48.0, 48.0),
                );
                let mut layer = ShapeLayer {
                    origin: Origin::User,
                    name: "Pattern 1".to_owned(),
                    geometry: ShapeGeometry::Ellipse,
                    style: ShapeStyle {
                        fill: Some(ShapeFill::Pattern(koharu_scene::PatternSpec {
                            repeat: koharu_scene::PatternRepeat::Tile,
                            origin: [0.0, 0.0],
                            scale: 0.25,
                            rotation_degrees: 0,
                            fallback,
                        })),
                        stroke: None,
                    },
                    angle_degrees: Some(0.0),
                    rasterized: false,
                };
                // The nearest possible image: one opaque red pixel, so every
                // tile is red and any correct render is red inside the ellipse.
                let mut encoded = std::io::Cursor::new(Vec::new());
                RgbaImage::from_pixel(1, 1, Rgba([255, 0, 0, 255]))
                    .write_to(&mut encoded, image::ImageFormat::Png)
                    .expect("encode the pattern png");
                edit.set_asset(
                    entity,
                    &AssetRole::new("pattern")?,
                    AssetInput::new(
                        Arc::<[u8]>::from(encoded.into_inner()),
                        "image/png",
                        AssetMetadata {
                            width: Some(1),
                            height: Some(1),
                            attributes: BTreeMap::new(),
                        },
                    ),
                )?;
                layer.rasterized = false;
                edit.set(entity, &layer)?;
                Ok(())
            })
            .expect("build the pattern patch");
        let snapshot = session.commit(patch).await.unwrap().snapshot;

        let frame = Renderer::default().render(&snapshot, page).await.unwrap();
        let rasterizer = Arc::new(Rasterizer::new().expect("a WGPU adapter is required"));
        let document = build(
            &snapshot,
            &frame,
            rasterizer,
            RasterOptions::default(),
            &PsdExportOptions::default(),
        )
        .await
        .expect("a pattern fill must not break export");

        let shape_layer = document
            .layers
            .iter()
            .find(|layer| layer.name.starts_with("Shape"))
            .expect("the shape layer must be emitted");
        // Red-dominant pixels prove the pattern drew; a blue fallback would be
        // blue-dominant, and an empty crop would have no pixels at all.
        let mut red = 0usize;
        let mut blue = 0usize;
        for pixel in shape_layer.pixels.pixels() {
            if pixel[3] == 0 {
                continue;
            }
            if pixel[0] > pixel[2] {
                red += 1;
            } else if pixel[2] > pixel[0] {
                blue += 1;
            }
        }
        assert!(red > 0, "the exported shape must contain pattern pixels");
        assert!(
            blue == 0,
            "the fallback colour leaked into the export ({blue} blue pixels, {red} red)"
        );
    }

    #[tokio::test]
    async fn a_rotated_shape_is_not_clipped_by_its_crop() {        // Task 4 made `Layer::bounds` the AABB of the rotated frame box, which
        // is what `Frame::cropped` clips to. This pins that result for export: a
        // 45°-rotated 48x32 rect must crop at least as tall as its swept height
        // (48/√2 + 32/√2 ≈ 56.6), not the unrotated 32.
        let (mut session, page) = scene_with_a_page(128, 128, [0, 0, 0, 0]).await;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let entity = edit.add_entity(page, At::End)?;
                edit.set(
                    entity,
                    &ShapeLayer {
                        origin: Origin::User,
                        name: "Ellipse 1".to_owned(),
                        geometry: ShapeGeometry::Ellipse,
                        style: ShapeStyle {
                            fill: Some(ShapeFill::Solid([255, 0, 0, 255])),
                            stroke: None,
                        },
                        angle_degrees: Some(45.0),
                        rasterized: false,
                    },
                )?;
                edit.set(entity, &rotated_rectangle(64.0, 64.0, 48.0, 32.0, 45.0))?;
                Ok(())
            })
            .expect("build rotated shape patch");
        let snapshot = session.commit(patch).await.unwrap().snapshot;

        let frame = Renderer::default().render(&snapshot, page).await.unwrap();
        let document = build(
            &snapshot,
            &frame,
            Arc::new(Rasterizer::new().expect("a WGPU adapter is required")),
            RasterOptions::default(),
            &PsdExportOptions::default(),
        )
        .await
        .expect("a rotated shape must export");

        let shape = document
            .layers
            .iter()
            .find(|layer| layer.name.starts_with("Shape"))
            .expect("a shape layer must be emitted");
        let swept = (48.0_f64 + 32.0) / 2.0_f64.sqrt();
        assert!(
            f64::from(shape.pixels.height()) >= swept - 1.0,
            "a rotated shape's crop must span its swept AABB ({swept:.1}), was {}",
            shape.pixels.height()
        );
    }

    /// The corners of a `width`x`height` rectangle centred at `(cx, cy)`, rotated
    /// by `angle` degrees about that centre.
    fn rotated_rectangle(cx: f64, cy: f64, width: f64, height: f64, angle: f64) -> Geometry {
        let (sin, cos) = angle.to_radians().sin_cos();
        let (half_width, half_height) = (width * 0.5, height * 0.5);
        Geometry {
            origin: Origin::User,
            points: [
                (-half_width, -half_height),
                (half_width, -half_height),
                (half_width, half_height),
                (-half_width, half_height),
            ]
            .into_iter()
            .map(|(x, y)| koharu_scene::Point {
                x: cx + x * cos - y * sin,
                y: cy + x * sin + y * cos,
            })
            .collect(),
        }
    }

    /// A horizontal arrow's drag box is a hair tall while its head is
    /// `width` tall. `Layer::bounds` feeds `Frame::cropped`, so bounds that
    /// counted only the frame box would clip roughly half the head off.
    #[tokio::test]
    async fn a_horizontal_arrow_is_not_clipped_by_its_crop() {
        const HEAD_WIDTH: f32 = 12.0;
        let (mut session, page) = scene_with_a_page(128, 128, [0, 0, 0, 0]).await;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let entity = edit.add_entity(page, At::End)?;
                edit.set(
                    entity,
                    &ShapeLayer {
                        origin: Origin::User,
                        name: "Line 1".to_owned(),
                        geometry: ShapeGeometry::Line {
                            start: [16.0, 64.0],
                            end: [112.0, 64.0],
                            head: Some(koharu_scene::ArrowHead {
                                length: 18.0,
                                width: HEAD_WIDTH,
                            }),
                        },
                        style: ShapeStyle {
                            fill: None,
                            stroke: Some(koharu_scene::StrokeStyle {
                                color: [0, 0, 0, 255],
                                width: 2.0,
                                dash: None,
                                cap: koharu_scene::LineCap::Butt,
                                join: koharu_scene::LineJoin::Miter,
                            }),
                        },
                        angle_degrees: Some(0.0),
                        rasterized: false,
                    },
                )?;
                // The polygon is the hit-testing sample: a nearly flat drag box.
                // The frontend clamps a zero-height frame to `1e-6` so the
                // polygon stays drawable, which is what a horizontal drag
                // really produces (`lib/shapes.ts` `MIN_FRAME_EXTENT`).
                edit.set(
                    entity,
                    &Geometry {
                        origin: Origin::User,
                        points: vec![
                            koharu_scene::Point { x: 16.0, y: 64.0 },
                            koharu_scene::Point { x: 64.0, y: 64.0 },
                            koharu_scene::Point {
                                x: 112.0,
                                y: 64.0 + 1e-6,
                            },
                        ],
                    },
                )?;
                Ok(())
            })
            .expect("build arrow patch");
        let snapshot = session.commit(patch).await.unwrap().snapshot;

        let frame = Renderer::default().render(&snapshot, page).await.unwrap();
        let document = build(
            &snapshot,
            &frame,
            Arc::new(Rasterizer::new().expect("a WGPU adapter is required")),
            RasterOptions::default(),
            &PsdExportOptions::default(),
        )
        .await
        .expect("an arrow must export");

        let arrow = document
            .layers
            .iter()
            .find(|layer| layer.name.starts_with("Shape"))
            .expect("a shape layer must be emitted");
        assert!(
            f64::from(arrow.pixels.height()) >= f64::from(HEAD_WIDTH),
            "an arrow's crop must span its head width ({HEAD_WIDTH}), was {}x{}",
            arrow.pixels.width(),
            arrow.pixels.height()
        );
    }

    #[tokio::test]
    async fn a_serialized_shape_layer_carries_a_readable_name() {
        let (mut session, page) = scene_with_a_page(64, 64, [0, 0, 0, 0]).await;
        let mut shape = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                shape = Some(add_ellipse(
                    edit,
                    page,
                    "Ellipse 1",
                    Geometry::rectangle(8.0, 8.0, 48.0, 32.0),
                ));
                Ok(())
            })
            .expect("build shape patch");
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let shape = shape.unwrap();

        let frame = Renderer::default().render(&snapshot, page).await.unwrap();
        let bytes = crate::export::export_page(
            Arc::new(Rasterizer::new().expect("a WGPU adapter is required")),
            &snapshot,
            &frame,
            &PsdExportOptions::default(),
        )
        .await
        .expect("PSD export must succeed with a shape layer");

        let names = psd_layer_names(&bytes);
        let expected = format!("Shape 001 {shape}");
        assert!(
            names.iter().any(|name| name == &expected),
            "expected {expected:?} among {names:?}"
        );
    }
}
