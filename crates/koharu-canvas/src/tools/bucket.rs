//! The paint bucket: a click fills the selected raster's pixels joined to the
//! one under it, across or down, that are exactly its colour, with the brush
//! colour, as Photoshop's Paint Bucket samples the current layer at a
//! tolerance of 0 with Contiguous on and Anti-alias off.

use image::{Rgba as Pixel, RgbaImage};
use koharu_scene::{Edit, Op, PixelFormat, Rect, Ref};

use crate::{Effect, Notice, gesture::Context, view::Sample};

pub(crate) fn fill(cx: &Context, sample: Sample) -> anyhow::Result<Vec<Effect>> {
    let (page, size) = (sample.page, cx.snapshot.size());
    if !(0.0..size.width).contains(&page.x) || !(0.0..size.height).contains(&page.y) {
        return Ok(Vec::new());
    }
    let target = cx
        .selection
        .single()
        .and_then(|layer| Some((layer, cx.snapshot.writable(layer)?)))
        .filter(|(_, (_, format))| *format == PixelFormat::Rgba8);
    let Some((layer, (offset, _))) = target else {
        return Ok(vec![Effect::Notice(Notice::FillNeedsRaster)]);
    };
    // The page, in the raster's own pixels: a fill stops at the page's edge.
    let (width, height) = (size.width as u32, size.height as u32);
    let own = Rect {
        x: -offset.x,
        y: -offset.y,
        width,
        height,
    };
    let pixels = cx.session.lock().pixels(layer, own)?;
    let mut image = RgbaImage::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow::anyhow!("a raster's pixels are RGBA"))?;
    let color = Pixel(
        cx.brush
            .color
            .0
            .map(|channel| (channel * 255.0).round() as u8),
    );
    let (x, y) = (page.x as u32, page.y as u32);
    // Already that colour, the fill would change nothing.
    if *image.get_pixel(x, y) == color {
        return Ok(Vec::new());
    }
    imageproc::drawing::flood_fill_mut(&mut image, x, y, color);
    Ok(vec![Effect::Edit(Edit {
        label: "fill".to_owned(),
        ops: vec![Op::WritePixels {
            layer: Ref::from(layer),
            rect: own,
            pixels: image.into_raw(),
        }],
    })])
}
