//! Pixels and imported files, whose bytes live in the store by content hash,
//! out of undo's reach, so undo steps stay small.

use std::io::Cursor;

use image::{
    RgbaImage,
    imageops::colorops::{brighten_in_place, contrast_in_place, huerotate_in_place},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use specta::Type;

use super::Run;
use crate::{Error, Hash, LayerId, PixelFormat, Rect, Ref, Result, Size, project::Picture, tiles};

/// A colour adjustment of a picture's pixels, as Photoshop's Image >
/// Adjustments. Each keeps alpha, and a clear pixel its colour too, since its
/// colour shows if its alpha is raised later.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Type, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ColorAdjust {
    /// `delta`, in -255..=255, added to each colour channel.
    Brightness { delta: i32 },
    /// The distance from mid-grey scaled by `percent`, in -100..=100; 0
    /// changes nothing and 100 doubles it.
    Contrast { percent: f32 },
    /// The channels from `low` to `high` stretched onto 0..=255.
    Levels { low: u8, high: u8 },
    /// The hue turned by `degrees`, in -180..=180.
    Hue { degrees: i32 },
    /// Rec. 709 luminance in each colour channel.
    Grayscale,
}

impl ColorAdjust {
    fn invalid(self) -> Option<&'static str> {
        match self {
            Self::Brightness { delta } if !(-255..=255).contains(&delta) => {
                Some("brightness is -255 to 255")
            }
            Self::Contrast { percent } if !(-100.0..=100.0).contains(&percent) => {
                Some("contrast is -100 to 100 percent")
            }
            Self::Levels { low, high } if low >= high => Some("the low level is below the high"),
            Self::Hue { degrees } if !(-180..=180).contains(&degrees) => {
                Some("hue is -180 to 180 degrees")
            }
            _ => None,
        }
    }

    /// Adjusts a tile's RGBA pixels.
    fn apply(self, raw: &mut Vec<u8>) {
        let side = tiles::SIZE as u32;
        let before = raw.clone();
        let mut image =
            RgbaImage::from_raw(side, side, std::mem::take(raw)).expect("a tile is a tile");
        match self {
            Self::Brightness { delta } => brighten_in_place(&mut image, delta),
            Self::Contrast { percent } => contrast_in_place(&mut image, percent),
            Self::Levels { low, high } => {
                let (low, span) = (f32::from(low), f32::from(high - low));
                for pixel in image.pixels_mut() {
                    for channel in &mut pixel.0[..3] {
                        let value = (f32::from(*channel) - low) * 255.0 / span;
                        *channel = value.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
            Self::Hue { degrees } => huerotate_in_place(&mut image, degrees),
            Self::Grayscale => {
                for pixel in image.pixels_mut() {
                    let [red, green, blue, _] = pixel.0.map(f32::from);
                    let luma = (0.2126 * red + 0.7152 * green + 0.0722 * blue).round() as u8;
                    pixel.0[..3].fill(luma);
                }
            }
        }
        *raw = image.into_raw();
        // `contrast_in_place` scales alpha with the colour; alpha goes back.
        let originals = before.as_chunks::<4>().0;
        for (pixel, original) in raw.as_chunks_mut::<4>().0.iter_mut().zip(originals) {
            match original[3] {
                0 => *pixel = *original,
                alpha => pixel[3] = alpha,
            }
        }
    }
}

impl Run<'_, '_> {
    /// The picture `layer` names, which must be unlocked to change.
    fn picture(&self, op: usize, layer: &Ref) -> Result<(LayerId, Picture)> {
        let id = LayerId(self.resolve(layer)?);
        let Some(picture) = self.project.picture(id) else {
            return Err(match self.project.index.contains_key(&id) {
                true => Error::invalid(op, format!("layer {id} has no pixels")),
                false => Error::NotFound(format!("layer {id}")),
            });
        };
        if picture.locked {
            return Err(Error::Locked { op, layer: id });
        }
        self.unlocked(op, picture.holder)?;
        Ok((id, picture))
    }

    pub(super) fn adjust_colors(
        &mut self,
        op: usize,
        layer: &Ref,
        adjust: ColorAdjust,
    ) -> Result<()> {
        if let Some(reason) = adjust.invalid() {
            return Err(Error::invalid(op, reason));
        }
        let (id, picture) = self.picture(op, layer)?;
        if picture.format != PixelFormat::Rgba8 {
            return Err(Error::invalid(
                op,
                "only a colour picture takes a colour adjustment",
            ));
        }
        let Some(mut map) = self.project.tiles.get(&id).cloned() else {
            return Ok(());
        };
        let current = map
            .iter()
            .map(|((x, y), hash)| Ok((*x, *y, self.tx.blob(hash)?)))
            .collect::<Result<Vec<_>>>()?;
        for (x, y, bytes) in tiles::map(&current, picture.format, |raw| adjust.apply(raw))? {
            match bytes {
                Some(bytes) => {
                    map.insert((x, y), self.tx.put_blob(&bytes)?);
                }
                None => {
                    map.remove(&(x, y));
                }
            }
        }
        self.project.tiles.insert(id, map);
        Ok(())
    }

    pub(super) fn write_pixels(
        &mut self,
        op: usize,
        layer: &Ref,
        rect: Rect,
        pixels: &[u8],
    ) -> Result<()> {
        let (id, picture) = self.picture(op, layer)?;
        let format = picture.format;
        let expected = rect.width as usize * rect.height as usize * tiles::bytes_per_pixel(format);
        if pixels.len() != expected {
            return Err(Error::invalid(
                op,
                format!(
                    "{} bytes of pixels, but the rectangle needs {expected}",
                    pixels.len()
                ),
            ));
        }
        let end = |start: i32, length: u32| i64::from(start) + i64::from(length);
        if end(rect.x, rect.width) > 1 << 31 || end(rect.y, rect.height) > 1 << 31 {
            return Err(Error::invalid(op, "the rectangle runs out of range"));
        }
        let Some(((x0, x1), (y0, y1))) = tiles::covered(rect) else {
            return Ok(());
        };
        let mut map = self.project.tiles.get(&id).cloned().unwrap_or_default();
        let mut current = Vec::new();
        for ((x, y), hash) in map.iter() {
            if (x0..=x1).contains(&i64::from(*x)) && (y0..=y1).contains(&i64::from(*y)) {
                current.push((*x, *y, self.tx.blob(hash)?));
            }
        }
        for (x, y, bytes) in tiles::write(&current, format, rect, pixels)? {
            match bytes {
                Some(bytes) => {
                    map.insert((x, y), self.tx.put_blob(&bytes)?);
                }
                None => {
                    map.remove(&(x, y));
                }
            }
        }
        match map.is_empty() {
            true => self.project.tiles.remove(&id),
            false => self.project.tiles.insert(id, map),
        };
        Ok(())
    }

    pub(super) fn add_file(&mut self, bytes: &[u8]) -> Result<()> {
        self.tx.put_blob(bytes)?;
        Ok(())
    }

    /// The size of an added image, which a page made from it takes.
    pub(super) fn image_size(&self, op: usize, file: &Hash) -> Result<Size> {
        let bytes = match self.tx.blob(file) {
            Err(Error::NotFound(_)) => {
                return Err(Error::invalid(op, format!("file {file} was not added")));
            }
            bytes => bytes?,
        };
        let (width, height) = image::ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|error| Error::invalid(op, error.to_string()))?
            .into_dimensions()
            .map_err(|error| Error::invalid(op, format!("file {file} isn't an image: {error}")))?;
        Ok(Size { width, height })
    }
}
