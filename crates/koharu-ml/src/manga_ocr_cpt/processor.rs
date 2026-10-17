//! Manga OCR CPT-flex image and text processing.
//!
//! Port of the `fit="aspect"` path of the upstream CPU preprocessing and of
//! `MangaOcrCPT._text`:
//! https://github.com/muscgab/manga-ocr-CPT/blob/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt/preprocess.py
//! https://github.com/muscgab/manga-ocr-CPT/blob/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt/__init__.py

use std::{fs, path::Path};

use anyhow::{Result, ensure};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, GrayImage, Luma};
use koharu_torch::{Device, Kind, Tensor};
use serde::Deserialize;

use super::config::{MAX_PATCHES, PATCH_FEATURES, PATCH_SIZE, SPECIAL_TOKENS};

#[derive(Debug)]
pub(super) struct ImageProcessor {
    device: Device,
}

/// One preprocessed crop.
#[derive(Debug)]
pub(super) struct ImageInput {
    /// Patch features `(1, MAX_PATCHES, PATCH_FEATURES)` in `[-1, 1]`, zero
    /// after the valid patches.
    pub(super) patches: Tensor,
    /// Valid-patch mask `(1, MAX_PATCHES)`.
    pub(super) valid: Tensor,
    /// Patch grid `(height, width)` of the valid patches.
    pub(super) grid: [i64; 2],
}

impl ImageProcessor {
    pub(super) fn new(device: Device) -> Self {
        Self { device }
    }

    pub(super) fn preprocess(&self, image: &DynamicImage) -> Result<ImageInput> {
        if image.width() == 0 || image.height() == 0 {
            anyhow::bail!("cannot recognize an empty image");
        }
        let canvas = canvas(&grey(image))?;
        let [grid_h, grid_w] = canvas.grid;
        let count = grid_h * grid_w;
        // (gh*16, gw*16) -> (gh*gw, channel, row, column), the upstream
        // `transpose(0, 2, 4, 1, 3)` order; the three channels are equal.
        let patches = Tensor::from_slice(canvas.image.as_raw())
            .to_device(self.device)
            .to_kind(Kind::Float)
            .view([grid_h, PATCH_SIZE, grid_w, PATCH_SIZE])
            .permute([0, 2, 1, 3])
            .reshape([count, 1, PATCH_SIZE * PATCH_SIZE])
            .expand([count, 3, PATCH_SIZE * PATCH_SIZE], false)
            .reshape([count, PATCH_FEATURES]);
        let patches = (patches / 255.0 - 0.5) / 0.5;
        let padding = Tensor::zeros(
            [MAX_PATCHES - count, PATCH_FEATURES],
            (Kind::Float, self.device),
        );
        let patches = Tensor::cat(&[patches, padding], 0).unsqueeze(0);
        let valid = Tensor::arange(MAX_PATCHES, (Kind::Int64, self.device))
            .lt(count)
            .unsqueeze(0);
        Ok(ImageInput {
            patches,
            valid,
            grid: [grid_h, grid_w],
        })
    }
}

/// Pillow `Image.convert("L")` from RGB: ITU-R 601-2 luma in 16-bit fixed
/// point. The `image` crate's luma uses Rec. 709 weights and would not match
/// the training input.
fn luma([red, green, blue]: [u8; 3]) -> u8 {
    ((u32::from(red) * 19595 + u32::from(green) * 38470 + u32::from(blue) * 7471 + 0x8000) >> 16)
        as u8
}

/// Upstream `_rgb` drops alpha without compositing, as `to_rgb8` does.
fn grey(image: &DynamicImage) -> GrayImage {
    let rgb = image.to_rgb8();
    GrayImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        Luma([luma(rgb.get_pixel(x, y).0)])
    })
}

/// Upstream `grid`: the largest uniform scale whose rounded-up patch grid
/// stays within the budget, found by 60 bisection steps over `[1e-4, 1000]`.
fn grid(height: u32, width: u32) -> (f64, i64, i64) {
    let cells = |size: u32, scale: f64| (f64::from(size) * scale / PATCH_SIZE as f64).ceil() as i64;
    let (mut low, mut high) = (1e-4, 1000.0);
    for _ in 0..60 {
        let middle = (low + high) / 2.0;
        if cells(height, middle) * cells(width, middle) <= MAX_PATCHES {
            low = middle;
        } else {
            high = middle;
        }
    }
    (low, cells(height, low), cells(width, low))
}

/// Resized side inside the grid canvas: `max(1, min(cells * 16, round(size * scale)))`.
/// Python's `round` is round-half-to-even.
fn resized_extent(size: u32, scale: f64, cells: i64) -> u32 {
    (f64::from(size) * scale)
        .round_ties_even()
        .clamp(1.0, (cells * PATCH_SIZE) as f64) as u32
}

/// `np.median(border).astype(uint8)` over the first and last row and column,
/// corners counted twice. NumPy averages the two middle values of an even
/// count in floating point and `astype` truncates.
fn border_median(image: &GrayImage) -> u8 {
    let (width, height) = image.dimensions();
    let mut border = Vec::with_capacity(2 * (width + height) as usize);
    border.extend((0..width).map(|x| image.get_pixel(x, 0)[0]));
    border.extend((0..width).map(|x| image.get_pixel(x, height - 1)[0]));
    border.extend((0..height).map(|y| image.get_pixel(0, y)[0]));
    border.extend((0..height).map(|y| image.get_pixel(width - 1, y)[0]));
    border.sort_unstable();
    let middle = border.len() / 2;
    if border.len() % 2 == 1 {
        border[middle]
    } else {
        ((f64::from(border[middle - 1]) + f64::from(border[middle])) / 2.0) as u8
    }
}

/// The resized crop at the top left of a whole-patch canvas filled with its
/// border median.
#[derive(Debug)]
struct Canvas {
    image: GrayImage,
    grid: [i64; 2],
}

fn canvas(grey: &GrayImage) -> Result<Canvas> {
    let (scale, grid_h, grid_w) = grid(grey.height(), grey.width());
    let width = resized_extent(grey.width(), scale, grid_w);
    let height = resized_extent(grey.height(), scale, grid_h);
    let small = resize_bilinear(grey, width, height)?;
    let mut image = GrayImage::from_pixel(
        (grid_w * PATCH_SIZE) as u32,
        (grid_h * PATCH_SIZE) as u32,
        Luma([border_median(&small)]),
    );
    image::imageops::replace(&mut image, &small, 0, 0);
    Ok(Canvas {
        image,
        grid: [grid_h, grid_w],
    })
}

/// Upstream resizes with Pillow `BILINEAR`; `fast_image_resize` uses the same
/// triangle convolution but can differ at coefficient rounding.
fn resize_bilinear(image: &GrayImage, width: u32, height: u32) -> Result<GrayImage> {
    if image.dimensions() == (width, height) {
        return Ok(image.clone());
    }
    let mut resized = GrayImage::new(width, height);
    Resizer::new()
        .resize(
            image,
            &mut resized,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        )
        .map_err(|error| anyhow::anyhow!("failed to resize Manga OCR CPT-flex input: {error}"))?;
    Ok(resized)
}

#[derive(Debug, Deserialize)]
struct VocabularyFile {
    character_count: usize,
    characters: Vec<String>,
}

#[derive(Debug)]
pub(super) struct Vocabulary {
    characters: Vec<String>,
}

impl Vocabulary {
    pub(super) fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let file: VocabularyFile = serde_json::from_str(&fs::read_to_string(path)?)?;
        ensure!(
            file.characters.len() == file.character_count,
            "vocabulary lists {} characters but declares {}",
            file.characters.len(),
            file.character_count
        );
        Ok(Self {
            characters: file.characters,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.characters.len()
    }

    /// Upstream `_text` without the end cut, which generation already makes:
    /// ids below `SPECIAL_TOKENS` (pad, unk, start, end, mask) are dropped and
    /// character `i` is id `SPECIAL_TOKENS + i`.
    pub(super) fn decode(&self, ids: &[i64]) -> String {
        ids.iter()
            .filter_map(|&id| usize::try_from(id - SPECIAL_TOKENS).ok())
            .filter_map(|index| self.characters.get(index))
            .map(String::as_str)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use image::{GrayImage, Luma};

    use super::*;

    #[test]
    fn luma_matches_pillow_convert_l() {
        assert_eq!(luma([255, 255, 255]), 255);
        assert_eq!(luma([0, 0, 0]), 0);
        assert_eq!(luma([255, 0, 0]), 76);
        assert_eq!(luma([0, 255, 0]), 150);
        assert_eq!(luma([0, 0, 255]), 29);
        assert_eq!(luma([10, 200, 90]), 131);
    }

    #[test]
    fn grid_matches_the_upstream_binary_search() {
        // (height, width) -> (grid h, grid w, resized width, resized height),
        // all taken from the upstream `grid` and resize extents.
        for ((height, width), (cells_h, cells_w, resized_w, resized_h)) in [
            ((1024, 1024), (14, 14, 224, 224)),
            ((120, 900), (5, 38, 600, 80)),
            ((900, 120), (38, 5, 80, 600)),
            ((4, 4), (14, 14, 224, 224)),
            ((30, 2000), (2, 98, 1568, 24)),
        ] {
            let (scale, grid_h, grid_w) = grid(height, width);
            assert_eq!((grid_h, grid_w), (cells_h, cells_w), "{height}x{width}");
            assert!(grid_h * grid_w <= MAX_PATCHES);
            assert_eq!(
                resized_extent(width, scale, grid_w),
                resized_w,
                "{height}x{width}"
            );
            assert_eq!(
                resized_extent(height, scale, grid_h),
                resized_h,
                "{height}x{width}"
            );
        }
    }

    #[test]
    fn resized_extent_rounds_half_to_even_like_python() {
        assert_eq!(resized_extent(5, 0.5, 14), 2);
        assert_eq!(resized_extent(7, 0.5, 14), 4);
        assert_eq!(resized_extent(1, 0.01, 14), 1);
        assert_eq!(resized_extent(1000, 1.0, 2), 32);
    }

    #[test]
    fn border_median_averages_then_truncates_like_numpy() {
        // A 2x2 image has border [1, 2, 3, 4, 1, 3, 2, 4] (corners twice).
        let image = GrayImage::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(border_median(&image), 2);
        let image = GrayImage::from_raw(2, 1, vec![10, 11]).unwrap();
        assert_eq!(border_median(&image), 10);
        assert_eq!(border_median(&GrayImage::from_pixel(3, 3, Luma([7]))), 7);
    }

    #[test]
    fn the_fixture_canvas_matches_upstream() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/fixtures/ocr/title.png");
        let canvas = canvas(&grey(&image::open(path).unwrap())).unwrap();
        assert_eq!(canvas.grid, [6, 32]);
        assert_eq!(canvas.image.dimensions(), (512, 96));
        // Right of the 507-wide resize is the median fill.
        assert_eq!(canvas.image.get_pixel(511, 0)[0], 255);
        // Pillow sums the canvas to 10,422,116; `fast_image_resize` may round a
        // few coefficients differently, so allow one level per pixel on average.
        let sum: u64 = canvas
            .image
            .as_raw()
            .iter()
            .map(|&value| u64::from(value))
            .sum();
        assert!(sum.abs_diff(10_422_116) <= 512 * 96, "sum {sum}");
    }

    #[test]
    fn decoding_skips_special_tokens_and_offsets_characters() {
        let vocabulary = Vocabulary {
            characters: vec!["a".into(), "b".into(), "c".into()],
        };
        assert_eq!(vocabulary.decode(&[5, 0, 7, 1, 4, 6]), "acb");
        assert_eq!(vocabulary.decode(&[]), "");
    }
}
