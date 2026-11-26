//! Seamless Manga Inpainting with Semantics Awareness, run as IOPaint's
//! `Manga` model runs it: MangaLineExtraction draws the page's lines, and the
//! inpaintor fills the mask from the gray page, the lines, and the
//! screentone it reads from them. The result is gray.
//!
//! Upstream:
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/manga.py
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/base.py
//!
//! Both networks are published only as IOPaint's TorchScript traces, which
//! libtorch's C API here can't run; their weights are read out of the traces
//! once, under the traces' own names, into safetensors beside them.

mod erika;
mod inpaintor;

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, ensure};
use image::{GrayImage, RgbImage, imageops};
use koharu_runtime::{Asset, Library, Turn};
use koharu_torch::{Device, Generator, Kind, Tensor};

use self::{erika::Erika, inpaintor::Inpaintor};
use crate::nn;

const INPAINTOR: Asset = Asset::new(
    "https://github.com/Sanster/models/releases/download/manga/manga_inpaintor.jit",
    "dc1622dc9d96387b18f6a6391ce1a8ed10be05d99e6c01e3ca2ddb8c9a3c592b",
);
const LINES: Asset = Asset::new(
    "https://github.com/Sanster/models/releases/download/manga/erika.jit",
    "33f214b1c4010c2f7d0fc82d4e9dde0185c8fd754419b6332946d34038b1c173",
);

/// `hd_strategy_crop_trigger_size`: past it a page goes a crop around each
/// masked region at a time, with `hd_strategy_crop_margin` around it.
pub const WHOLE: u32 = 800;
const MARGIN: u32 = 128;
/// `Manga.pad_mod`.
const MODULO: u32 = 16;
/// `Manga.seed`, which seeds the noise the inpaintor takes.
const SEED: u64 = 42;

#[derive(Debug)]
pub struct MangaInpainting {
    device: Device,
    // The stores own the weights the networks read.
    _stores: [nn::VarStore; 2],
    erika: Erika,
    inpaintor: Inpaintor,
}

impl MangaInpainting {
    pub async fn load(device: &crate::Device) -> Result<Self> {
        let (device, _) = crate::torch::device(device);
        let inpaintor_weights = weights(&INPAINTOR.resolve().await?)?;
        let line_weights = weights(&LINES.resolve().await?)?;
        let _turn = Turn::take(Library::Torch);
        // IOPaint runs both in float32.
        let mut line_store = nn::VarStore::new(device, Kind::Float);
        let erika = Erika::new(&line_store.root())?;
        line_store.load(line_weights)?;
        let mut inpaintor_store = nn::VarStore::new(device, Kind::Float);
        let inpaintor = Inpaintor::new(&inpaintor_store.root())?;
        inpaintor_store.load(inpaintor_weights)?;
        Ok(Self {
            device,
            _stores: [line_store, inpaintor_store],
            erika,
            inpaintor,
        })
    }

    /// `image` repainted where `mask` is white, `InpaintModel.__call__` at
    /// IOPaint's defaults: a page over 800 pixels on its long side goes a crop
    /// at a time.
    pub fn inference(&self, image: &RgbImage, mask: &GrayImage) -> Result<RgbImage> {
        ensure!(
            image.dimensions() == mask.dimensions(),
            "image and mask dimensions differ"
        );
        crate::torch::inference(|| {
            let (width, height) = image.dimensions();
            if width.max(height) <= WHOLE {
                return self.pad_forward(image, mask);
            }
            let mut result = image.clone();
            for found in crate::lama::boxes_from_mask(mask) {
                let [left, top, right, bottom] =
                    crate::lama::crop_box(width, height, found, MARGIN);
                let (crop_width, crop_height) = (right - left, bottom - top);
                let crop = imageops::crop_imm(image, left, top, crop_width, crop_height).to_image();
                let crop_mask =
                    imageops::crop_imm(mask, left, top, crop_width, crop_height).to_image();
                let painted = self.pad_forward(&crop, &crop_mask)?;
                imageops::replace(&mut result, &painted, left.into(), top.into());
            }
            Ok(result)
        })
    }

    /// `_pad_forward` with `Manga.forward`: padded symmetrically to a multiple
    /// of 16, the lines drawn and the mask filled, cropped back, and blended
    /// over the image by the mask.
    fn pad_forward(&self, image: &RgbImage, mask: &GrayImage) -> Result<RgbImage> {
        let (width, height) = image.dimensions();
        let padded = |side: u32| side.div_ceil(MODULO) * MODULO;
        let (padded_width, padded_height) = (padded(width), padded(height));
        let index = crate::image::symmetric_index;
        // `cv2.COLOR_RGB2GRAY` on 8 bits: BT.601 in 14-bit fixed point.
        let gray = GrayImage::from_fn(padded_width, padded_height, |x, y| {
            let [r, g, b] = image.get_pixel(index(x, width), index(y, height)).0;
            let y = (u32::from(r) * 4899 + u32::from(g) * 9617 + u32::from(b) * 1868 + 8192) >> 14;
            image::Luma([y as u8])
        });
        let held = GrayImage::from_fn(padded_width, padded_height, |x, y| {
            *mask.get_pixel(index(x, width), index(y, height))
        });

        let size = [1, 1, i64::from(padded_height), i64::from(padded_width)];
        let gray = Tensor::from_slice(gray.as_raw())?
            .view(size)?
            .to(self.device)?
            .to_kind(Kind::Float)?;
        let lines = self
            .erika
            .forward(&gray)?
            .clamp(Some(0.0.into()), Some(255.0.into()))?;
        let held = Tensor::from_slice(held.as_raw())?
            .view(size)?
            .to(self.device)?
            .gt_scalar(0.0)?
            .to_kind(Kind::Float)?;
        // `torch.manual_seed(42)` then `torch.randn_like(mask)`.
        let generator = Generator::new(self.device)?;
        generator.set_seed(SEED)?;
        let noise = held.empty_like(None, None)?;
        noise.normal_(0.0, 1.0, Some(&generator))?;
        let unit = |x: &Tensor| x.div_scalar(255.0)?.mul_scalar(2.0)?.sub_scalar(1.0, 1);
        let painted = self
            .inpaintor
            .forward(&unit(&gray)?, &unit(&lines)?, &held, &noise)?
            .mul_scalar(127.5)?
            .add_scalar(127.5, 1)?
            .to_kind(Kind::Uint8)?
            .narrow(2, 0, i64::from(height))?
            .narrow(3, 0, i64::from(width))?
            .contiguous(koharu_torch::MemoryFormat::Contiguous)?
            .to(Device::Cpu)?
            .view([-1])?;
        let painted = GrayImage::from_raw(width, height, Vec::<u8>::try_from(&painted)?)
            .context("the inpaintor's output isn't an image")?;
        // `sd_keep_unmasked_area`, the gray result in each channel.
        let mut result = image.clone();
        for (x, y, pixel) in result.enumerate_pixels_mut() {
            let alpha = f32::from(mask.get_pixel(x, y).0[0]) / 255.0;
            let new = f32::from(painted.get_pixel(x, y).0[0]);
            for old in &mut pixel.0 {
                *old = (new * alpha + f32::from(*old) * (1.0 - alpha)) as u8;
            }
        }
        Ok(result)
    }
}

/// The weights of the trace at `archive` as safetensors, made once beside it.
fn weights(archive: &Path) -> Result<PathBuf> {
    let converted = archive.with_extension("safetensors");
    if !converted.is_file() {
        crate::traced::convert(archive, &converted)
            .with_context(|| format!("couldn't read the weights of {}", archive.display()))?;
    }
    Ok(converted)
}
