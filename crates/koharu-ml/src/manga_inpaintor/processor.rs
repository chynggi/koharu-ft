//! The Manga inpainter only fills in grayscale: the page is converted to
//! gray, the line model recovers the line work, and the inpaintor fills the
//! masked region. The result is replicated to RGB so unmasked pixels keep
//! their original color.
//!
//! Original pipeline:
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/manga.py

use anyhow::{Result, ensure};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_torch::{Cuda, Device, Kind, Tensor};

use crate::{
    inpaint_ops::{dispatch_hd_strategy, pad_img_to_modulo, post_process},
    lama::InpaintRequest,
    torchscript::TorchScript,
};

/// `manga.py`'s `pad_mod = 16`.
const PAD_MOD: u32 = 16;
/// `manga.py` reseeds every forward call with `self.seed = 42`, which makes
/// the noise channel of the inpaintor input reproducible run to run.
const SEED: i64 = 42;

#[derive(Debug)]
pub(super) struct Processor {
    device: Device,
}

impl Processor {
    pub(super) fn new(device: Device) -> Self {
        Self { device }
    }

    pub(super) fn call(
        &self,
        inpaintor: &TorchScript,
        line_model: &TorchScript,
        image: &DynamicImage,
        mask: &GrayImage,
        config: &InpaintRequest,
    ) -> Result<RgbImage> {
        let image = image.to_rgb8();
        ensure!(
            image.dimensions() == mask.dimensions(),
            "image and mask dimensions differ: image={:?}, mask={:?}",
            image.dimensions(),
            mask.dimensions()
        );
        ensure!(
            image.width() > 0 && image.height() > 0,
            "image dimensions must be non-zero"
        );

        dispatch_hd_strategy(&image, mask, config, &|image, mask| {
            self.pad_forward(inpaintor, line_model, image, mask, config)
        })
    }

    /// `manga.py`'s `_pad_forward` + `forward`. The line model sees the gray
    /// page in 0..255, and the inpaintor sees
    /// `(gray, lines, mask, noise, ones)` with gray and lines in [-1, 1] and
    /// the mask in {0, 1}.
    fn pad_forward(
        &self,
        inpaintor: &TorchScript,
        line_model: &TorchScript,
        image: &RgbImage,
        mask: &GrayImage,
        config: &InpaintRequest,
    ) -> Result<RgbImage> {
        let width = image.width();
        let height = image.height();

        let image_tensor = Tensor::from_slice(image.as_raw())
            .view([i64::from(height), i64::from(width), 3])
            .to_device(self.device)
            .permute([2, 0, 1])
            .unsqueeze(0)
            .contiguous();
        let mask_tensor = Tensor::from_slice(mask.as_raw())
            .view([i64::from(height), i64::from(width)])
            .to_device(self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .contiguous();

        let gray = pad_img_to_modulo(
            Tensor::from_slice(rgb_to_gray(image).as_raw())
                .view([i64::from(height), i64::from(width)])
                .to_device(self.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .contiguous()
                .to_kind(Kind::Float),
            PAD_MOD,
        );
        let lines = line_model.forward(&[&gray])?.clamp(0.0, 255.0) / 255.0 * 2.0 - 1.0;
        let model_mask = pad_img_to_modulo(mask_tensor.to_kind(Kind::Float), PAD_MOD)
            .gt(0.5)
            .to_kind(Kind::Float);

        // `manga.py` reseeds before every forward, so identical shapes get the
        // identical noise the upstream model was tuned for.
        koharu_torch::manual_seed(SEED);
        if self.device.is_cuda() {
            Cuda::manual_seed_all(u64::try_from(SEED).expect("the seed fits in u64"));
        }
        let noise = model_mask.randn_like();
        let ones = model_mask.ones_like();

        let gray = &gray / 255.0 * 2.0 - 1.0;
        let output = inpaintor.forward(&[&gray, &lines, &model_mask, &noise, &ones])?;

        // `astype(uint8)` truncates, then the gray channel is replicated to
        // RGB like `cv2.cvtColor(GRAY2BGR)`.
        let output = (output * 127.5 + 127.5)
            .narrow(2, 0, i64::from(height))
            .narrow(3, 0, i64::from(width))
            .to_kind(Kind::Uint8)
            .to_kind(Kind::Float)
            .repeat([1, 3, 1, 1]);
        let output = if config.sd_keep_unmasked_area {
            let alpha = mask_tensor.to_kind(Kind::Float) / 255.0;
            (output * &alpha + image_tensor.to_kind(Kind::Float) * (alpha.ones_like() - alpha))
                .to_kind(Kind::Uint8)
        } else {
            output.to_kind(Kind::Uint8)
        };
        post_process(&output, width, height)
    }
}

/// OpenCV's `COLOR_RGB2GRAY` (BT.601) with its fixed-point coefficients, so
/// the gray page the line model sees matches upstream byte for byte. The
/// `image` crate's own luma conversion uses different coefficients.
fn rgb_to_gray(image: &RgbImage) -> GrayImage {
    let mut gray = GrayImage::new(image.width(), image.height());
    for (pixel, luma) in image.pixels().zip(gray.pixels_mut()) {
        let [red, green, blue] = pixel.0;
        let value =
            (u32::from(red) * 9798 + u32::from(green) * 19235 + u32::from(blue) * 3735 + 16384)
                >> 15;
        *luma = image::Luma([u8::try_from(value).expect("the weighted sum fits in u8")]);
    }
    gray
}
