//! MI-GAN only accepts square 512x512 input. Larger content is cropped to the
//! mask's bounding box, shrunk to fit inside 512, inferred as a padded square,
//! and pasted back so only the masked area changes.
//!
//! Original orchestration and input convention:
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/mi_gan.py

use anyhow::{Result, ensure};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_torch::{Device, Kind, Tensor};

use crate::{
    inpaint_ops::{
        boxes_from_mask, crop_box, pad_img_to_modulo, post_process, resize_dimensions, resize_gray,
        resize_rgb,
    },
    lama::InpaintRequest,
    torchscript::TorchScript,
};

/// `mi_gan.py`'s `min_size` / `pad_mod` / `pad_to_square`, which together
/// force every model input to exactly 512x512.
const SIZE: u32 = 512;
/// `mi_gan.py` forces `hd_strategy_crop_margin = 128` regardless of the request.
const CROP_MARGIN: u32 = 128;
/// `mi_gan.py`'s `(mask > 120) * 255` binarization.
const MASK_THRESHOLD: f64 = 120.0;

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
        model: &TorchScript,
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

        if image.width() == SIZE && image.height() == SIZE {
            return self.pad_forward(model, &image, mask, config);
        }

        let mut result = image.clone();
        for bounding_box in boxes_from_mask(mask) {
            let [left, top, right, bottom] =
                crop_box(image.width(), image.height(), bounding_box, CROP_MARGIN);
            let (width, height) = (right - left, bottom - top);
            let crop_image = image::imageops::crop_imm(&image, left, top, width, height).to_image();
            let crop_mask = image::imageops::crop_imm(mask, left, top, width, height).to_image();

            // `resize_max_size` only shrinks. A crop that already fits inside
            // 512 is padded up to the square instead of being upscaled.
            let (small_width, small_height) = if width.max(height) > SIZE {
                resize_dimensions(width, height, SIZE)
            } else {
                (width, height)
            };
            let resized = small_width != width || small_height != height;
            let small_image = if resized {
                resize_rgb(&crop_image, small_width, small_height)?
            } else {
                crop_image.clone()
            };
            let small_mask = if resized {
                resize_gray(&crop_mask, small_width, small_height)?
            } else {
                crop_mask.clone()
            };

            let inpainted = self.pad_forward(model, &small_image, &small_mask, config)?;
            let mut restored = if resized {
                resize_rgb(&inpainted, width, height)?
            } else {
                inpainted
            };

            // Paste only the masked area back; the resize round trip would
            // otherwise soften the untouched pixels. This is `mi_gan.py`'s
            // `original_pixel_indices`.
            for (index, value) in crop_mask.as_raw().iter().enumerate() {
                if *value < 127 {
                    let offset = index * 3;
                    restored.as_flat_samples_mut().samples[offset..offset + 3]
                        .copy_from_slice(&crop_image.as_raw()[offset..offset + 3]);
                }
            }

            image::imageops::replace(&mut result, &restored, left.into(), top.into());
        }
        Ok(result)
    }

    /// `mi_gan.py`'s `_pad_forward` + `forward`. The model input is the
    /// 4-channel `cat([0.5 - mask, image * (1 - mask)])` with the image in
    /// [-1, 1] and the mask in {0, 1}.
    fn pad_forward(
        &self,
        model: &TorchScript,
        image: &RgbImage,
        mask: &GrayImage,
        config: &InpaintRequest,
    ) -> Result<RgbImage> {
        let width = image.width();
        let height = image.height();
        // The traced graph only accepts 512x512. Every path in `call` keeps
        // both sides at or below 512, and padding to modulo 512 lifts each
        // side to exactly 512 — upstream's `pad_to_square`.
        ensure!(
            width <= SIZE && height <= SIZE,
            "MI-GAN input exceeds its {SIZE}px limit: {width}x{height}"
        );

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

        let model_image =
            pad_img_to_modulo(image_tensor.to_kind(Kind::Float) / 255.0 * 2.0 - 1.0, SIZE);
        let model_mask = pad_img_to_modulo(
            mask_tensor
                .to_kind(Kind::Float)
                .gt(MASK_THRESHOLD)
                .to_kind(Kind::Float),
            SIZE,
        );
        ensure!(
            model_image.size()[2] == i64::from(SIZE) && model_image.size()[3] == i64::from(SIZE),
            "MI-GAN padding must produce a {SIZE}x{SIZE} square, got {:?}",
            model_image.size()
        );

        let erased = &model_image * (model_mask.ones_like() - &model_mask);
        let input = Tensor::cat(&[&model_mask * -1.0 + 0.5, erased], 1);

        let output = model.forward(&[&input])?;
        let output = (output * 127.5 + 127.5)
            .round()
            .clamp(0.0, 255.0)
            .narrow(2, 0, i64::from(height))
            .narrow(3, 0, i64::from(width))
            .to_kind(Kind::Uint8);
        let output = if config.sd_keep_unmasked_area {
            let alpha = mask_tensor.to_kind(Kind::Float) / 255.0;
            (output.to_kind(Kind::Float) * &alpha
                + image_tensor.to_kind(Kind::Float) * (alpha.ones_like() - alpha))
                .to_kind(Kind::Uint8)
        } else {
            output
        };
        post_process(&output, width, height)
    }
}
