//! IOPaint-compatible LaMa preprocessing, crop orchestration, and postprocessing.
//!
//! Original implementations:
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/base.py#L57-L192
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/helper.py#L187-L267

use anyhow::{Result, ensure};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_torch::{Device, Kind, Tensor};

use super::{backend::Backend, config::InpaintRequest};
use crate::inpaint_ops::{dispatch_hd_strategy, pad_img_to_modulo, post_process};

#[derive(Debug)]
pub(super) struct InpaintModel {
    device: Device,
}

impl InpaintModel {
    pub(super) fn new(device: Device) -> Self {
        Self { device }
    }

    pub(super) fn call(
        &self,
        model: &Backend,
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
            self.pad_forward(model, image, mask, config)
        })
    }

    fn pad_forward(
        &self,
        model: &Backend,
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
        let model_image = pad_img_to_modulo(image_tensor.to_kind(Kind::Float) / 255.0, 8);
        let model_mask = pad_img_to_modulo(mask_tensor.gt(0.0).to_kind(Kind::Float), 8);
        let output = model
            .forward(&model_image, &model_mask)?
            .narrow(2, 0, i64::from(height))
            .narrow(3, 0, i64::from(width))
            .clamp(0.0, 1.0)
            * 255.0;
        let output = output.to_kind(Kind::Uint8);
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
