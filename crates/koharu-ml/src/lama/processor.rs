//! IOPaint-compatible LaMa preprocessing, crop orchestration, and postprocessing.
//!
//! Original implementations:
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/model/base.py#L57-L192
//! https://github.com/Sanster/IOPaint/blob/61a759fb3f332bacdce8b2813f4837495c9b86e0/iopaint/helper.py#L187-L267

use anyhow::{Result, ensure};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_torch::{Device, Kind, Tensor};

use super::{
    backend::Backend,
    config::{HDStrategy, InpaintRequest},
};
use crate::inpaint_ops::{
    boxes_from_mask, crop_box, pad_img_to_modulo, post_process, resize_dimensions, resize_gray,
    resize_rgb,
};

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

        match config.hd_strategy {
            HDStrategy::Crop
                if image.width().max(image.height()) > config.hd_strategy_crop_trigger_size =>
            {
                let boxes = boxes_from_mask(mask);
                let mut crop_results = Vec::with_capacity(boxes.len());
                for bounding_box in boxes {
                    let crop_box = crop_box(
                        image.width(),
                        image.height(),
                        bounding_box,
                        config.hd_strategy_crop_margin,
                    );
                    let [left, top, right, bottom] = crop_box;
                    let crop_image =
                        image::imageops::crop_imm(&image, left, top, right - left, bottom - top)
                            .to_image();
                    let crop_mask =
                        image::imageops::crop_imm(mask, left, top, right - left, bottom - top)
                            .to_image();
                    crop_results.push((
                        self.pad_forward(model, &crop_image, &crop_mask, config)?,
                        crop_box,
                    ));
                }

                let mut result = image;
                for (crop_result, [left, top, _, _]) in crop_results {
                    image::imageops::replace(&mut result, &crop_result, left.into(), top.into());
                }
                Ok(result)
            }
            HDStrategy::Resize
                if image.width().max(image.height()) > config.hd_strategy_resize_limit =>
            {
                let (width, height) = resize_dimensions(
                    image.width(),
                    image.height(),
                    config.hd_strategy_resize_limit,
                );
                let resized_image = resize_rgb(&image, width, height)?;
                let resized_mask = resize_gray(mask, width, height)?;
                let resized_result =
                    self.pad_forward(model, &resized_image, &resized_mask, config)?;
                let mut result = resize_rgb(&resized_result, image.width(), image.height())?;
                for (index, value) in mask.as_raw().iter().enumerate() {
                    if *value < 127 {
                        let offset = index * 3;
                        result.as_mut()[offset..offset + 3]
                            .copy_from_slice(&image.as_raw()[offset..offset + 3]);
                    }
                }
                Ok(result)
            }
            _ => self.pad_forward(model, &image, mask, config),
        }
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
