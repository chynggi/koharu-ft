//! PowerPaint image and mask processing.
//!
//! PowerPaint V1 is an SD1.5 inpainting model, so inference happens at a square
//! 512 and the caller's dimensions and unmasked pixels are restored afterwards.

use anyhow::{Result, ensure};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, GenericImageView, GrayImage, Luma, RgbImage};
use imageproc::{distance_transform::Norm, morphology::dilate};

#[derive(Debug, Clone, PartialEq)]
pub struct PowerPaintOptions {
    /// Square inference resolution. PowerPaint V1 was trained at 512.
    pub resolution: u32,
    /// Mask growth in model-space pixels. Text removal benefits from a little
    /// growth so anti-aliased glyph edges fall inside the mask.
    pub mask_dilation: u8,
    pub num_inference_steps: i32,
    pub guidance_scale: f32,
    pub strength: f32,
    pub seed: i64,
}

impl Default for PowerPaintOptions {
    fn default() -> Self {
        Self {
            resolution: 512,
            mask_dilation: 2,
            num_inference_steps: 20,
            guidance_scale: 7.0,
            strength: 1.0,
            seed: -1,
        }
    }
}

pub(super) struct Processor;

impl Processor {
    pub(super) fn validate(
        image: &DynamicImage,
        mask: &GrayImage,
        options: &PowerPaintOptions,
    ) -> Result<()> {
        ensure!(
            image.width() > 0 && image.height() > 0,
            "image dimensions must be non-zero"
        );
        ensure!(
            image.dimensions() == mask.dimensions(),
            "image and mask dimensions differ: image={:?}, mask={:?}",
            image.dimensions(),
            mask.dimensions()
        );
        ensure!(
            options.resolution >= 256 && options.resolution % 64 == 0,
            "PowerPaint resolution must be at least 256 and a multiple of 64"
        );
        ensure!(
            options.num_inference_steps > 0,
            "num_inference_steps must be greater than zero"
        );
        ensure!(
            options.guidance_scale.is_finite() && options.guidance_scale > 0.0,
            "guidance_scale must be finite and greater than zero"
        );
        ensure!(
            options.strength.is_finite() && options.strength > 0.0 && options.strength <= 1.0,
            "PowerPaint strength must be finite, greater than zero, and at most one"
        );
        Ok(())
    }

    pub(super) fn resize_image(image: &RgbImage, resolution: u32) -> Result<RgbImage> {
        if image.dimensions() == (resolution, resolution) {
            return Ok(image.clone());
        }
        let mut output = RgbImage::new(resolution, resolution);
        Resizer::new().resize(
            image,
            &mut output,
            &ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                .use_alpha(false),
        )?;
        Ok(output)
    }

    pub(super) fn resize_mask(
        mask: &GrayImage,
        resolution: u32,
        dilation: u8,
    ) -> Result<GrayImage> {
        let mut binary = mask.clone();
        for pixel in binary.pixels_mut() {
            *pixel = Luma([if pixel.0[0] < 128 { 0 } else { 255 }]);
        }
        let mut output = if binary.dimensions() == (resolution, resolution) {
            binary
        } else {
            let mut output = GrayImage::new(resolution, resolution);
            Resizer::new().resize(
                &binary,
                &mut output,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Nearest)
                    .use_alpha(false),
            )?;
            output
        };
        if dilation > 0 {
            output = dilate(&output, Norm::LInf, dilation);
        }
        Ok(output)
    }

    pub(super) fn composite(
        original: &RgbImage,
        native_mask: &GrayImage,
        generated: &RgbImage,
    ) -> Result<RgbImage> {
        let generated = Self::resize_rgb(generated, original.width(), original.height())?;
        let mask = Self::resize_gray(native_mask, original.width(), original.height())?;
        let mut output = generated;
        for (index, &masked) in mask.as_raw().iter().enumerate() {
            if masked == 0 {
                let offset = index * 3;
                output.as_flat_samples_mut().samples[offset..offset + 3]
                    .copy_from_slice(&original.as_raw()[offset..offset + 3]);
            }
        }
        Ok(output)
    }

    fn resize_rgb(image: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
        if image.dimensions() == (width, height) {
            return Ok(image.clone());
        }
        let mut output = RgbImage::new(width, height);
        Resizer::new().resize(
            image,
            &mut output,
            &ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                .use_alpha(false),
        )?;
        Ok(output)
    }

    fn resize_gray(image: &GrayImage, width: u32, height: u32) -> Result<GrayImage> {
        if image.dimensions() == (width, height) {
            return Ok(image.clone());
        }
        let mut output = GrayImage::new(width, height);
        Resizer::new().resize(
            image,
            &mut output,
            &ResizeOptions::new()
                .resize_alg(ResizeAlg::Nearest)
                .use_alpha(false),
        )?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, imageops};

    use super::*;

    #[test]
    fn defaults_match_the_verified_settings() {
        assert_eq!(
            PowerPaintOptions::default(),
            PowerPaintOptions {
                resolution: 512,
                mask_dilation: 2,
                num_inference_steps: 20,
                guidance_scale: 7.0,
                strength: 1.0,
                seed: -1,
            }
        );
    }

    #[test]
    fn a_resolution_off_the_latent_grid_is_rejected() {
        let image = DynamicImage::new_rgb8(64, 64);
        let mask = GrayImage::new(64, 64);
        let options = PowerPaintOptions {
            resolution: 500,
            ..PowerPaintOptions::default()
        };

        let error = Processor::validate(&image, &mask, &options).unwrap_err();

        assert!(error.to_string().contains("multiple of 64"));
    }

    #[test]
    fn a_full_strength_pass_is_allowed() {
        let image = DynamicImage::new_rgb8(64, 64);
        let mask = GrayImage::new(64, 64);
        let options = PowerPaintOptions {
            strength: 1.0,
            ..PowerPaintOptions::default()
        };

        Processor::validate(&image, &mask, &options).unwrap();
    }

    #[test]
    fn mask_is_binary_and_can_be_dilated() {
        let mut mask = GrayImage::new(512, 512);
        mask.put_pixel(256, 256, Luma([128]));

        let mask = Processor::resize_mask(&mask, 512, 1).unwrap();

        assert_eq!(mask.get_pixel(255, 255), &Luma([255]));
        assert_eq!(mask.get_pixel(256, 256), &Luma([255]));
        assert_eq!(mask.get_pixel(258, 258), &Luma([0]));
    }

    #[test]
    fn compositing_restores_original_size_and_unmasked_pixels() {
        let original = RgbImage::from_pixel(1920, 1080, Rgb([10, 20, 30]));
        let generated = RgbImage::from_pixel(512, 512, Rgb([200, 210, 220]));
        let mut mask = GrayImage::new(512, 512);
        imageops::replace(
            &mut mask,
            &GrayImage::from_pixel(256, 512, Luma([255])),
            256,
            0,
        );

        let output = Processor::composite(&original, &mask, &generated).unwrap();

        assert_eq!(output.dimensions(), original.dimensions());
        assert_eq!(output.get_pixel(0, 540), &Rgb([10, 20, 30]));
        assert_eq!(output.get_pixel(1919, 540), &Rgb([200, 210, 220]));
    }
}
