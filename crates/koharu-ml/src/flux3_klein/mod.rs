//! FLUX 3 Image Klein inpainting behind Black Forest Labs' prompt upsampler.
//!
//! Component mapping follows stable-diffusion.cpp at:
//! https://github.com/leejet/stable-diffusion.cpp/blob/8cd731e7b4667eb8f40a153315b9a7125d09a58d/docs/flux3.md
//! The upsampler message layout follows the model card at:
//! https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/337337621b8526e120f2275d250242f4a6bedd75/README.md
//! and its system prompts are the official file from
//! https://huggingface.co/black-forest-labs/flux-3-image-klein-pup/blob/3f3edb52b2c55de0d49feab28598b68f7f8e4830/system_prompts.json
//! The upsampler weights themselves are a community GGUF conversion.

use anyhow::{Context as _, Result, ensure};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, RgbImage};
use koharu_diffusion::{
    GuidanceParams, ImageGenerationParams, SampleMethod, SampleParams, Scheduler,
};

use self::model::{Model, ModelPaths};
use crate::flux2_klein::processor::Flux2ImageProcessor;

mod model;
mod source;
mod upsampler;

pub use self::{
    source::{ComponentSource, Flux3KleinSource},
    upsampler::Flux3PromptUpsampler,
};

crate::model_repository!("DogukanUrker/FLUX.3-Image-Klein-GGUF" @ "652e3dae6f7f26777c8749c99dcb3bb643a8350c" {
    TRANSFORMER_WEIGHTS = "flux3-image-klein-Q4_K.gguf"
});
crate::model_repository!("Qwen/Qwen3-VL-4B-Instruct-GGUF" @ "1cd86afb9a95c410a6038ab3b40d8b578c892266" {
    TEXT_ENCODER_WEIGHTS = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
});
// FLUX 3 decodes with the FLUX.2 autoencoder. Black Forest Labs' own copies
// sit in gated repositories; this ungated file has the same SHA-256 as the
// `vae/flux2-vae.safetensors` carried by ml-resources/flux-3-image-klein-comfy,
// an ungated mirror of BFL's ComfyUI repository.
crate::model_repository!("Comfy-Org/flux2-klein" @ "5f526678002e43af5551dadb73ce2e8c91b43afe" {
    VAE_WEIGHTS = "split_files/vae/flux2-vae.safetensors"
});
crate::model_repository!("prithivMLmods/flux-3-image-klein-prompt-upsampler-GGUF" @ "71713afaf9bce90074c078254d07f5b348f9c372" {
    UPSAMPLER_WEIGHTS = "flux-3-image-klein-pup.Q4_K_M.gguf",
    UPSAMPLER_PROJECTOR = "flux-3-image-klein-pup.mmproj-q8_0.gguf",
});
crate::model_repository!("black-forest-labs/flux-3-image-klein-pup" @ "3f3edb52b2c55de0d49feab28598b68f7f8e4830" {
    SYSTEM_PROMPTS = "system_prompts.json"
});

/// The default instruction the prompt upsampler expands into an inpainting caption.
pub const DEFAULT_REQUEST: &str =
    "Remove all text and lettering, and reconstruct the background behind it.";

/// The working area every tile is shrunk to before denoising.
pub const DEFAULT_MAX_PIXELS: u32 = 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct Flux3KleinInpaintOptions {
    pub padding_mask_crop: Option<u32>,
    pub strength: f64,
    /// The distilled checkpoint is built for 16 steps.
    pub num_inference_steps: usize,
    pub seed: i64,
    pub max_pixels: u32,
}

impl Default for Flux3KleinInpaintOptions {
    fn default() -> Self {
        Self {
            padding_mask_crop: None,
            strength: 0.8,
            num_inference_steps: 16,
            seed: -1,
            max_pixels: DEFAULT_MAX_PIXELS,
        }
    }
}

async fn resolve(source: &Flux3KleinSource) -> Result<ModelPaths> {
    let (transformer, text_encoder, vae) = tokio::try_join!(
        source.transformer.resolve(TRANSFORMER_WEIGHTS.into()),
        source.text_encoder.resolve(TEXT_ENCODER_WEIGHTS.into()),
        source.vae.resolve(VAE_WEIGHTS.into()),
    )
    .context("failed to resolve FLUX 3 Image Klein model assets")?;
    Ok(ModelPaths {
        transformer,
        text_encoder,
        vae,
    })
}

#[derive(Debug)]
pub struct Flux3KleinInpaint {
    model: Model,
}

impl Flux3KleinInpaint {
    pub async fn load(device: crate::Device, source: &Flux3KleinSource) -> Result<Self> {
        source.validate()?;
        let model = Model::new(&device, resolve(source).await?)?;
        Ok(Self { model })
    }

    /// Inpaints the masked area of `image` from a caption the prompt upsampler
    /// wrote. Mask preparation and the final overlay are the FLUX.2 Klein
    /// inpaint port's; only the sampler settings belong to FLUX 3.
    pub fn inference(
        &self,
        caption: &str,
        image: &DynamicImage,
        mask_image: &DynamicImage,
        options: &Flux3KleinInpaintOptions,
    ) -> Result<DynamicImage> {
        ensure!(
            image.width() == mask_image.width() && image.height() == mask_image.height(),
            "image/mask dimensions differ: image={}x{}, mask={}x{}",
            image.width(),
            image.height(),
            mask_image.width(),
            mask_image.height()
        );
        ensure!(
            !caption.contains('\0'),
            "caption contains an interior NUL byte"
        );
        ensure!(
            options.strength > 0.0 && options.strength <= 1.0,
            "FLUX 3 inpaint strength must be greater than zero and at most one"
        );
        ensure!(
            options.num_inference_steps > 0,
            "num_inference_steps must be greater than zero"
        );
        Flux2ImageProcessor::check_max_pixels(options.max_pixels)?;

        let image = Flux2ImageProcessor::fit_to_area(image, options.max_pixels);
        let width = (image.width() / 16) * 16;
        let height = (image.height() / 16) * 16;
        ensure!(
            width > 0 && height > 0,
            "image is smaller than one 16-pixel FLUX 3 patch"
        );
        let source = image.to_rgb8();
        let mut image = RgbImage::new(width, height);
        Resizer::new()
            .resize(
                &source,
                &mut image,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                    .use_alpha(false),
            )
            .expect("source and destination images have the same pixel type");
        let source_mask = mask_image.to_luma8();
        let mut mask_image = image::GrayImage::new(width, height);
        Resizer::new()
            .resize(
                &source_mask,
                &mut mask_image,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                    .use_alpha(false),
            )
            .expect("source and destination masks have the same pixel type");

        let crop_coords = options.padding_mask_crop.and_then(|padding| {
            Flux2ImageProcessor::get_crop_region(&mask_image, width, height, padding)
        });
        let (init_image, mut native_mask) = if let Some((x1, y1, x2, y2)) = crop_coords {
            let image_crop = DynamicImage::ImageRgb8(
                image::imageops::crop_imm(&image, x1, y1, x2 - x1, y2 - y1).to_image(),
            );
            let mask_crop = DynamicImage::ImageLuma8(
                image::imageops::crop_imm(&mask_image, x1, y1, x2 - x1, y2 - y1).to_image(),
            );
            (
                Flux2ImageProcessor::_resize_and_fill(&image_crop, width, height).to_rgb8(),
                Flux2ImageProcessor::_resize_and_fill(&mask_crop, width, height).to_luma8(),
            )
        } else {
            (image.clone(), mask_image.clone())
        };
        Flux2ImageProcessor::binarize(&mut native_mask);

        let strength =
            Flux2ImageProcessor::sd_strength(options.strength, options.num_inference_steps);

        // Following the FLUX.2 Klein inpaint port, the init image is also the edit
        // reference. FLUX 3 has no upstream inpaint pipeline, so whether this
        // helps or brings the erased lettering back is unverified until the
        // quality gate.
        let reference_images = vec![init_image.clone()];
        let generated = self
            .model
            .forward(&ImageGenerationParams {
                prompt: caption.to_owned(),
                width: i32::try_from(width)?,
                height: i32::try_from(height)?,
                init_image: Some(init_image),
                reference_images,
                reference_image_args: Some("resize_before_vae=0".into()),
                mask_image: Some(native_mask),
                sample: SampleParams {
                    guidance: GuidanceParams {
                        text_cfg: 1.0,
                        ..GuidanceParams::default()
                    },
                    // FLUX.3's own discrete schedule; the unset flow shift resolves to
                    // stable-diffusion.cpp's FLUX.3 default of 6.93.
                    scheduler: Scheduler::Discrete,
                    sample_method: SampleMethod::Euler,
                    sample_steps: i32::try_from(options.num_inference_steps)?,
                    ..SampleParams::default()
                },
                seed: options.seed,
                batch_count: 1,
                strength,
                ..ImageGenerationParams::default()
            })?
            .into_iter()
            .next()
            .context("FLUX 3 Image Klein returned no inpainted image")?;

        let generated =
            Flux2ImageProcessor::apply_overlay(&mask_image, &image, generated, crop_coords)?;
        Ok(DynamicImage::ImageRgb8(generated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_follow_the_distilled_checkpoint() {
        let options = Flux3KleinInpaintOptions::default();
        assert_eq!(options.num_inference_steps, 16);
        assert_eq!(options.max_pixels, DEFAULT_MAX_PIXELS);
    }
}
