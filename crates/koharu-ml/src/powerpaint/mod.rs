//! PowerPaint V1 inpainting through stable-diffusion.cpp.
//!
//! PowerPaint learns a task through a word rather than an architecture: three
//! sets of ten token embeddings select context-aware filling, text-guided
//! object insertion, or shape-guided insertion from one SD1.5 inpainting UNet.
//! Paper: https://arxiv.org/abs/2312.03594
//! Weights: https://huggingface.co/Sanster/PowerPaint-V1-stable-diffusion-inpainting
//!
//! Erasing manga text is context-aware filling, so this integration pins the
//! `P_ctxt` task and pushes `P_obj` into the negative prompt — the combination
//! the reference demo uses for object removal. The other two tasks insert new
//! content, which is never what cleaning a page wants, so they are not exposed.
//!
//! The model files are produced by `scripts/convert_powerpaint.py` and named by
//! the caller; nothing is downloaded.

mod model;
mod processor;

use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_diffusion::{
    GuidanceParams, ImageGenerationParams, SampleMethod, SampleParams, Scheduler,
};

use self::{
    model::{CONTEXT_TOKEN, Model, ModelPaths, OBJECT_TOKEN},
    processor::Processor,
};

pub use self::processor::PowerPaintOptions;

/// Where the converted model and its task embeddings live.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PowerPaintPaths {
    /// The converted GGUF holding the UNet, text encoder, and VAE.
    pub model: PathBuf,
    /// Directory holding `P_ctxt.safetensors` and `P_obj.safetensors`.
    pub embeddings_dir: PathBuf,
}

impl PowerPaintPaths {
    /// Rejects what can be seen without loading gigabytes, so a bad setting is
    /// reported when it is saved rather than mid-page.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model.is_absolute(),
            "PowerPaint model path must be absolute: {}",
            self.model.display()
        );
        ensure!(
            self.model.is_file(),
            "PowerPaint model file does not exist: {}",
            self.model.display()
        );
        ensure!(
            self.embeddings_dir.is_absolute(),
            "PowerPaint embeddings directory must be absolute: {}",
            self.embeddings_dir.display()
        );
        ensure!(
            self.embeddings_dir.is_dir(),
            "PowerPaint embeddings directory does not exist: {}",
            self.embeddings_dir.display()
        );
        Ok(())
    }
}

#[derive(Debug)]
pub struct PowerPaint {
    model: Model,
}

impl PowerPaint {
    pub fn load(device: crate::Device, paths: &PowerPaintPaths) -> Result<Self> {
        paths.validate()?;
        let model = Model::new(
            &device,
            ModelPaths {
                model: paths.model.clone(),
                embeddings_dir: paths.embeddings_dir.clone(),
            },
        )
        .context("failed to load PowerPaint")?;
        Ok(Self { model })
    }

    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &GrayImage,
        options: &PowerPaintOptions,
    ) -> Result<RgbImage> {
        Processor::validate(image, mask, options)?;

        let original = image.to_rgb8();
        let init_image = Processor::resize_image(&original, options.resolution)?;
        let native_mask = Processor::resize_mask(mask, options.resolution, options.mask_dilation)?;
        if native_mask.as_raw().iter().all(|&value| value == 0) {
            return Ok(original);
        }

        let generated = self
            .model
            .forward(&ImageGenerationParams {
                prompt: CONTEXT_TOKEN.to_owned(),
                negative_prompt: OBJECT_TOKEN.to_owned(),
                init_image: Some(init_image),
                mask_image: Some(native_mask.clone()),
                width: i32::try_from(options.resolution)?,
                height: i32::try_from(options.resolution)?,
                sample: SampleParams {
                    guidance: GuidanceParams {
                        text_cfg: options.guidance_scale,
                        ..GuidanceParams::default()
                    },
                    scheduler: Scheduler::Discrete,
                    sample_method: SampleMethod::EulerA,
                    sample_steps: options.num_inference_steps,
                    ..SampleParams::default()
                },
                strength: options.strength,
                seed: options.seed,
                batch_count: 1,
                ..ImageGenerationParams::default()
            })?
            .into_iter()
            .next()
            .context("PowerPaint returned no inpainted image")?;
        ensure!(
            generated.dimensions() == (options.resolution, options.resolution),
            "PowerPaint returned an unexpected image size: {:?}",
            generated.dimensions()
        );

        Processor::composite(&original, &native_mask, &generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_relative_model_path_is_rejected() {
        let paths = PowerPaintPaths {
            model: PathBuf::from("powerpaint-v1.gguf"),
            embeddings_dir: PathBuf::from("embeddings"),
        };

        assert!(
            paths
                .validate()
                .unwrap_err()
                .to_string()
                .contains("absolute")
        );
    }

    #[test]
    fn a_missing_model_file_is_reported_before_loading() {
        let directory = tempfile::tempdir().unwrap();
        let paths = PowerPaintPaths {
            model: directory.path().join("powerpaint-v1.gguf"),
            embeddings_dir: directory.path().to_path_buf(),
        };

        assert!(
            paths
                .validate()
                .unwrap_err()
                .to_string()
                .contains("does not exist")
        );
    }
}
