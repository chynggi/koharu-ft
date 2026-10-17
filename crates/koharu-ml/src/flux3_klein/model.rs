//! Native FLUX 3 Image Klein model assembly.
//!
//! Component mapping follows stable-diffusion.cpp at:
//! https://github.com/leejet/stable-diffusion.cpp/blob/8cd731e7b4667eb8f40a153315b9a7125d09a58d/docs/flux3.md

use std::{path::PathBuf, sync::Mutex};

use anyhow::{Context as _, Result, anyhow, ensure};
use koharu_diffusion::{Context, ContextParams, ImageGenerationParams, RgbImage, VaeFormat};

use crate::Backend;

#[derive(Debug)]
pub(super) struct ModelPaths {
    pub transformer: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
}

#[derive(Debug)]
pub(super) struct Model {
    context: Mutex<Context>,
}

impl Model {
    pub fn new(device: &crate::Device, paths: ModelPaths) -> Result<Self> {
        let context = Context::new(&context_params(device, paths))
            .context("failed to load FLUX 3 Image Klein components")?;
        ensure!(
            context.supports_image_generation(),
            "the loaded FLUX 3 Image Klein context does not support image generation"
        );
        Ok(Self {
            context: Mutex::new(context),
        })
    }

    pub fn forward(&self, params: &ImageGenerationParams) -> Result<Vec<RgbImage>> {
        let mut context = self
            .context
            .lock()
            .map_err(|_| anyhow!("FLUX 3 Image Klein context lock was poisoned"))?;
        context
            .generate_image(params)
            .context("FLUX 3 Image Klein inference failed")
    }
}

fn context_params(device: &crate::Device, paths: ModelPaths) -> ContextParams {
    let use_accelerator = device.backend != Backend::Cpu;
    let use_cuda = device.backend == Backend::Cuda;
    ContextParams {
        diffusion_model_path: Some(paths.transformer),
        llm_path: Some(paths.text_encoder),
        vae_path: Some(paths.vae),
        enable_mmap: true,
        flash_attention: use_cuda,
        diffusion_flash_attention: use_cuda,
        diffusion_conv_direct: use_cuda,
        vae_conv_direct: use_cuda,
        // FLUX 3 Image Klein decodes with the FLUX.2 autoencoder.
        vae_format: VaeFormat::Flux2,
        backend: Some(device.name.clone()),
        // Unlike FLUX.2 Klein, weights always stay in RAM so the prompt
        // upsampler can claim the accelerator between pages.
        params_backend: use_accelerator.then(|| "*=cpu".to_owned()),
        ..ContextParams::default()
    }
}
