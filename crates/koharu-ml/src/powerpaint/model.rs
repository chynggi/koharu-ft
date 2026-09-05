//! Native PowerPaint V1 model assembly.
//!
//! The converted GGUF carries the UNet, text encoder, and VAE together, so only
//! `model_path` is set. The task prompts stay outside it as textual inversion
//! embeddings, which stable-diffusion.cpp loads lazily when their trigger word
//! appears in a prompt.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context as _, Result, anyhow, ensure};
use koharu_diffusion::{Context, ContextParams, Embedding, ImageGenerationParams, RgbImage};

use crate::Backend;

/// Trigger words for the task prompts, matching the file names written by
/// `scripts/convert_powerpaint.py`.
pub(super) const CONTEXT_TOKEN: &str = "P_ctxt";
pub(super) const OBJECT_TOKEN: &str = "P_obj";

#[derive(Debug)]
pub(super) struct ModelPaths {
    pub model: PathBuf,
    pub embeddings_dir: PathBuf,
}

#[derive(Debug)]
pub(super) struct Model {
    context: Mutex<Context>,
}

impl Model {
    pub(super) fn new(device: &crate::Device, paths: ModelPaths) -> Result<Self> {
        let context = Context::new(&context_params(device, paths)?)
            .context("failed to load PowerPaint components")?;
        ensure!(
            context.supports_image_generation(),
            "the loaded PowerPaint context does not support image generation"
        );
        Ok(Self {
            context: Mutex::new(context),
        })
    }

    pub(super) fn forward(&self, params: &ImageGenerationParams) -> Result<Vec<RgbImage>> {
        let mut context = self
            .context
            .lock()
            .map_err(|_| anyhow!("PowerPaint context lock was poisoned"))?;
        context
            .generate_image(params)
            .context("PowerPaint inference failed")
    }
}

fn context_params(device: &crate::Device, paths: ModelPaths) -> Result<ContextParams> {
    let use_accelerator = device.backend != Backend::Cpu;
    Ok(ContextParams {
        model_path: Some(paths.model),
        embeddings: task_embeddings(&paths.embeddings_dir)?,
        enable_mmap: true,
        flash_attention: use_accelerator,
        diffusion_flash_attention: use_accelerator,
        backend: Some(if use_accelerator {
            device.name.to_ascii_lowercase()
        } else {
            "cpu".to_owned()
        }),
        ..ContextParams::default()
    })
}

/// A missing embedding would otherwise surface as the trigger word being
/// tokenised as ordinary text, which inpaints something plausible instead of
/// failing, so the files are checked up front.
fn task_embeddings(directory: &Path) -> Result<Vec<Embedding>> {
    [CONTEXT_TOKEN, OBJECT_TOKEN]
        .into_iter()
        .map(|name| {
            let path = directory.join(format!("{name}.safetensors"));
            ensure!(
                path.is_file(),
                "PowerPaint task embedding is missing: {}",
                path.display()
            );
            Ok(Embedding {
                name: name.to_owned(),
                path,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_task_embedding_is_reported_with_its_path() {
        let directory = tempfile::tempdir().unwrap();

        let error = task_embeddings(directory.path()).unwrap_err();

        assert!(error.to_string().contains("P_ctxt.safetensors"));
    }

    #[test]
    fn both_task_embeddings_are_loaded_in_order() {
        let directory = tempfile::tempdir().unwrap();
        for name in [CONTEXT_TOKEN, OBJECT_TOKEN] {
            std::fs::write(directory.path().join(format!("{name}.safetensors")), b"").unwrap();
        }

        let embeddings = task_embeddings(directory.path()).unwrap();

        let names: Vec<_> = embeddings.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, [CONTEXT_TOKEN, OBJECT_TOKEN]);
    }
}
