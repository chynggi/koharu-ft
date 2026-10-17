//! FLUX 3 Image Klein inpainting behind the official prompt upsampler.
//!
//! Component mapping follows stable-diffusion.cpp at:
//! https://github.com/leejet/stable-diffusion.cpp/blob/8cd731e/docs/flux3.md
//! The upsampler message layout follows the model card at:
//! https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/337337621b8526e120f2275d250242f4a6bedd75/README.md

mod source;

pub use self::source::{ComponentSource, Flux3KleinSource};

crate::model_repository!("DogukanUrker/FLUX.3-Image-Klein-GGUF" @ "652e3dae6f7f26777c8749c99dcb3bb643a8350c" {
    TRANSFORMER_WEIGHTS = "flux3-image-klein-Q4_K.gguf"
});
crate::model_repository!("Qwen/Qwen3-VL-4B-Instruct-GGUF" @ "1cd86afb9a95c410a6038ab3b40d8b578c892266" {
    TEXT_ENCODER_WEIGHTS = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
});
// FLUX 3 ships this same FLUX.2 autoencoder as `ae.safetensors` (identical
// SHA-256), but the Black Forest Labs repository is gated; this copy is not.
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

/// The instruction the prompt upsampler expands into one caption per tile.
pub const DEFAULT_REQUEST: &str =
    "Remove all text and lettering, and reconstruct the background behind it.";
