//! Hayai OCR configuration.
//!
//! Upstream configuration:
//! https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.5-nova/blob/2803d0dc005fa4396ab1e7c5a472d2c450a247f7/configuration_hayai.py
//! The vision encoder is the `Siglip2VisionModel` instantiated by
//! `modeling_hayai.HayaiModel` from `google/siglip2-base-patch16-naflex`, whose
//! released config only overrides the model type, so every value below is the
//! transformers default:
//! https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip2/configuration_siglip2.py

use std::{fs, path::Path};

use anyhow::{Result, ensure};
use serde::Deserialize;

/// Vision-encoder constants of `google/siglip2-base-patch16-naflex`.
pub(super) mod siglip2 {
    pub const HIDDEN_SIZE: i64 = 768;
    pub const INTERMEDIATE_SIZE: i64 = 3072;
    pub const NUM_HIDDEN_LAYERS: usize = 12;
    pub const NUM_ATTENTION_HEADS: i64 = 12;
    pub const HEAD_DIM: i64 = HIDDEN_SIZE / NUM_ATTENTION_HEADS;
    pub const NUM_CHANNELS: i64 = 3;
    pub const PATCH_SIZE: i64 = 16;
    /// NaFlex keeps a square base grid of position embeddings that is
    /// interpolated per image instead of forcing a fixed input resolution.
    pub const POSITION_EMBEDDING_SIZE: i64 = 16;
    pub const LAYER_NORM_EPS: f64 = 1e-6;
    /// Maximum number of 16x16 patches emitted by the SigLIP2 image processor.
    ///
    /// Nova's DSC projector merges 2x2 patches, so the decoder sees roughly a
    /// quarter of this budget. Upstream model card (JMangaBench_Mixed, 3,286
    /// crops, NFKC, T4 latency):
    ///
    /// | budget | decoder tokens | CER   | EM     | latency |
    /// |-------:|---------------:|------:|-------:|--------:|
    /// | 256    | <= 64          | 4.95% | 75.35% | 1.00x   |
    /// | 384    | <= 96          | 3.65% | 79.15% | ~1.19x  |
    /// | 512    | <= 128         | 3.10% | 80.68% | ~1.34x  |
    ///
    /// Koharu takes 512: 384 does not beat v2.1's 3.23% CER, and quality is
    /// worth the ~34% latency over 256.
    pub const MAX_NUM_PATCHES: usize = 512;
}

/// Decoder constants of `VisualCausalOCRDecoder`; upstream hardcodes them in
/// its layer construction rather than exposing them through the config file.
pub(super) mod decoder {
    pub const QUERY_HEADS: i64 = 8;
    pub const KEY_VALUE_HEADS: i64 = 2;
    pub const HEAD_DIM: i64 = 64;
    pub const ROPE_THETA: f64 = 10_000.0;
}

/// Spatial downscale of Nova's `DSCProjector`: a 2x2 pixel unshuffle merges
/// `SPATIAL_DOWNSCALE^2` patches into one decoder token.
pub(super) const SPATIAL_DOWNSCALE: i64 = 2;

/// Decoder visual grid of a `patch_height x patch_width` patch grid after the
/// unshuffle: `ceil(H / 2) x ceil(W / 2)`, so the visual prefix holds
/// `H_out * W_out` tokens.
pub(super) fn decoder_vision_grid(patch_height: i64, patch_width: i64) -> (i64, i64) {
    (
        (patch_height + SPATIAL_DOWNSCALE - 1) / SPATIAL_DOWNSCALE,
        (patch_width + SPATIAL_DOWNSCALE - 1) / SPATIAL_DOWNSCALE,
    )
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct HayaiConfig {
    pub(crate) vocab_size: i64,
    pub(crate) d_model: i64,
    pub(crate) d_vision: i64,
    #[allow(dead_code)]
    pub(crate) d_ffn: i64,
    pub(crate) n_layers: usize,
}

impl HayaiConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.vocab_size > 0,
            "Hayai OCR vocabulary size must be positive"
        );
        ensure!(
            self.d_vision == siglip2::HIDDEN_SIZE,
            "Hayai OCR expects the SigLIP2 NaFlex {}-dimensional vision encoder",
            siglip2::HIDDEN_SIZE
        );
        ensure!(
            self.d_model == decoder::QUERY_HEADS * decoder::HEAD_DIM,
            "decoder model size does not match the hardcoded query head layout"
        );
        ensure!(
            decoder::QUERY_HEADS % decoder::KEY_VALUE_HEADS == 0,
            "decoder query heads must be divisible by key/value heads"
        );
        ensure!(self.n_layers > 0, "decoder must have at least one layer");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::decoder_vision_grid;

    #[test]
    fn decoder_vision_grid_halves_each_side_rounding_up() {
        assert_eq!(decoder_vision_grid(16, 16), (8, 8));
        assert_eq!(decoder_vision_grid(3, 5), (2, 3));
        assert_eq!(decoder_vision_grid(1, 1), (1, 1));
        // Extreme aspect ratio at the full 512-patch budget.
        assert_eq!(decoder_vision_grid(1, 512), (1, 256));
    }
}
