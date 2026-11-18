//! Manga OCR CPT-flex constants and checkpoint configuration.
//!
//! Upstream hardcodes the network constants (`D`, `HEADS`, `EPS`, `MAX_NEW`)
//! and the input budget instead of reading them from `config.json`:
//! https://github.com/muscgab/manga-ocr-CPT/blob/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt/model.py
//! https://github.com/muscgab/manga-ocr-CPT/blob/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt/preprocess.py
//! `config.json` is therefore only checked against them.

use std::{fs, path::Path};

use anyhow::{Result, ensure};
use serde::Deserialize;

pub(super) const HIDDEN_SIZE: i64 = 768;
pub(super) const NUM_HEADS: i64 = 12;
pub(super) const HEAD_DIM: i64 = HIDDEN_SIZE / NUM_HEADS;
pub(super) const INTERMEDIATE_SIZE: i64 = 3072;
pub(super) const ENCODER_LAYERS: usize = 12;
pub(super) const DECODER_LAYERS: usize = 2;
/// Every LayerNorm, encoder and decoder alike, uses BERT's 1e-12.
pub(super) const LAYER_NORM_EPS: f64 = 1e-12;
pub(super) const PATCH_SIZE: i64 = 16;
/// Side of the learned position table of the original 224x224 ViT.
pub(super) const POSITION_GRID: i64 = 14;
pub(super) const MAX_PATCHES: i64 = POSITION_GRID * POSITION_GRID;
/// Three equal grey channels times one 16x16 patch.
pub(super) const PATCH_FEATURES: i64 = 3 * PATCH_SIZE * PATCH_SIZE;
pub(super) const DECODER_POSITIONS: i64 = 512;
pub(super) const CHARACTERS: i64 = 6817;
/// Ids 0..5 are pad, unk, start, end, and mask; character `i` is id `5 + i`.
pub(super) const SPECIAL_TOKENS: i64 = 5;
pub(super) const OUTPUT_TOKENS: i64 = CHARACTERS + SPECIAL_TOKENS;
pub(super) const START_TOKEN: i64 = 2;
pub(super) const END_TOKEN: i64 = 3;
/// At most 64 characters plus the end token.
pub(super) const MAX_NEW_TOKENS: i64 = 65;

#[derive(Debug, Deserialize)]
pub(super) struct ModelConfig {
    vocab: i64,
    special_tokens: SpecialTokens,
    input: InputConfig,
    max_new_tokens: i64,
}

#[derive(Debug, Deserialize)]
struct SpecialTokens {
    pad: i64,
    unk: i64,
    start: i64,
    end: i64,
    mask: i64,
    first_char: i64,
}

#[derive(Debug, Deserialize)]
struct InputConfig {
    grey: bool,
    fit: String,
    max_patches: i64,
    patch: i64,
    mean: f64,
    std: f64,
}

impl ModelConfig {
    pub(super) fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.vocab == CHARACTERS,
            "Manga OCR CPT-flex expects {CHARACTERS} characters, the config has {}",
            self.vocab
        );
        let tokens = &self.special_tokens;
        ensure!(
            (
                tokens.pad,
                tokens.unk,
                tokens.start,
                tokens.end,
                tokens.mask,
                tokens.first_char
            ) == (0, 1, START_TOKEN, END_TOKEN, 4, SPECIAL_TOKENS),
            "Manga OCR CPT-flex special token ids differ from the ported layout"
        );
        ensure!(
            self.input.fit == "aspect",
            "Manga OCR CPT-flex expects the aspect-preserving input, the config has fit = {:?}",
            self.input.fit
        );
        ensure!(self.input.grey, "Manga OCR CPT-flex expects grey input");
        ensure!(
            self.input.max_patches == MAX_PATCHES && self.input.patch == PATCH_SIZE,
            "Manga OCR CPT-flex expects at most {MAX_PATCHES} patches of {PATCH_SIZE} px"
        );
        ensure!(
            self.input.mean == 0.5 && self.input.std == 0.5,
            "Manga OCR CPT-flex expects mean and std of 0.5"
        );
        ensure!(
            self.max_new_tokens == MAX_NEW_TOKENS,
            "Manga OCR CPT-flex expects at most {MAX_NEW_TOKENS} generated tokens"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;

    /// `config.json` of the pinned checkpoint, reduced to the fields Koharu reads.
    const RELEASED: &str = r#"{
        "vocab": 6817,
        "special_tokens": {"pad": 0, "unk": 1, "start": 2, "end": 3, "mask": 4, "first_char": 5},
        "input": {"grey": true, "fit": "aspect", "max_patches": 196, "patch": 16,
                  "resample": "bilinear", "fill": "median border value", "mean": 0.5, "std": 0.5},
        "max_new_tokens": 65
    }"#;

    #[test]
    fn the_released_config_matches_the_ported_constants() {
        let config: ModelConfig = serde_json::from_str(RELEASED).unwrap();
        config.validate().unwrap();
    }

    #[test]
    fn the_square_input_variant_is_rejected() {
        let config: ModelConfig =
            serde_json::from_str(&RELEASED.replace("\"aspect\"", "\"square\"")).unwrap();
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("aspect"), "{error}");
    }
}
