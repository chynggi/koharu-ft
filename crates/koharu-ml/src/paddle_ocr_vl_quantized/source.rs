//! Where the quantized PaddleOCR-VL-1.6 components are loaded from.
//!
//! The shipped repositories stay the default. An override only changes which
//! file is handed to llama.cpp; the architecture is still read from the file's
//! own metadata, so a checkpoint the pinned backend does not know fails at
//! load rather than being silently reinterpreted.

use anyhow::{Context as _, Result};

pub use crate::source::ComponentSource;

/// A LoRA adapter attached to the language model and the scale it is applied at.
#[derive(Clone, Debug, PartialEq)]
pub struct LoraSource {
    pub adapter: ComponentSource,
    pub scale: f32,
}

impl Default for LoraSource {
    fn default() -> Self {
        Self {
            adapter: ComponentSource::Builtin,
            scale: 1.0,
        }
    }
}

impl LoraSource {
    pub fn validate(&self) -> Result<()> {
        self.adapter.validate().context("LoRA adapter")?;
        Ok(())
    }
}

/// The model, projector, and optional LoRA a quantized PaddleOCR-VL-1.6
/// context is assembled from.
///
/// The manga reader variant is the same language model with a replaced
/// projector (the fine-tuned vision tower) and a LoRA adapter, so both are
/// optional here and selected by the caller.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PaddleOCRVLQuantizedSource {
    pub model: ComponentSource,
    pub projector: ComponentSource,
    pub lora: Option<LoraSource>,
}

impl PaddleOCRVLQuantizedSource {
    pub fn validate(&self) -> Result<()> {
        self.model.validate().context("PaddleOCR-VL model")?;
        self.projector
            .validate()
            .context("PaddleOCR-VL projector")?;
        if let Some(lora) = &self.lora {
            lora.validate().context("PaddleOCR-VL LoRA")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_source_uses_every_builtin_repository_without_a_lora() {
        let source = PaddleOCRVLQuantizedSource::default();
        assert_eq!(source.model, ComponentSource::Builtin);
        assert_eq!(source.projector, ComponentSource::Builtin);
        assert_eq!(source.lora, None);
        source.validate().unwrap();
    }

    #[test]
    fn a_lora_source_defaults_to_unit_scale() {
        let lora = LoraSource::default();
        assert_eq!(lora.scale, 1.0);
        assert_eq!(lora.adapter, ComponentSource::Builtin);
        lora.validate().unwrap();
    }
}
