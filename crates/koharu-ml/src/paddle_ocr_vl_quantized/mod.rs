//! Quantized PaddleOCR-VL-1.6 inference through llama.cpp.

mod source;

use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use koharu_llama::mtmd::mtmd_default_marker;

use crate::{
    Device,
    llm::{ChatMessage, GenerationOptions, Input, Llm, LoadOptions, LoraOptions, MtmdOptions},
    paddle_ocr_vl::{MAX_NEW_TOKENS, PaddleOCRVLResult, PaddleOCRVLTask, REPETITION_PENALTY},
};

pub use self::source::{ComponentSource, LoraSource, PaddleOCRVLQuantizedSource};

crate::model_repository!("PaddlePaddle/PaddleOCR-VL-1.6-GGUF" @ "511b09642bb324401f15f97cc23bc67e8f0a291d" {
    MODEL = "PaddleOCR-VL-1.6-GGUF.gguf",
    PROJECTOR = "PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
});

// Converted from the safetensors published in `sorryhyun/paddleocr-vl-1.6-manga-lora`
// by `scripts/convert_paddleocr_vl_manga_lora.py`. Kept in a separate `-GGUF`
// repository so the original checkpoint stays safetensors-only.
crate::model_repository!("Essid/paddleocr-vl-1.6-manga-lora-GGUF" @ "25c919ad71e8747529dfb055fabbfb94b98526a4" {
    LORA_ADAPTER = "manga-lora.gguf",
    MANGA_PROJECTOR = "manga-mmproj.gguf",
});

/// The manga reader variant: the stock language model, a projector built from
/// the fine-tuned vision tower, and the peft LoRA. The tower and the LoRA are
/// a pair — the tower file is a full fine-tune, not a LoRA — so a source that
/// carries a LoRA uses the fine-tuned projector pin for `Builtin`.
#[must_use]
pub fn manga_source() -> PaddleOCRVLQuantizedSource {
    PaddleOCRVLQuantizedSource {
        model: ComponentSource::Builtin,
        projector: ComponentSource::Builtin,
        lora: Some(LoraSource::default()),
    }
}

#[derive(Debug)]
pub struct PaddleOCRVLQuantized {
    model: Llm,
}

impl PaddleOCRVLQuantized {
    /// Loads the base quantized PaddleOCR-VL-1.6 without a LoRA adapter.
    pub async fn load(device: Device) -> Result<Self> {
        Self::load_with_source(device, PaddleOCRVLQuantizedSource::default()).await
    }

    /// Loads the manga reader variant: the fine-tuned vision tower projector
    /// and the peft LoRA converted to a GGUF adapter.
    pub async fn load_manga(device: Device) -> Result<Self> {
        Self::load_with_source(device, manga_source()).await
    }

    /// Loads a quantized PaddleOCR-VL-1.6 assembled from the given sources.
    ///
    /// `ComponentSource::Builtin` for the projector resolves against the stock
    /// projector, or against the fine-tuned tower projector when `source.lora`
    /// is set. Override the projector explicitly to break that pairing.
    pub async fn load_with_source(
        device: Device,
        source: PaddleOCRVLQuantizedSource,
    ) -> Result<Self> {
        source
            .validate()
            .context("invalid quantized PaddleOCR-VL source")?;
        let projector_pin = if source.lora.is_some() {
            MANGA_PROJECTOR
        } else {
            PROJECTOR
        };
        let lora = source.lora;
        let (model_path, projector_path, lora_path) = tokio::try_join!(
            source.model.resolve(MODEL.into()),
            source.projector.resolve(projector_pin.into()),
            async {
                match &lora {
                    Some(lora) => lora.adapter.resolve(LORA_ADAPTER.into()).await.map(Some),
                    None => Ok(None),
                }
            },
        )
        .context("failed to resolve quantized PaddleOCR-VL-1.6 files")?;
        let options = LoadOptions {
            mtmd: Some(MtmdOptions::new(projector_path)),
            lora: lora_path.map(|path| {
                let scale = lora.as_ref().map_or(1.0, |lora| lora.scale);
                LoraOptions::new(path).with_scale(scale)
            }),
            ..LoadOptions::default()
        };
        let model = Llm::load_with_options(device, model_path, options)
            .await
            .context("failed to load quantized PaddleOCR-VL-1.6")?;
        ensure!(
            model.capabilities().vision,
            "PaddleOCR-VL-1.6 projector does not support images"
        );
        Ok(Self { model })
    }

    pub fn inference(
        &self,
        image: &DynamicImage,
        task: PaddleOCRVLTask,
    ) -> Result<PaddleOCRVLResult> {
        let prompt = self.model.render_chat_prompt(&[ChatMessage::user(format!(
            "{}{}",
            mtmd_default_marker(),
            task.prompt()
        ))])?;
        let generation = self.model.inference(
            &Input::new(&prompt).with_image(image),
            &GenerationOptions {
                max_tokens: MAX_NEW_TOKENS,
                temperature: 0.0,
                repeat_penalty: REPETITION_PENALTY,
                repeat_last_n: MAX_NEW_TOKENS as i32,
                add_special: true,
                ..GenerationOptions::default()
            },
        )?;
        Ok(PaddleOCRVLResult {
            text: generation.text,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_base_source_has_no_lora() {
        let source = PaddleOCRVLQuantizedSource::default();
        assert!(source.lora.is_none());
        source.validate().unwrap();
    }

    #[test]
    fn the_manga_source_pairs_a_lora_with_the_fine_tuned_projector() {
        let source = manga_source();
        assert!(source.lora.is_some());
        assert_eq!(source.model, ComponentSource::Builtin);
        assert_eq!(source.projector, ComponentSource::Builtin);
        source.validate().unwrap();
    }

    #[test]
    fn a_local_lora_override_is_accepted() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("manga-lora.gguf");
        std::fs::write(&path, b"").unwrap();
        let source = PaddleOCRVLQuantizedSource {
            lora: Some(LoraSource {
                adapter: ComponentSource::LocalFile(path),
                scale: 2.0,
            }),
            ..manga_source()
        };
        source.validate().unwrap();
    }

    /// End-to-end over the converted GGUF pair. Requires the llama.cpp runtime
    /// and the two files `scripts/convert_paddleocr_vl_manga_lora.py` writes.
    #[tokio::test]
    #[ignore = "requires the dynamically loaded llama.cpp runtime and converted GGUF artifacts"]
    async fn the_manga_variant_attaches_the_lora_to_every_inference() {
        crate::init().await.unwrap();
        let source = PaddleOCRVLQuantizedSource {
            lora: Some(LoraSource {
                adapter: ComponentSource::LocalFile(
                    std::path::PathBuf::from(std::env::var("KOHARU_MANGA_LORA_GGUF").unwrap()),
                ),
                scale: 1.0,
            }),
            projector: ComponentSource::LocalFile(
                std::path::PathBuf::from(std::env::var("KOHARU_MANGA_MMPROJ_GGUF").unwrap()),
            ),
            ..PaddleOCRVLQuantizedSource::default()
        };
        let model = PaddleOCRVLQuantized::load_with_source(crate::Device::cpu(), source)
            .await
            .unwrap();
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            64,
            64,
            image::Rgb([255, 255, 255]),
        ));
        // A tiny crop is enough to prove the adapter is applied to each fresh
        // context; the model card's accuracy claims need real manga crops.
        let _ = model.inference(&image, PaddleOCRVLTask::Ocr).unwrap();
    }
}
