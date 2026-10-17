//! The official FLUX 3 Image Klein prompt upsampler.
//!
//! The i2i message layout is `upsample(prompt, images)` in the model card at:
//! https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/337337621b8526e120f2275d250242f4a6bedd75/README.md

use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context as _, Result, bail, ensure};
use image::DynamicImage;

use super::{Flux3KleinSource, SYSTEM_PROMPTS, UPSAMPLER_PROJECTOR, UPSAMPLER_WEIGHTS};
use crate::{
    Device,
    llm::{
        ChatMessage, FinishReason, GenerationOptions, Input, Llm, LoadOptions, MtmdOptions,
        media_marker,
    },
};

/// `max_new_tokens` in the model card's `upsample`.
const MAX_NEW_TOKENS: usize = 3584;

/// Where the upsampler lives, never the loaded model. Each `caption` call loads
/// it, captions every image and drops it before returning, so the accelerator
/// is free for FLUX 3 again; the weights stay in the OS file cache between
/// pages.
#[derive(Clone, Debug)]
pub struct Flux3PromptUpsampler {
    device: Device,
    model: PathBuf,
    projector: PathBuf,
    system_prompt: String,
}

impl Flux3PromptUpsampler {
    /// Resolves (and downloads) the upsampler files and reads the i2i system
    /// prompt. Nothing is loaded onto the accelerator.
    pub async fn load(device: Device, source: &Flux3KleinSource) -> Result<Self> {
        source.validate()?;
        let (model, projector, prompts) = tokio::try_join!(
            source.upsampler.resolve(UPSAMPLER_WEIGHTS.into()),
            source.upsampler_mmproj.resolve(UPSAMPLER_PROJECTOR.into()),
            SYSTEM_PROMPTS.resolve(),
        )
        .context("failed to resolve the FLUX 3 prompt upsampler")?;
        let system_prompt = i2i_system_prompt(
            &std::fs::read_to_string(&prompts).context("failed to read system_prompts.json")?,
        )?;
        Ok(Self {
            device,
            model,
            projector,
            system_prompt,
        })
    }

    /// One caption per image, in order. The model holds the accelerator only
    /// for the duration of this call.
    pub async fn caption(&self, images: Vec<DynamicImage>, request: String) -> Result<Vec<String>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let llm = Llm::load_with_options(
            self.device.clone(),
            self.model.clone(),
            LoadOptions {
                mtmd: Some(MtmdOptions::new(self.projector.clone())),
                ..LoadOptions::default()
            },
        )
        .await
        .context("failed to load the FLUX 3 prompt upsampler")?;
        ensure!(
            llm.capabilities().vision,
            "the FLUX 3 prompt upsampler projector does not support images"
        );
        let system_prompt = self.system_prompt.clone();
        tokio_rayon::spawn(move || {
            images
                .iter()
                .enumerate()
                .map(|(index, image)| {
                    caption_one(&llm, &system_prompt, &request, image)
                        .with_context(|| format!("failed to caption tile {index}"))
                })
                .collect::<Result<Vec<_>>>()
        })
        .await
    }
}

fn caption_one(
    llm: &Llm,
    system_prompt: &str,
    request: &str,
    image: &DynamicImage,
) -> Result<String> {
    let prompt = llm.render_chat_prompt(&messages(system_prompt, request))?;
    let generation = llm.inference(
        &Input::new(&prompt).with_image(image),
        &GenerationOptions {
            max_tokens: MAX_NEW_TOKENS,
            // `do_sample=False` in the model card.
            temperature: 0.0,
            add_special: true,
            ..GenerationOptions::default()
        },
    )?;
    if generation.finish_reason == FinishReason::Length {
        bail!("the caption reached {MAX_NEW_TOKENS} tokens without finishing");
    }
    let caption = generation.text.trim().to_owned();
    ensure!(
        !caption.is_empty(),
        "the prompt upsampler returned an empty caption"
    );
    Ok(caption)
}

/// The model card's i2i conversation: the system prompt, then one user turn
/// holding the reference image followed by the request.
fn messages(system_prompt: &str, request: &str) -> [ChatMessage; 2] {
    [
        ChatMessage::system(system_prompt),
        ChatMessage::user(format!("{}{}", media_marker(), user_text(request))),
    ]
}

fn user_text(request: &str) -> String {
    format!(
        "Reference images follow in order (the first image, the second image, ...).\nUser request: {request}\nOutput the caption."
    )
}

fn i2i_system_prompt(json: &str) -> Result<String> {
    let prompts: HashMap<String, String> =
        serde_json::from_str(json).context("system_prompts.json is not a map of strings")?;
    prompts
        .get("i2i")
        .cloned()
        .context("system_prompts.json has no i2i prompt")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_user_turn_reproduces_the_model_card_template() {
        assert_eq!(
            user_text("Erase the lettering."),
            "Reference images follow in order (the first image, the second image, ...).\nUser request: Erase the lettering.\nOutput the caption."
        );
    }

    #[test]
    fn the_image_precedes_the_request_in_the_user_turn() {
        let [system, user] = messages("SYSTEM", "Erase the lettering.");
        assert_eq!(system, ChatMessage::system("SYSTEM"));
        assert_eq!(
            user,
            ChatMessage::user(format!(
                "{}{}",
                media_marker(),
                user_text("Erase the lettering.")
            ))
        );
    }

    #[test]
    fn the_i2i_prompt_is_selected_from_the_published_keys() {
        let json = r#"{"t2i":"A","i2i":"B","t2i_bbox":"C","i2i_bbox":"D"}"#;
        assert_eq!(i2i_system_prompt(json).unwrap(), "B");
        assert!(i2i_system_prompt(r#"{"t2i":"A"}"#).is_err());
    }

    /// Requires the llama.cpp runtime and downloads the upsampler (~6 GB).
    #[tokio::test]
    #[ignore = "requires the dynamically loaded llama.cpp runtime and the upsampler checkpoints"]
    async fn the_upsampler_captions_a_crop() {
        crate::init().await.unwrap();
        let upsampler =
            Flux3PromptUpsampler::load(crate::device(false), &Flux3KleinSource::default())
                .await
                .unwrap();
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            256,
            256,
            image::Rgb([240, 240, 240]),
        ));
        let captions = upsampler
            .caption(vec![image], super::super::DEFAULT_REQUEST.to_owned())
            .await
            .unwrap();
        assert_eq!(captions.len(), 1);
        assert!(!captions[0].is_empty());
    }
}
