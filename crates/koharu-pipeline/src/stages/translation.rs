use anyhow::Result;
use async_trait::async_trait;
use koharu_scene::{Authored, LanguageTag, Origin, SourceText, Translation};
use koharu_translator::{TranslationRequest, Translator};

use crate::TranslationConfig;

use super::{StageInput, StageProcessor, finish, generation};

const PRODUCER: &str = "dev.koharu.pipeline.translation";

pub(super) struct Processor {
    config: TranslationConfig,
    translator: Translator,
}

impl Processor {
    pub(super) fn new(config: TranslationConfig, translator: Translator) -> Self {
        Self { config, translator }
    }

    pub(super) fn batch_size(&self) -> usize {
        if self.config.combine_pages {
            self.config.max_pages_per_request as usize
        } else {
            1
        }
    }

    pub(super) async fn process_batch(
        &self,
        inputs: Vec<StageInput>,
    ) -> Result<koharu_scene::Patch> {
        let first = inputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("translation batch cannot be empty"))?;
        let mut targets = Vec::new();
        let mut page_ranges = Vec::new();
        for (page_number, input) in inputs.iter().enumerate() {
            let start = targets.len();
            if let Some(group) = input.scene.page(input.page)?.text_group()? {
                for layer in group.text_layers()? {
                    if !input.contains_entity(layer.id())? {
                        continue;
                    }
                    let content = layer.content()?;
                    let Some(source) = content.source()? else {
                        continue;
                    };
                    if !source.text.value.trim().is_empty() {
                        targets.push((content.id(), source.text.value));
                    }
                }
            }
            page_ranges.push((page_number + 1, start, targets.len()));
        }
        let mut request = TranslationRequest::new(
            targets.iter().map(|(_, source)| source.clone()),
            self.config.target_language,
        );
        if inputs.len() > 1 {
            let boundaries = page_ranges
                .iter()
                .map(|(page, start, end)| {
                    if start == end {
                        format!("Page {page}: no source segments.")
                    } else {
                        format!("Page {page}: segment IDs {start} through {}.", end - 1)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            let mut instructions = format!(
                "The input segments are from the listed manga pages. Keep character names, terminology, voices, and plot continuity consistent across the entire batch. Page boundaries are:\n{boundaries}\n"
            );
            if self
                .translator
                .supports_vision(&self.config.model, &self.config.generation)
            {
                instructions.push_str(
                    "No page images are attached for this multi-page request; use only the supplied text segments.\n",
                );
            }
            if let Some(user_instructions) = self.config.instructions.as_deref() {
                instructions.push_str("\nAdditional user guidance:\n");
                instructions.push_str(user_instructions);
            }
            request = request.with_instructions(instructions);
        } else if let Some(instructions) = self.config.instructions.as_deref() {
            request = request.with_instructions(instructions);
        }
        // 한 페이지 요청에만 원본 이미지를 붙인다. 여러 페이지를 한 요청에 묶으면
        // 어느 페이지의 이미자인지 말할 수 없으므로 텍스트만 보낸다.
        if inputs.len() == 1
            && self
                .translator
                .supports_vision(&self.config.model, &self.config.generation)
            && let Some(image) = first.images.get(&first.scene, first.page, "source").await?
        {
            request = request.with_image(image);
        }
        let (provider, translated) = self
            .translator
            .translate(&self.config.model, self.config.generation, request)
            .await?;
        let language = LanguageTag::new(self.config.target_language.tag())?;
        let generated = generation(PRODUCER, provider)?;
        let mut edit = first.scene.edit_as(generated.clone());
        for (entity, _) in &targets {
            edit.observe::<SourceText>(*entity)?;
            edit.observe::<Translation>(*entity)?;
        }
        for ((entity, source), text) in targets.into_iter().zip(translated) {
            if first
                .scene
                .component::<Translation>(entity)?
                .is_some_and(|value| matches!(value.text.origin, Origin::User))
            {
                continue;
            }
            let text = if source.trim() == "\u{2026}" {
                "\u{2026}".to_owned()
            } else {
                text
            };
            edit.set(
                entity,
                &Translation {
                    text: Authored::generated(text, generated.clone()),
                    language: Some(language.clone()),
                },
            )?;
        }
        finish(edit)
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        Translator::model(&self.config.model)
    }

    fn unload(&self) -> bool {
        self.translator.unload()
    }

    async fn load(&self) -> Result<()> {
        self.translator.load_model(&self.config.model).await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        self.process_batch(vec![input]).await
    }
}
