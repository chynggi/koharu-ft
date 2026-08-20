//! User-registered GGUF models and llama.cpp runtime settings for the local
//! provider.
//!
//! The built-in catalog in [`super::catalog`] stays a static, download-backed
//! list. This module adds a second source of models that already exist on disk
//! and the layered runtime settings that apply to both. Model files are only
//! ever read; nothing here writes to a `.gguf`.
//!
//! Layering, lowest priority first:
//!
//! ```text
//! GenerationOptions::default()          koharu-ml
//! descriptor.generation                 built-in catalog tuning
//! providers.local.runtime               global runtime settings
//! providers.local.profiles[model]       per-model override
//! pipeline.translation.generation       per-run, edited in the UI
//! ```

use std::{
    collections::BTreeMap,
    num::NonZeroU32,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, bail, ensure};
use koharu_ml::llm::{
    GenerationOptions, KvCacheType, LlamaFlashAttentionType, LoadOptions, MtmdOptions,
};
use serde::{Deserialize, Serialize};
use specta::Type;

use super::catalog::{self, LocalModelDescriptor, ResolvedLocalModel, SupportedLanguages};
use crate::{ModelGeneration, ModelSelection};

const GGUF_EXTENSION: &str = "gguf";

/// `[providers.local]`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct LocalConfig {
    /// Runtime settings applied to every local model.
    pub runtime: LlmRuntimeConfig,
    /// Per-model overrides keyed by model id, layered over `runtime`.
    pub profiles: BTreeMap<String, LlmRuntimeConfig>,
    /// GGUF files registered by the user.
    pub models: Vec<CustomModel>,
}

impl LocalConfig {
    /// Runtime settings for one model, with its profile layered over the global
    /// settings.
    #[must_use]
    pub(crate) fn runtime_for(&self, model: &str) -> LlmRuntimeConfig {
        self.profiles
            .get(model)
            .map_or_else(|| self.runtime.clone(), |profile| self.runtime.overlay(profile))
    }

    /// Rejects configuration that llama.cpp would only fail on later, and ids
    /// that would shadow or duplicate another model.
    pub fn validate(&self) -> Result<()> {
        self.runtime
            .validate()
            .context("invalid local model runtime settings")?;
        for (model, profile) in &self.profiles {
            profile
                .validate()
                .with_context(|| format!("invalid runtime settings for '{model}'"))?;
        }

        let mut seen = std::collections::BTreeSet::new();
        for model in &self.models {
            model.validate()?;
            ensure!(
                !catalog::MODELS
                    .iter()
                    .any(|descriptor| descriptor.id == model.id),
                "'{}' is the id of a built-in model; choose another id",
                model.id
            );
            ensure!(
                seen.insert(model.id.as_str()),
                "duplicate local model id '{}'",
                model.id
            );
        }
        Ok(())
    }
}

/// llama.cpp settings a power user can override. Every field is optional so the
/// layers above can be merged without inventing values for what they do not set.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct LlmRuntimeConfig {
    pub context: Option<ContextMode>,
    /// Upper bound on generated tokens. The per-run generation settings win.
    pub max_output_tokens: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub gpu_layers: Option<GpuLayers>,
    pub n_threads: Option<i32>,
    pub n_threads_batch: Option<i32>,
    pub kv_cache_type_k: Option<KvCacheChoice>,
    pub kv_cache_type_v: Option<KvCacheChoice>,
    pub flash_attention: Option<FlashAttentionMode>,
}

impl LlmRuntimeConfig {
    /// Returns `self` with every value `over` sets replaced.
    #[must_use]
    fn overlay(&self, over: &Self) -> Self {
        Self {
            context: over.context.or(self.context),
            max_output_tokens: over.max_output_tokens.or(self.max_output_tokens),
            n_batch: over.n_batch.or(self.n_batch),
            n_ubatch: over.n_ubatch.or(self.n_ubatch),
            gpu_layers: over.gpu_layers.or(self.gpu_layers),
            n_threads: over.n_threads.or(self.n_threads),
            n_threads_batch: over.n_threads_batch.or(self.n_threads_batch),
            kv_cache_type_k: over.kv_cache_type_k.or(self.kv_cache_type_k),
            kv_cache_type_v: over.kv_cache_type_v.or(self.kv_cache_type_v),
            flash_attention: over.flash_attention.or(self.flash_attention),
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if let Some(context) = self.context {
            context.validate()?;
        }
        if let Some(max_output_tokens) = self.max_output_tokens {
            ensure!(max_output_tokens > 0, "max_output_tokens must be positive");
        }
        if let Some(n_batch) = self.n_batch {
            ensure!(n_batch > 0, "n_batch must be positive");
        }
        if let Some(n_ubatch) = self.n_ubatch {
            ensure!(n_ubatch > 0, "n_ubatch must be positive");
        }
        if let (Some(n_batch), Some(n_ubatch)) = (self.n_batch, self.n_ubatch) {
            ensure!(n_ubatch <= n_batch, "n_ubatch must not exceed n_batch");
        }
        for (label, threads) in [
            ("n_threads", self.n_threads),
            ("n_threads_batch", self.n_threads_batch),
        ] {
            if let Some(threads) = threads {
                ensure!(threads > 0, "{label} must be positive");
            }
        }
        Ok(())
    }

    /// Applies the load-time settings. Leaving `gpu_layers` unset keeps
    /// [`LoadOptions`]'s default, which offloads every layer when a GPU is
    /// selected.
    pub(crate) fn load_options(&self, projector: Option<PathBuf>) -> LoadOptions {
        let defaults = LoadOptions::default();
        LoadOptions {
            gpu_layers: self
                .gpu_layers
                .map_or(defaults.gpu_layers, GpuLayers::layers),
            mtmd: projector.map(MtmdOptions::new),
            ..defaults
        }
    }

    /// Applies the per-inference context and batching settings.
    pub(crate) fn apply(&self, options: &mut GenerationOptions) {
        let context = self.context.unwrap_or_default();
        options.n_ctx = context.fixed();
        options.n_ctx_min = context.minimum();
        options.n_ctx_max = context.maximum();
        options.n_batch = self.n_batch;
        options.n_ubatch = self.n_ubatch;
        options.n_threads = self.n_threads;
        options.n_threads_batch = self.n_threads_batch;
        options.kv_cache_type_k = self.kv_cache_type_k.map(KvCacheChoice::into_llama);
        options.kv_cache_type_v = self.kv_cache_type_v.map(KvCacheChoice::into_llama);
        options.flash_attention = self
            .flash_attention
            .unwrap_or_default()
            .into_llama();
    }
}

/// How the llama.cpp context is sized.
///
/// `Dynamic` is the default and reproduces Koharu's per-inference sizing: the
/// context is exactly what the prepared prompt plus the requested output needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContextMode {
    /// Size the context from the prompt on every call.
    #[default]
    Dynamic,
    /// Always allocate `size` positions. Smaller than a prompt requires is an error.
    Fixed { size: u32 },
    /// Size dynamically, then clamp into the given range.
    Bounded {
        #[serde(default)]
        minimum: Option<u32>,
        #[serde(default)]
        maximum: Option<u32>,
    },
}

impl ContextMode {
    fn validate(self) -> Result<()> {
        match self {
            Self::Dynamic => Ok(()),
            Self::Fixed { size } => {
                ensure!(size > 0, "a fixed context size must be positive");
                Ok(())
            }
            Self::Bounded { minimum, maximum } => {
                if let Some(minimum) = minimum {
                    ensure!(minimum > 0, "a minimum context must be positive");
                }
                if let Some(maximum) = maximum {
                    ensure!(maximum > 0, "a maximum context must be positive");
                }
                if let (Some(minimum), Some(maximum)) = (minimum, maximum) {
                    ensure!(
                        minimum <= maximum,
                        "the minimum context must not exceed the maximum"
                    );
                }
                Ok(())
            }
        }
    }

    fn fixed(self) -> Option<NonZeroU32> {
        match self {
            Self::Fixed { size } => NonZeroU32::new(size),
            Self::Dynamic | Self::Bounded { .. } => None,
        }
    }

    fn minimum(self) -> Option<NonZeroU32> {
        match self {
            Self::Bounded { minimum, .. } => minimum.and_then(NonZeroU32::new),
            Self::Dynamic | Self::Fixed { .. } => None,
        }
    }

    fn maximum(self) -> Option<NonZeroU32> {
        match self {
            Self::Bounded { maximum, .. } => maximum.and_then(NonZeroU32::new),
            Self::Dynamic | Self::Fixed { .. } => None,
        }
    }
}

/// How many model layers to offload to the accelerator.
///
/// `All` is the default and matches Koharu's existing behaviour. A layer count
/// larger than the model has is clamped by llama.cpp. When the selected device
/// is the CPU, no layers are offloaded regardless of this setting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GpuLayers {
    #[default]
    All,
    Custom { layers: u32 },
}

impl GpuLayers {
    fn layers(self) -> u32 {
        match self {
            // Matches koharu-ml's DEFAULT_GPU_LAYERS: more layers than any
            // current model has, so llama.cpp offloads all of them.
            Self::All => 1000,
            Self::Custom { layers } => layers,
        }
    }
}

/// KV cache element types llama.cpp accepts for `type_k` / `type_v`.
///
/// This is deliberately narrower than [`KvCacheType`]: the k-quant and most
/// i-quant types exist in ggml but are not valid KV cache types, and offering
/// them would only produce runtime failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheChoice {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q5_1,
    Q5_0,
    Q4_1,
    Q4_0,
    Iq4Nl,
}

impl KvCacheChoice {
    fn into_llama(self) -> KvCacheType {
        match self {
            Self::F32 => KvCacheType::F32,
            Self::F16 => KvCacheType::F16,
            Self::Bf16 => KvCacheType::BF16,
            Self::Q8_0 => KvCacheType::Q8_0,
            Self::Q5_1 => KvCacheType::Q5_1,
            Self::Q5_0 => KvCacheType::Q5_0,
            Self::Q4_1 => KvCacheType::Q4_1,
            Self::Q4_0 => KvCacheType::Q4_0,
            Self::Iq4Nl => KvCacheType::IQ4_NL,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttentionMode {
    /// Let llama.cpp decide per backend and model.
    #[default]
    Auto,
    On,
    Off,
}

impl FlashAttentionMode {
    fn into_llama(self) -> LlamaFlashAttentionType {
        match self {
            Self::Auto => LlamaFlashAttentionType::Auto,
            Self::On => LlamaFlashAttentionType::Enabled,
            Self::Off => LlamaFlashAttentionType::Disabled,
        }
    }
}

/// A GGUF file the user registered from their own filesystem.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct CustomModel {
    /// Stable identifier stored in the pipeline's model selection.
    pub id: String,
    /// Display name shown in the model picker.
    pub name: String,
    /// Absolute path to the `.gguf` weights.
    pub path: PathBuf,
    /// Optional MTMD projector; present means the model is used with vision.
    #[serde(default)]
    pub projector: Option<PathBuf>,
}

impl CustomModel {
    fn validate(&self) -> Result<()> {
        ensure!(!self.id.trim().is_empty(), "a local model needs an id");
        ensure!(
            self.id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')),
            "local model id '{}' may only contain letters, digits, '-', '_' and '.'",
            self.id
        );
        ensure!(
            !self.name.trim().is_empty(),
            "local model '{}' needs a name",
            self.id
        );
        validate_gguf(&self.path, "weights", &self.id)?;
        if let Some(projector) = &self.projector {
            validate_gguf(projector, "projector", &self.id)?;
        }
        Ok(())
    }
}

fn validate_gguf(path: &Path, role: &str, id: &str) -> Result<()> {
    ensure!(
        path.is_absolute(),
        "local model '{id}' {role} path must be absolute: {}",
        path.display()
    );
    ensure!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case(GGUF_EXTENSION)),
        "local model '{id}' {role} must be a .gguf file: {}",
        path.display()
    );
    ensure!(
        path.is_file(),
        "local model '{id}' {role} does not exist: {}",
        path.display()
    );
    Ok(())
}

/// A local model resolved from either source.
#[derive(Debug)]
pub(crate) enum LocalModel {
    Builtin(LocalModelDescriptor),
    Custom(CustomModel),
}

impl LocalModel {
    pub(crate) fn find(config: &LocalConfig, id: &str) -> Result<Self> {
        if let Some(descriptor) = catalog::MODELS.iter().find(|entry| entry.id == id) {
            return Ok(Self::Builtin(*descriptor));
        }
        if let Some(model) = config.models.iter().find(|entry| entry.id == id) {
            return Ok(Self::Custom(model.clone()));
        }
        bail!("unknown local translator '{id}'")
    }

    pub(crate) async fn resolve(&self, selection: &ModelSelection) -> Result<ResolvedLocalModel> {
        match self {
            Self::Builtin(descriptor) => descriptor.resolve(selection).await,
            Self::Custom(model) => {
                // Registered files are already on disk; re-check because the
                // user may have moved or deleted them since registration.
                validate_gguf(&model.path, "weights", &model.id)?;
                if let Some(projector) = &model.projector {
                    validate_gguf(projector, "projector", &model.id)?;
                }
                Ok(ResolvedLocalModel {
                    model: model.path.clone(),
                    projector: model.projector.clone(),
                })
            }
        }
    }

    /// Sampling defaults. Custom models have none: we do not invent tuning for
    /// a model we know nothing about, so the user's settings govern entirely.
    pub(crate) fn generation(&self) -> ModelGeneration {
        match self {
            Self::Builtin(descriptor) => descriptor.generation,
            Self::Custom(_) => ModelGeneration::default(),
        }
    }

    pub(crate) fn target_languages(&self) -> SupportedLanguages {
        match self {
            Self::Builtin(descriptor) => descriptor.target_languages,
            // Unknown, so nothing is refused up front; a model that cannot
            // produce the target language fails visibly in its output instead.
            Self::Custom(_) => SupportedLanguages::All,
        }
    }

    pub(crate) fn has_projector(&self) -> bool {
        match self {
            Self::Builtin(descriptor) => descriptor.projector.is_some(),
            Self::Custom(model) => model.projector.is_some(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gguf(directory: &Path, name: &str) -> PathBuf {
        let path = directory.join(name);
        std::fs::write(&path, b"GGUF").unwrap();
        path
    }

    fn custom(directory: &Path) -> CustomModel {
        CustomModel {
            id: "custom-qwen-manga".to_owned(),
            name: "Qwen Manga Translator".to_owned(),
            path: gguf(directory, "model.gguf"),
            projector: None,
        }
    }

    #[test]
    fn defaults_deserialize_from_an_empty_table() {
        let config = toml::from_str::<LocalConfig>("").unwrap();

        assert_eq!(config, LocalConfig::default());
        assert!(config.models.is_empty());
        assert_eq!(config.runtime.context, None);
        assert_eq!(config.runtime.gpu_layers, None);
    }

    /// `koharu_config::load` serializes `T::default()` into a `toml::Value`
    /// before merging the stored document over it. An all-`None` default must
    /// survive that round trip or the whole `[providers]` section fails to load.
    #[test]
    fn defaults_serialize_for_the_configuration_merge() {
        let defaults = toml::Value::try_from(LocalConfig::default()).unwrap();
        let stored = toml::from_str::<toml::Value>(
            r#"
                [runtime]
                context = { kind = "bounded", minimum = 4096 }
                flash_attention = "on"

                [[models]]
                id = "custom"
                name = "Custom"
                path = "/models/custom.gguf"
            "#,
        )
        .unwrap();

        let mut merged = defaults;
        deep_merge(&mut merged, stored);
        let config = merged.try_into::<LocalConfig>().unwrap();

        assert_eq!(
            config.runtime.context,
            Some(ContextMode::Bounded {
                minimum: Some(4096),
                maximum: None,
            })
        );
        assert_eq!(config.runtime.flash_attention, Some(FlashAttentionMode::On));
        assert_eq!(config.runtime.n_batch, None);
        assert_eq!(config.models.len(), 1);
    }

    /// Mirrors `koharu_config`'s merge: tables recurse, everything else replaces.
    fn deep_merge(base: &mut toml::Value, update: toml::Value) {
        match (base, update) {
            (toml::Value::Table(base), toml::Value::Table(update)) => {
                for (key, value) in update {
                    match base.get_mut(&key) {
                        Some(base) => deep_merge(base, value),
                        None => {
                            base.insert(key, value);
                        }
                    }
                }
            }
            (base, update) => *base = update,
        }
    }

    #[test]
    fn defaults_reproduce_the_dynamic_context_behaviour() {
        let mut options = GenerationOptions::default();
        LlmRuntimeConfig::default().apply(&mut options);

        assert_eq!(options.n_ctx, None);
        assert_eq!(options.n_ctx_min, None);
        assert_eq!(options.n_ctx_max, None);
        assert_eq!(options.n_batch, None);
        assert_eq!(options.n_ubatch, None);
        assert_eq!(options.kv_cache_type_k, None);
        assert_eq!(options.flash_attention, LlamaFlashAttentionType::Auto);
        assert_eq!(
            LlmRuntimeConfig::default().load_options(None).gpu_layers,
            LoadOptions::default().gpu_layers
        );
    }

    #[test]
    fn context_modes_map_to_generation_options() {
        let mut fixed = GenerationOptions::default();
        LlmRuntimeConfig {
            context: Some(ContextMode::Fixed { size: 8192 }),
            ..Default::default()
        }
        .apply(&mut fixed);
        assert_eq!(fixed.n_ctx, NonZeroU32::new(8192));
        assert_eq!(fixed.n_ctx_min, None);

        let mut bounded = GenerationOptions::default();
        LlmRuntimeConfig {
            context: Some(ContextMode::Bounded {
                minimum: Some(4096),
                maximum: Some(32768),
            }),
            ..Default::default()
        }
        .apply(&mut bounded);
        assert_eq!(bounded.n_ctx, None);
        assert_eq!(bounded.n_ctx_min, NonZeroU32::new(4096));
        assert_eq!(bounded.n_ctx_max, NonZeroU32::new(32768));
    }

    #[test]
    fn runtime_settings_reach_llama_types() {
        let mut options = GenerationOptions::default();
        LlmRuntimeConfig {
            n_batch: Some(2048),
            n_ubatch: Some(512),
            n_threads: Some(8),
            kv_cache_type_k: Some(KvCacheChoice::Q8_0),
            kv_cache_type_v: Some(KvCacheChoice::Q4_0),
            flash_attention: Some(FlashAttentionMode::On),
            ..Default::default()
        }
        .apply(&mut options);

        assert_eq!(options.n_batch, Some(2048));
        assert_eq!(options.n_ubatch, Some(512));
        assert_eq!(options.n_threads, Some(8));
        assert_eq!(options.kv_cache_type_k, Some(KvCacheType::Q8_0));
        assert_eq!(options.kv_cache_type_v, Some(KvCacheType::Q4_0));
        assert_eq!(options.flash_attention, LlamaFlashAttentionType::Enabled);
    }

    #[test]
    fn gpu_layers_select_a_layer_count() {
        assert_eq!(
            LlmRuntimeConfig {
                gpu_layers: Some(GpuLayers::Custom { layers: 24 }),
                ..Default::default()
            }
            .load_options(None)
            .gpu_layers,
            24
        );
        assert_eq!(
            LlmRuntimeConfig {
                gpu_layers: Some(GpuLayers::All),
                ..Default::default()
            }
            .load_options(None)
            .gpu_layers,
            1000
        );
    }

    #[test]
    fn per_model_profile_overrides_global_runtime() {
        let config = LocalConfig {
            runtime: LlmRuntimeConfig {
                gpu_layers: Some(GpuLayers::All),
                n_batch: Some(1024),
                ..Default::default()
            },
            profiles: BTreeMap::from([(
                "custom-qwen-manga".to_owned(),
                LlmRuntimeConfig {
                    gpu_layers: Some(GpuLayers::Custom { layers: 40 }),
                    ..Default::default()
                },
            )]),
            models: Vec::new(),
        };

        let resolved = config.runtime_for("custom-qwen-manga");
        assert_eq!(resolved.gpu_layers, Some(GpuLayers::Custom { layers: 40 }));
        // Untouched by the profile, so the global value survives.
        assert_eq!(resolved.n_batch, Some(1024));

        let other = config.runtime_for("gemma4-12b-it");
        assert_eq!(other.gpu_layers, Some(GpuLayers::All));
    }

    #[test]
    fn local_config_round_trips_through_toml() {
        let directory = tempfile::tempdir().unwrap();
        let config = LocalConfig {
            runtime: LlmRuntimeConfig {
                context: Some(ContextMode::Bounded {
                    minimum: Some(4096),
                    maximum: Some(32768),
                }),
                max_output_tokens: Some(2048),
                kv_cache_type_k: Some(KvCacheChoice::Q8_0),
                flash_attention: Some(FlashAttentionMode::On),
                ..Default::default()
            },
            profiles: BTreeMap::from([(
                "custom-qwen-manga".to_owned(),
                LlmRuntimeConfig {
                    gpu_layers: Some(GpuLayers::Custom { layers: 40 }),
                    ..Default::default()
                },
            )]),
            models: vec![custom(directory.path())],
        };

        let document = toml::to_string(&config).unwrap();
        let restored = toml::from_str::<LocalConfig>(&document).unwrap();

        assert_eq!(restored, config);
    }

    #[test]
    fn custom_model_cannot_shadow_a_builtin_id() {
        let directory = tempfile::tempdir().unwrap();
        let config = LocalConfig {
            models: vec![CustomModel {
                id: catalog::MODELS[0].id.to_owned(),
                ..custom(directory.path())
            }],
            ..Default::default()
        };

        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("built-in model"), "{error}");
    }

    #[test]
    fn duplicate_custom_model_ids_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let model = custom(directory.path());
        let config = LocalConfig {
            models: vec![model.clone(), model],
            ..Default::default()
        };

        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
    }

    #[test]
    fn custom_model_requires_an_existing_gguf_file() {
        let directory = tempfile::tempdir().unwrap();

        let missing = LocalConfig {
            models: vec![CustomModel {
                path: directory.path().join("absent.gguf"),
                ..custom(directory.path())
            }],
            ..Default::default()
        };
        assert!(
            missing
                .validate()
                .unwrap_err()
                .to_string()
                .contains("does not exist")
        );

        let wrong_extension = LocalConfig {
            models: vec![CustomModel {
                path: gguf(directory.path(), "model.safetensors"),
                ..custom(directory.path())
            }],
            ..Default::default()
        };
        assert!(
            wrong_extension
                .validate()
                .unwrap_err()
                .to_string()
                .contains(".gguf")
        );
    }

    #[test]
    fn invalid_runtime_settings_are_rejected() {
        assert!(
            LlmRuntimeConfig {
                context: Some(ContextMode::Bounded {
                    minimum: Some(8192),
                    maximum: Some(4096),
                }),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            LlmRuntimeConfig {
                n_batch: Some(256),
                n_ubatch: Some(512),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            LlmRuntimeConfig {
                n_threads: Some(0),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn custom_models_resolve_without_downloading() {
        let directory = tempfile::tempdir().unwrap();
        let config = LocalConfig {
            models: vec![custom(directory.path())],
            ..Default::default()
        };

        let model = LocalModel::find(&config, "custom-qwen-manga").unwrap();
        assert!(matches!(model, LocalModel::Custom(_)));
        assert!(!model.has_projector());
        assert_eq!(model.generation(), ModelGeneration::default());

        assert!(LocalModel::find(&config, "nope").is_err());
        assert!(matches!(
            LocalModel::find(&config, catalog::MODELS[0].id).unwrap(),
            LocalModel::Builtin(_)
        ));
    }
}
