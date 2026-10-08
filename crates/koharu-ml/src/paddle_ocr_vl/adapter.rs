//! The peft LoRA and the fine-tuned vision tower published alongside the
//! PaddleOCR-VL-1.6 checkpoint.
//!
//! Transformers renames the checkpoint's `visual`, `model`, and `mlp_AR`
//! prefixes while loading, so an adapter trained through Transformers names its
//! weights with the attribute paths (`model.language_model.layers.…`,
//! `model.projector.…`) rather than the checkpoint paths this port loads. The
//! mapping in this module is the whole reason it exists: everything else about
//! an adapter is generic.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use koharu_torch::{Kind, Tensor, nn};
use serde::Deserialize;

use crate::source::ComponentSource;

crate::model_repository!("sorryhyun/paddleocr-vl-1.6-manga-lora" @ "26292839d1469c14212a12a1e01b5b1fe01bff15" {
    ADAPTER_CONFIG = "adapter_config.json",
    ADAPTER_WEIGHTS = "adapter_model.safetensors",
    TOWER_WEIGHTS = "tower.safetensors",
});

/// The peft LoRA and the fine-tuned vision tower published for the
/// PaddleOCR-VL-1.6 manga reader. The tower is not a LoRA: it replaces the
/// vision tower and projector outright, so the two files are only useful
/// together.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PaddleOCRVLAdapter {
    pub config: ComponentSource,
    pub weights: ComponentSource,
    pub tower: ComponentSource,
}

impl PaddleOCRVLAdapter {
    /// Resolves the adapter files and applies them to a loaded model.
    pub(super) async fn apply(&self, vs: &mut nn::VarStore) -> Result<()> {
        let (config, weights, tower) = tokio::try_join!(
            self.config.resolve(ADAPTER_CONFIG.into()),
            self.weights.resolve(ADAPTER_WEIGHTS.into()),
            self.tower.resolve(TOWER_WEIGHTS.into()),
        )
        .context("failed to resolve the PaddleOCR-VL adapter files")?;

        let config = tokio::fs::read_to_string(&config)
            .await
            .with_context(|| format!("failed to read {}", config.display()))?;
        let spec = LoraSpec::from_json(&config)?;
        let weights = Tensor::read_safetensors(&weights)?;
        let tower = Tensor::read_safetensors(&tower)?;

        merge_lora(vs, &weights, &spec).context("failed to merge the LoRA adapter")?;
        replace_tower(vs, &tower).context("failed to replace the vision tower")?;
        Ok(())
    }
}

/// Which half of a LoRA pair a weight is. The pair is merged once both halves
/// have been read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LoraHalf {
    A,
    B,
}

/// The rank and scaling a peft configuration prescribes.
#[derive(Debug, Clone, Copy)]
pub(super) struct LoraSpec {
    pub(super) rank: i64,
    pub(super) alpha: f64,
}

#[derive(Deserialize)]
struct PeftConfig {
    r: i64,
    lora_alpha: f64,
}

impl LoraSpec {
    pub(super) fn from_json(config: &str) -> Result<Self> {
        let config: PeftConfig =
            serde_json::from_str(config).context("failed to parse the peft adapter config")?;
        ensure!(config.r > 0, "LoRA rank must be positive: {}", config.r);
        Ok(Self {
            rank: config.r,
            alpha: config.lora_alpha,
        })
    }
}

const PEFT_PREFIX: &str = "base_model.model.";
const LANGUAGE_MODEL_PREFIX: &str = "model.language_model.";

/// Maps one peft LoRA weight onto the checkpoint name of the linear layer it
/// adapts. Transformers keeps the language model under a `language_model`
/// attribute that the checkpoint does not name, so that one segment is dropped.
pub(super) fn lora_target(key: &str) -> Result<(String, LoraHalf)> {
    let rest = key
        .strip_prefix(PEFT_PREFIX)
        .with_context(|| format!("weight is not from a peft adapter: {key}"))?;
    let (half, rest) = if let Some(rest) = rest.strip_suffix(".lora_A.weight") {
        (LoraHalf::A, rest)
    } else if let Some(rest) = rest.strip_suffix(".lora_B.weight") {
        (LoraHalf::B, rest)
    } else {
        bail!("weight is not a LoRA A/B weight: {key}");
    };
    let module = rest.strip_prefix(LANGUAGE_MODEL_PREFIX).with_context(|| {
        format!("LoRA targets a module outside the language model layers: {key}")
    })?;
    Ok((format!("model.{module}.weight"), half))
}

/// Maps one fine-tuned vision weight onto the checkpoint name it replaces. The
/// checkpoint calls the projector `mlp_AR` where Transformers calls it
/// `projector`.
pub(super) fn tower_target(key: &str) -> Result<String> {
    let rest = key
        .strip_prefix("model.")
        .with_context(|| format!("weight is not from the vision tower checkpoint: {key}"))?;
    if let Some(rest) = rest.strip_prefix("visual.") {
        Ok(format!("visual.{rest}"))
    } else if let Some(rest) = rest.strip_prefix("projector.") {
        Ok(format!("mlp_AR.{rest}"))
    } else {
        bail!("weight is neither a vision tower nor a projector weight: {key}");
    }
}

/// Merges a peft LoRA into the language model layers it adapts, returning how
/// many layers were merged.
///
/// The published halves are float32 while the loaded checkpoint follows the
/// device's precision, so the sum is taken in float32 and cast back to whatever
/// the variable already holds.
pub(super) fn merge_lora(
    vs: &mut nn::VarStore,
    tensors: &[(String, Tensor)],
    spec: &LoraSpec,
) -> Result<usize> {
    let scaling = spec.alpha / spec.rank as f64;

    // The safetensors reader always produces host tensors; the store may live
    // on an accelerator, and the merge is an elementwise sum, so the halves move
    // once, while they are collected.
    let device = vs.device();
    let mut pairs: HashMap<String, (Option<Tensor>, Option<Tensor>)> = HashMap::new();
    for (key, tensor) in tensors {
        let (path, half) = lora_target(key)?;
        let pair = pairs.entry(path.clone()).or_default();
        let slot = match half {
            LoraHalf::A => &mut pair.0,
            LoraHalf::B => &mut pair.1,
        };
        ensure!(slot.is_none(), "the adapter repeats a LoRA weight for {path}");
        *slot = Some(tensor.to_device(device));
    }
    ensure!(!pairs.is_empty(), "the adapter carries no LoRA weights");

    let mut variables = vs.variables();
    for (path, (a, b)) in &pairs {
        let (Some(a), Some(b)) = (a, b) else {
            bail!("the adapter is missing one half of the LoRA pair for {path}");
        };
        let variable = variables.get_mut(path).with_context(|| {
            format!("the adapter adapts a weight this port does not build: {path}")
        })?;
        koharu_torch::no_grad(|| -> Result<()> {
            let delta = b.f_matmul(a)?.f_mul_scalar(scaling)?;
            let updated = variable
                .f_to_kind(Kind::Float)?
                .f_add(&delta)?
                .f_to_kind(variable.kind())?;
            variable.f_copy_(&updated)?;
            Ok(())
        })?;
    }
    Ok(pairs.len())
}

/// Replaces the vision tower and projector weights with their fine-tuned
/// counterparts, returning how many weights were replaced. The checkpoint
/// carries only what the inference path reads, so nothing outside it is
/// touched.
pub(super) fn replace_tower(vs: &mut nn::VarStore, tensors: &[(String, Tensor)]) -> Result<usize> {
    ensure!(!tensors.is_empty(), "the tower checkpoint carries no weights");

    let mut variables = vs.variables();
    for (key, tensor) in tensors {
        let path = tower_target(key)?;
        let variable = variables.get_mut(&path).with_context(|| {
            format!("the tower checkpoint carries a weight this port does not build: {path}")
        })?;
        koharu_torch::no_grad(|| -> Result<()> {
            variable.f_copy_(&tensor.f_to_kind(variable.kind())?)?;
            Ok(())
        })?;
    }
    Ok(tensors.len())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use koharu_torch::{Device, Tensor, nn};

    use super::{LoraHalf, LoraSpec, lora_target, merge_lora, replace_tower, tower_target};

    const Q_PROJ: &str = "base_model.model.model.language_model.layers.0.self_attn.q_proj";

    fn float_tensor(values: &[f32], size: &[i64]) -> Tensor {
        Tensor::from_slice(values).reshape(size)
    }

    /// Creates a variable holding `values`. Variables are trainable, so the
    /// initial copy has to run outside autograd.
    fn var(parent: &nn::Path<'_>, name: &str, values: &[f32], size: &[i64]) -> Tensor {
        let mut tensor = parent.var(name, size, nn::Init::Const(0.0));
        koharu_torch::no_grad(|| tensor.f_copy_(&float_tensor(values, size))).unwrap();
        tensor
    }

    /// The identity on both halves, so the merged delta is the scaling times
    /// `lora_A` itself and the arithmetic stays readable.
    fn q_proj_adapter() -> Vec<(String, Tensor)> {
        vec![
            (
                format!("{Q_PROJ}.lora_A.weight"),
                float_tensor(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]),
            ),
            (
                format!("{Q_PROJ}.lora_B.weight"),
                float_tensor(&[1.0, 0.0, 0.0, 1.0], &[2, 2]),
            ),
        ]
    }

    #[tokio::test]
    #[ignore = "requires the dynamically loaded LibTorch CUDA runtime"]
    async fn merging_a_lora_pair_reaches_a_model_on_the_gpu() -> Result<()> {
        crate::init().await?;
        let gpu = Device::Cuda(0);
        let mut vs = nn::VarStore::new(gpu);
        let root = vs.root();
        let q_proj = &root / "model" / "layers" / 0 / "self_attn" / "q_proj";
        let _ = var(&q_proj, "weight", &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);

        let spec = LoraSpec::from_json(r#"{"r": 2, "lora_alpha": 4}"#)?;
        assert_eq!(merge_lora(&mut vs, &q_proj_adapter(), &spec)?, 1);

        let merged = vs.variables()["model.layers.0.self_attn.q_proj.weight"]
            .shallow_clone()
            .to_device(Device::Cpu);
        assert!(
            merged.allclose(
                &float_tensor(&[3.0, 0.0, 0.0, 0.0, 3.0, 0.0], &[2, 3]),
                1e-6,
                1e-6,
                false
            ),
            "{merged}"
        );
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires the dynamically loaded LibTorch runtime"]
    async fn merging_a_lora_pair_adapts_the_target_weight() -> Result<()> {
        crate::init().await?;
        let mut vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let q_proj = &root / "model" / "layers" / 0 / "self_attn" / "q_proj";
        let _ = var(&q_proj, "weight", &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);

        let spec = LoraSpec::from_json(r#"{"r": 2, "lora_alpha": 4}"#)?;
        assert_eq!(merge_lora(&mut vs, &q_proj_adapter(), &spec)?, 1);

        let merged = vs.variables()["model.layers.0.self_attn.q_proj.weight"].shallow_clone();
        assert!(
            merged.allclose(
                &float_tensor(&[3.0, 0.0, 0.0, 0.0, 3.0, 0.0], &[2, 3]),
                1e-6,
                1e-6,
                false
            ),
            "{merged}"
        );
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires the dynamically loaded LibTorch runtime"]
    async fn a_lora_that_targets_a_module_the_port_does_not_build_is_rejected() -> Result<()> {
        crate::init().await?;
        let mut vs = nn::VarStore::new(Device::Cpu);
        let spec = LoraSpec::from_json(r#"{"r": 2, "lora_alpha": 4}"#)?;
        assert!(merge_lora(&mut vs, &q_proj_adapter(), &spec).is_err());
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires the dynamically loaded LibTorch runtime"]
    async fn an_incomplete_lora_pair_is_rejected() -> Result<()> {
        crate::init().await?;
        let mut vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let _ = var(
            &(&root / "model" / "layers" / 0 / "self_attn" / "q_proj"),
            "weight",
            &[0.0; 6],
            &[2, 3],
        );

        let spec = LoraSpec::from_json(r#"{"r": 2, "lora_alpha": 4}"#)?;
        let only_a = vec![q_proj_adapter().remove(0)];
        assert!(merge_lora(&mut vs, &only_a, &spec).is_err());
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires the dynamically loaded LibTorch runtime"]
    async fn the_tower_checkpoint_replaces_only_the_weights_it_carries() -> Result<()> {
        crate::init().await?;
        let mut vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();
        let layernorm = &root / "visual" / "vision_model" / "post_layernorm";
        let _ = var(&layernorm, "weight", &[1.0, 1.0], &[2]);
        let _ = var(&layernorm, "bias", &[0.0, 0.0], &[2]);
        let _ = var(&(&root / "mlp_AR" / "pre_norm"), "weight", &[5.0, 5.0], &[2]);

        let tensors = vec![(
            "model.visual.vision_model.post_layernorm.weight".to_owned(),
            float_tensor(&[2.0, 3.0], &[2]),
        )];
        assert_eq!(replace_tower(&mut vs, &tensors)?, 1);

        let variables = vs.variables();
        assert!(
            variables["visual.vision_model.post_layernorm.weight"].allclose(
                &float_tensor(&[2.0, 3.0], &[2]),
                0.0,
                0.0,
                false
            ),
            "{}",
            variables["visual.vision_model.post_layernorm.weight"]
        );
        assert_eq!(
            variables["visual.vision_model.post_layernorm.bias"].double_value(&[0]),
            0.0
        );
        assert_eq!(variables["mlp_AR.pre_norm.weight"].double_value(&[0]), 5.0);
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires the dynamically loaded LibTorch runtime"]
    async fn a_tower_weight_the_port_does_not_build_is_rejected() -> Result<()> {
        crate::init().await?;
        let mut vs = nn::VarStore::new(Device::Cpu);
        let tensors = vec![(
            "model.projector.linear_1.weight".to_owned(),
            float_tensor(&[1.0], &[1]),
        )];
        assert!(replace_tower(&mut vs, &tensors).is_err());
        Ok(())
    }

    #[test]
    fn a_lora_a_weight_maps_onto_the_ports_checkpoint_path() {
        let key = "base_model.model.model.language_model.layers.3.self_attn.q_proj.lora_A.weight";
        assert_eq!(
            lora_target(key).unwrap(),
            (
                "model.layers.3.self_attn.q_proj.weight".to_owned(),
                LoraHalf::A
            )
        );
    }

    #[test]
    fn a_lora_b_weight_maps_onto_the_ports_checkpoint_path() {
        let key = "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_B.weight";
        assert_eq!(
            lora_target(key).unwrap(),
            (
                "model.layers.0.mlp.gate_proj.weight".to_owned(),
                LoraHalf::B
            )
        );
    }

    #[test]
    fn a_key_that_is_not_a_lora_weight_is_rejected() {
        let key = "base_model.model.model.language_model.layers.0.self_attn.q_proj.weight";
        assert!(lora_target(key).is_err());
    }

    #[test]
    fn a_vision_tower_weight_maps_onto_the_ports_checkpoint_path() {
        let key = "model.visual.vision_model.encoder.layers.5.self_attn.q_proj.weight";
        assert_eq!(
            tower_target(key).unwrap(),
            "visual.vision_model.encoder.layers.5.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn a_projector_weight_maps_onto_the_ports_mlp_ar_path() {
        assert_eq!(
            tower_target("model.projector.pre_norm.weight").unwrap(),
            "mlp_AR.pre_norm.weight"
        );
    }

    #[test]
    fn a_weight_outside_the_vision_tower_and_projector_is_rejected() {
        assert!(tower_target("model.layers.0.self_attn.q_proj.weight").is_err());
    }

    #[test]
    fn the_lora_rank_and_alpha_are_read_from_the_peft_config() -> Result<()> {
        let config = r#"{
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"]
        }"#;
        let spec = LoraSpec::from_json(config)?;
        assert_eq!(spec.rank, 16);
        assert_eq!(spec.alpha, 32.0);
        Ok(())
    }

    #[test]
    fn a_lora_rank_of_zero_is_rejected() {
        assert!(LoraSpec::from_json(r#"{"r": 0, "lora_alpha": 32}"#).is_err());
    }
}
