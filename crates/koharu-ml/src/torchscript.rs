//! TorchScript archive execution path.
//!
//! Unlike a safetensors state_dict loaded through `nn::VarStore`, a
//! TorchScript archive carries its architecture inside the file, so no
//! corresponding Rust module definition is needed. Every checkpoint that
//! IOPaint reads with `load_jit_model` is in this format.

use std::path::Path;

use anyhow::{Context as _, Result};
use koharu_torch::{CModule, Device, Tensor};

#[derive(Debug)]
pub struct TorchScript {
    module: CModule,
}

impl TorchScript {
    /// Loads the archive onto the given device and pins it in eval mode.
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let path = path.as_ref();
        let mut module = CModule::load_on_device(path, device)
            .with_context(|| format!("failed to load TorchScript archive {}", path.display()))?;
        module.set_eval();
        Ok(Self { module })
    }

    /// Fails if the input count or shapes differ from what was traced.
    pub fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        self.module
            .forward_ts(inputs)
            .context("TorchScript forward failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(
        windows,
        ignore = "Windows resolves LibTorch only once Runtime::initialize preloads it"
    )]
    fn a_non_archive_file_fails_with_the_path_in_the_message() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("not-an-archive.pt");
        std::fs::write(&path, b"this is not a torchscript archive").unwrap();

        let error = TorchScript::load(&path, koharu_torch::Device::Cpu).unwrap_err();
        assert!(error.to_string().contains("not-an-archive.pt"));
    }

    /// Requires a real checkpoint, so it is excluded from the default run.
    /// Pass `KOHARU_BIG_LAMA_TS` with the path to `big-lama.pt` to run it.
    #[test]
    #[ignore = "requires a local big-lama.pt"]
    fn big_lama_accepts_an_image_and_a_mask() {
        let path = std::env::var("KOHARU_BIG_LAMA_TS").unwrap();
        let device = koharu_torch::Device::Cpu;
        let model = TorchScript::load(&path, device).unwrap();

        let image =
            koharu_torch::Tensor::zeros([1, 3, 512, 512], (koharu_torch::Kind::Float, device));
        let mask =
            koharu_torch::Tensor::zeros([1, 1, 512, 512], (koharu_torch::Kind::Float, device));
        let output = model.forward(&[&image, &mask]).unwrap();

        assert_eq!(output.size(), vec![1, 3, 512, 512]);
    }
}
