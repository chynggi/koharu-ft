//! The two distribution formats of the LaMa weights.
//!
//! safetensors is a state_dict, so `FFCResNetGenerator` has to exist in Rust,
//! while a TorchScript archive carries its architecture inside the file.
//! Preprocessing is identical for both, so `processor` only sees this trait.

use anyhow::Result;
use koharu_torch::Tensor;

use crate::torchscript::TorchScript;

use super::model::Model;

pub(super) trait Backend: std::fmt::Debug + Send {
    /// `image` is [1,3,H,W] in 0..1 and `mask` is [1,1,H,W] of 0 or 1.
    fn forward(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor>;
}

impl Backend for Model {
    fn forward(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor> {
        Ok(Model::forward(self, image, mask))
    }
}

impl Backend for TorchScript {
    fn forward(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor> {
        TorchScript::forward(self, &[image, mask])
    }
}
