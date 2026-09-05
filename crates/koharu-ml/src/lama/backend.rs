//! The two distribution formats of the LaMa weights.
//!
//! safetensors is a state_dict, so `FFCResNetGenerator` has to exist in Rust,
//! while a TorchScript archive carries its architecture inside the file.
//! Preprocessing is identical for both, so `processor` only sees this enum.

use anyhow::Result;
use koharu_torch::Tensor;

use crate::torchscript::TorchScript;

use super::model::Model;

#[derive(Debug)]
pub(super) enum Backend {
    SafeTensors(Box<Model>),
    TorchScript(TorchScript),
}

impl Backend {
    /// `image` is [1,3,H,W] in 0..1 and `mask` is [1,1,H,W] of 0 or 1.
    pub(super) fn forward(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor> {
        match self {
            Self::SafeTensors(m) => Ok(m.forward(image, mask)),
            Self::TorchScript(m) => m.forward(&[image, mask]),
        }
    }
}
