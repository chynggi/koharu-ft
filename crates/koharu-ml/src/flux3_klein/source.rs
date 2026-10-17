//! Where the FLUX 3 Image Klein components are loaded from.
//!
//! The shipped repositories stay the default. An override only changes which
//! file is handed to stable-diffusion.cpp or llama.cpp; the architecture is
//! still read from the file's own metadata.

use anyhow::{Context as _, Result};

pub use crate::source::ComponentSource;

/// The five checkpoints an inpainting call is assembled from: three for the
/// image model and two for the prompt upsampler that writes its captions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Flux3KleinSource {
    pub transformer: ComponentSource,
    pub text_encoder: ComponentSource,
    pub vae: ComponentSource,
    pub upsampler: ComponentSource,
    pub upsampler_mmproj: ComponentSource,
}

impl Flux3KleinSource {
    pub fn validate(&self) -> Result<()> {
        self.transformer
            .validate()
            .context("FLUX 3 Image Klein transformer")?;
        self.text_encoder
            .validate()
            .context("FLUX 3 Image Klein text encoder")?;
        self.vae.validate().context("FLUX 3 Image Klein VAE")?;
        self.upsampler
            .validate()
            .context("FLUX 3 Image Klein prompt upsampler")?;
        self.upsampler_mmproj
            .validate()
            .context("FLUX 3 Image Klein upsampler projector")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn the_default_source_uses_every_builtin_repository() {
        let source = Flux3KleinSource::default();
        assert_eq!(source.transformer, ComponentSource::Builtin);
        assert_eq!(source.text_encoder, ComponentSource::Builtin);
        assert_eq!(source.vae, ComponentSource::Builtin);
        assert_eq!(source.upsampler, ComponentSource::Builtin);
        assert_eq!(source.upsampler_mmproj, ComponentSource::Builtin);
        source.validate().unwrap();
    }

    #[test]
    fn an_invalid_override_names_its_component() {
        let source = Flux3KleinSource {
            upsampler_mmproj: ComponentSource::LocalFile(PathBuf::from("relative.gguf")),
            ..Flux3KleinSource::default()
        };
        let error = format!("{:#}", source.validate().unwrap_err());
        assert!(error.contains("upsampler projector"), "{error}");
    }
}
