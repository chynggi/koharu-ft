//! Where the FLUX.2 Klein components are loaded from.
//!
//! The shipped repositories stay the default. An override only changes which
//! file is handed to stable-diffusion.cpp; the architecture is still read from
//! the file's own metadata, so a checkpoint the pinned backend does not know
//! fails at context creation rather than being silently reinterpreted.

use anyhow::{Context as _, Result};

pub use crate::source::ComponentSource;

/// The three checkpoints a FLUX.2 Klein context is assembled from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Flux2KleinSource {
    pub transformer: ComponentSource,
    pub text_encoder: ComponentSource,
    pub vae: ComponentSource,
}

impl Flux2KleinSource {
    pub fn validate(&self) -> Result<()> {
        self.transformer
            .validate()
            .context("FLUX.2 Klein transformer")?;
        self.text_encoder
            .validate()
            .context("FLUX.2 Klein text encoder")?;
        self.vae.validate().context("FLUX.2 Klein VAE")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_source_uses_every_builtin_repository() {
        let source = Flux2KleinSource::default();
        assert_eq!(source.transformer, ComponentSource::Builtin);
        assert_eq!(source.text_encoder, ComponentSource::Builtin);
        assert_eq!(source.vae, ComponentSource::Builtin);
        source.validate().unwrap();
    }
}
