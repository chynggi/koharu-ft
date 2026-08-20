//! Where the FLUX.2 Klein components are loaded from.
//!
//! The shipped repositories stay the default. An override only changes which
//! file is handed to stable-diffusion.cpp; the architecture is still read from
//! the file's own metadata, so a checkpoint the pinned backend does not know
//! fails at context creation rather than being silently reinterpreted.

use std::path::PathBuf;

use anyhow::{Context as _, Result, ensure};
use koharu_runtime::HuggingFaceFile;

/// Where one FLUX.2 Klein component comes from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ComponentSource {
    /// The repository Koharu pins for this component.
    #[default]
    Builtin,
    /// A file already on disk. Nothing is downloaded and the file is only read.
    LocalFile(PathBuf),
    /// Any Hugging Face repository. Without a revision the repository head is
    /// resolved once per process.
    HuggingFace {
        repository: String,
        revision: Option<String>,
        filename: String,
    },
}

impl ComponentSource {
    pub(super) async fn resolve(&self, builtin: HuggingFaceFile<'_>) -> Result<PathBuf> {
        match self {
            Self::Builtin => builtin.resolve().await,
            Self::LocalFile(path) => {
                self.validate()?;
                Ok(path.clone())
            }
            Self::HuggingFace {
                repository,
                revision,
                filename,
            } => {
                self.validate()?;
                match revision {
                    Some(revision) => HuggingFaceFile::pinned(repository, revision, filename),
                    None => HuggingFaceFile::latest(repository, filename),
                }
                .resolve()
                .await
            }
        }
    }

    /// Rejects the mistakes that are visible without touching the network, so a
    /// bad setting is reported when it is saved instead of mid-inference.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Builtin => Ok(()),
            Self::LocalFile(path) => {
                ensure!(
                    path.is_absolute(),
                    "model path must be absolute: {}",
                    path.display()
                );
                ensure!(
                    path.is_file(),
                    "model file does not exist: {}",
                    path.display()
                );
                Ok(())
            }
            Self::HuggingFace {
                repository,
                revision,
                filename,
            } => {
                ensure!(
                    repository.split('/').count() == 2
                        && repository.split('/').all(|part| !part.is_empty()),
                    "Hugging Face repository must be owner/name: {repository}"
                );
                ensure!(!filename.is_empty(), "Hugging Face filename is empty");
                if let Some(revision) = revision {
                    ensure!(
                        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
                        "revision must be a full commit hash: {revision}"
                    );
                }
                Ok(())
            }
        }
    }
}

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
    fn a_relative_local_path_is_rejected() {
        let source = ComponentSource::LocalFile(PathBuf::from("model.gguf"));
        assert!(source.validate().unwrap_err().to_string().contains("absolute"));
    }

    #[test]
    fn a_local_file_must_exist() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("missing.gguf");
        assert!(
            ComponentSource::LocalFile(path)
                .validate()
                .unwrap_err()
                .to_string()
                .contains("does not exist")
        );

        let path = directory.path().join("present.gguf");
        std::fs::write(&path, b"").unwrap();
        ComponentSource::LocalFile(path).validate().unwrap();
    }

    #[test]
    fn a_hugging_face_override_is_checked_before_the_network() {
        let repository = |repository: &str| ComponentSource::HuggingFace {
            repository: repository.to_owned(),
            revision: None,
            filename: "model.gguf".to_owned(),
        };
        assert!(repository("unsloth").validate().is_err());
        assert!(repository("a/b/c").validate().is_err());
        repository("unsloth/FLUX.2-klein-4B-GGUF").validate().unwrap();

        assert!(
            ComponentSource::HuggingFace {
                repository: "unsloth/FLUX.2-klein-4B-GGUF".to_owned(),
                revision: Some("main".to_owned()),
                filename: "model.gguf".to_owned(),
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn the_default_source_uses_every_builtin_repository() {
        let source = Flux2KleinSource::default();
        assert_eq!(source.transformer, ComponentSource::Builtin);
        assert_eq!(source.text_encoder, ComponentSource::Builtin);
        assert_eq!(source.vae, ComponentSource::Builtin);
        source.validate().unwrap();
    }
}
