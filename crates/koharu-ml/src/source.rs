//! Where one model component is loaded from.
//!
//! Every model that ships a pinned default but accepts an override names its
//! files through the same three-or-four ways, so the type lives here rather
//! than beside any one of them. An override only changes which file is handed
//! to the backend; the architecture is still read from the file's own
//! metadata, so a checkpoint the pinned backend does not know fails at load
//! rather than being silently reinterpreted.

use std::path::PathBuf;

use anyhow::{Result, ensure};
use koharu_runtime::{HuggingFaceFile, PinnedFile};

/// Where one model component comes from.
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
    /// An arbitrary URL, pinned by digest. The digest is required: without it
    /// a changed upstream file would silently poison the cache.
    Url { url: String, digest: String },
}

impl ComponentSource {
    pub async fn resolve(&self, builtin: PinnedFile<'_>) -> Result<PathBuf> {
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
            Self::Url { url, digest } => {
                self.validate()?;
                koharu_runtime::RemoteFile::pinned(url, digest).resolve().await
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
            Self::Url { url, digest } => {
                koharu_runtime::RemoteFile::pinned(url, digest).validate()
            }
        }
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
    fn a_url_override_requires_a_digest() {
        let source = ComponentSource::Url {
            url: "https://example.com/model.pt".to_owned(),
            digest: "short".to_owned(),
        };
        assert!(
            source
                .validate()
                .unwrap_err()
                .to_string()
                .contains("digest must be 64 hex characters")
        );

        ComponentSource::Url {
            url: "https://example.com/model.pt".to_owned(),
            digest: "a".repeat(64),
        }
        .validate()
        .unwrap();
    }
}
