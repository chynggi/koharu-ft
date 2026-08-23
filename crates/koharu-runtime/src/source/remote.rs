//! Fetches a file from an arbitrary URL, pinned by its digest.
//!
//! Some checkpoints are not on Hugging Face — the TorchScript archives IOPaint
//! publishes as GitHub releases, for instance — so they are taken straight from
//! the original repository. A URL has no equivalent of a commit hash, so a
//! BLAKE3 digest plays that role. The cache path is addressed by the digest as
//! well, so an upstream that replaces the file behind a URL cannot poison an
//! existing cache entry.

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};

use crate::{download, store::Store};

/// An immutable file behind a single URL.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RemoteFile<'a> {
    url: &'a str,
    digest: &'a str,
}

impl<'a> RemoteFile<'a> {
    /// `digest` is the BLAKE3 hash of the file contents, written as 64
    /// hexadecimal characters. Either case is accepted; the cache path always
    /// uses the lowercase form, so the same file pinned in either case shares
    /// one cache entry.
    #[must_use]
    pub const fn pinned(url: &'a str, digest: &'a str) -> Self {
        Self { url, digest }
    }

    /// Rejects the mistakes that are visible without touching the network.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.url.starts_with("https://"),
            "model URL must be https: {}",
            self.url
        );
        ensure!(
            self.file_name().is_some(),
            "model URL has no file name: {}",
            self.url
        );
        ensure!(
            self.digest.len() == 64 && self.digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "digest must be 64 hex characters: {}",
            self.digest
        );
        Ok(())
    }

    /// The last path segment of the URL. The query string is discarded.
    fn file_name(&self) -> Option<&'a str> {
        let path = self.url.split(['?', '#']).next()?;
        let name = path.rsplit('/').next()?;
        (!name.is_empty()
            && name != "."
            && name != ".."
            && name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')))
        .then_some(name)
    }

    fn cache_path(&self) -> Result<PathBuf> {
        self.validate()?;
        let name = self.file_name().context("model URL has no file name")?;
        Ok(Store::root()
            .join("remote")
            .join(self.digest.to_ascii_lowercase())
            .join(name))
    }

    #[tracing::instrument(skip_all)]
    pub async fn resolve(self) -> Result<PathBuf> {
        let target = self.cache_path()?;
        let url = self.url.to_owned();
        let digest = self.digest.to_owned();
        Store::file(target, move |stage| async move {
            download::fetch(&url, &stage).await?;
            tokio::task::spawn_blocking(move || verify(&stage, &digest))
                .await
                .context("digest verification task failed")?
                .with_context(|| format!("failed to verify {url}"))
        })
        .await
    }
}

fn verify(path: &Path, expected: &str) -> Result<()> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    hasher
        .update_reader(file)
        .with_context(|| format!("failed to hash {}", path.display()))?;
    let actual = hasher.finalize().to_hex();
    ensure!(
        actual.as_str().eq_ignore_ascii_case(expected),
        "digest mismatch: expected {expected}, got {actual}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_malformed_digest_is_rejected_before_the_network() {
        assert!(
            RemoteFile::pinned("https://example.com/model.pt", "not-a-hash")
                .validate()
                .is_err()
        );
        assert!(
            RemoteFile::pinned("https://example.com/model.pt", &"a".repeat(64))
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn a_non_https_url_is_rejected() {
        assert!(
            RemoteFile::pinned("http://example.com/model.pt", &"a".repeat(64))
                .validate()
                .is_err()
        );
    }

    #[test]
    fn a_file_only_verifies_against_its_own_digest() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), b"koharu").unwrap();
        let digest = blake3::hash(b"koharu").to_hex();

        assert!(verify(file.path(), digest.as_str()).is_ok());
        assert!(verify(file.path(), &"0".repeat(64)).is_err());
    }

    #[test]
    fn the_cache_path_is_addressed_by_the_digest() {
        let digest = "b".repeat(64);
        let file = RemoteFile::pinned(
            "https://github.com/Sanster/models/releases/download/migan/migan_traced.pt",
            &digest,
        );
        let path = file.cache_path().unwrap();

        assert!(path.ends_with(std::path::Path::new(&digest).join("migan_traced.pt")));
    }

    #[test]
    fn the_case_of_the_digest_does_not_split_the_cache() {
        let url = "https://example.com/model.pt";
        let lower = "0123456789abcdef".repeat(4);
        let upper = lower.to_ascii_uppercase();

        assert_eq!(
            RemoteFile::pinned(url, &lower).cache_path().unwrap(),
            RemoteFile::pinned(url, &upper).cache_path().unwrap()
        );
    }
}
