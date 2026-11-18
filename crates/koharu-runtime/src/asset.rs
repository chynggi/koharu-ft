//! A model file published outside Hugging Face, such as a GitHub release
//! asset, pinned by its BLAKE3 hash.

use std::path::PathBuf;

use anyhow::{Context, Result, ensure};

use crate::{store, transfer};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Asset {
    url: &'static str,
    blake3: &'static str,
}

impl Asset {
    /// A file is pinned only by a full BLAKE3 hash, which a constant checks as it
    /// builds.
    #[must_use]
    pub const fn new(url: &'static str, blake3: &'static str) -> Self {
        let hash = blake3.as_bytes();
        assert!(hash.len() == 64, "a BLAKE3 hash has 64 digits");
        let mut index = 0;
        while index < hash.len() {
            assert!(
                hash[index].is_ascii_hexdigit() && !hash[index].is_ascii_uppercase(),
                "a BLAKE3 hash is lowercase hexadecimal"
            );
            index += 1;
        }
        Self { url, blake3 }
    }

    /// The file's path, fetched first if the store doesn't have it. A file
    /// whose hash differs is never kept.
    #[tracing::instrument(skip_all, fields(url = self.url))]
    pub async fn resolve(&self) -> Result<PathBuf> {
        let name = self
            .url
            .rsplit('/')
            .next()
            .filter(|name| !name.is_empty())
            .context("an asset's URL ends in its file name")?;
        let path = store::root().join("assets").join(self.blake3).join(name);
        store::put(
            &path,
            path.clone(),
            |path| path.is_file(),
            |folder| async move {
                let staged = folder.join(name);
                transfer::fetch(self.url, &staged).await?;
                let hashed = staged.clone();
                let digest = tokio::task::spawn_blocking(move || -> Result<String> {
                    let mut hasher = blake3::Hasher::new();
                    hasher.update_reader(fs_err::File::open(&hashed)?)?;
                    Ok(hasher.finalize().to_hex().to_string())
                })
                .await??;
                ensure!(
                    digest == self.blake3,
                    "{} has BLAKE3 hash {digest}, not {}",
                    self.url,
                    self.blake3
                );
                Ok(staged)
            },
        )
        .await
    }
}
