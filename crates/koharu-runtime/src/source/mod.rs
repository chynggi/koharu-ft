mod archive;
mod hugging_face;
mod pypi;
mod remote;

pub use hugging_face::HuggingFaceFile;
pub use remote::RemoteFile;

pub(crate) use archive::extract;
pub(crate) use pypi::{Platform, wheel};

/// One pinned file, from either supported host.
///
/// Lets a caller hold a default artifact without caring whether it lives in a
/// Hugging Face repository or behind a plain URL.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PinnedFile<'a> {
    HuggingFace(HuggingFaceFile<'a>),
    Remote(RemoteFile<'a>),
}

impl PinnedFile<'_> {
    pub async fn resolve(self) -> anyhow::Result<std::path::PathBuf> {
        match self {
            Self::HuggingFace(file) => file.resolve().await,
            Self::Remote(file) => file.resolve().await,
        }
    }
}

impl<'a> From<HuggingFaceFile<'a>> for PinnedFile<'a> {
    fn from(value: HuggingFaceFile<'a>) -> Self {
        Self::HuggingFace(value)
    }
}

impl<'a> From<RemoteFile<'a>> for PinnedFile<'a> {
    fn from(value: RemoteFile<'a>) -> Self {
        Self::Remote(value)
    }
}
