use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, Result, bail};
use image::{ImageFormat, ImageReader};
use rayon::prelude::*;
use strum::{EnumIter, EnumMessage, EnumString};
use walkdir::WalkDir;

mod pdf;
mod rar;
mod zip;

#[derive(Clone, Copy, EnumIter, EnumMessage, EnumString)]
#[strum(ascii_case_insensitive)]
pub(super) enum Format {
    #[strum(
        serialize = "png",
        serialize = "jpg",
        serialize = "jpeg",
        serialize = "webp"
    )]
    Raster,
    #[strum(serialize = "cbz", serialize = "zip")]
    Zip,
    #[strum(serialize = "rar")]
    Rar,
    #[strum(serialize = "pdf")]
    Pdf,
}

#[derive(Debug)]
pub(super) struct EncodedPage {
    pub(super) name: String,
    pub(super) bytes: Vec<u8>,
}

/// A decoded page ready to be committed as a project page asset.
#[derive(Debug)]
pub struct Page {
    pub name: String,
    pub bytes: Arc<[u8]>,
    pub format: ImageFormat,
    pub width: u32,
    pub height: u32,
}

fn decode(path: &Path, source: EncodedPage) -> Result<Page> {
    let EncodedPage { name, bytes } = source;
    let format = image::guess_format(&bytes).with_context(|| {
        format!(
            "failed to identify imported image {} ({name})",
            path.display()
        )
    })?;
    let (width, height) = ImageReader::with_format(Cursor::new(bytes.as_slice()), format)
        .into_dimensions()
        .with_context(|| {
            format!(
                "failed to read dimensions of imported image {} ({name})",
                path.display()
            )
        })?;
    Ok(Page {
        name,
        bytes: Arc::<[u8]>::from(bytes),
        format,
        width,
        height,
    })
}

/// `Format`을 아는 확장자인지 본다.
pub fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.parse::<Format>().is_ok())
}

/// 가져올 파일 목록으로 펼친다.
///
/// 디렉터리는 재귀적으로 훑어 지원 형식만 취한다. 드래그 & 드롭이 폴더를 그대로
/// 넘기므로, 다이얼로그의 폴더 선택과 같은 규칙을 여기 한 곳에 둔다.
pub fn expand(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut expanded = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for path in paths {
        if path.is_dir() {
            for entry in WalkDir::new(&path)
                .follow_links(false)
                .into_iter()
                .filter_map(|entry| match entry {
                    Ok(entry) if entry.file_type().is_file() => Some(entry.into_path()),
                    Ok(_) => None,
                    Err(error) => {
                        tracing::warn!(%error, "could not inspect an import directory entry");
                        None
                    }
                })
            {
                if is_supported(&entry) && seen.insert(entry.clone()) {
                    expanded.push(entry);
                }
            }
        } else if path.is_file() && is_supported(&path) && seen.insert(path.clone()) {
            expanded.push(path);
        }
    }
    expanded
}

/// Decode image/archive/PDF paths into pages, sorted in reading order.
pub fn import(mut paths: Vec<PathBuf>) -> Result<Vec<Page>> {
    alphanumeric_sort::sort_slice_by_os_str_key(&mut paths, |path| {
        path.file_name().unwrap_or_else(|| path.as_os_str())
    });
    let mut groups = paths
        .into_par_iter()
        .map(|path| -> Result<Vec<Page>> {
            let extension = path
                .extension()
                .and_then(|extension| extension.to_str())
                .and_then(|extension| extension.parse::<Format>().ok());
            let encoded = match extension {
                Some(Format::Raster) => vec![EncodedPage {
                    name: path
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "page".to_owned()),
                    bytes: fs::read(&path)
                        .with_context(|| format!("failed to read {}", path.display()))?,
                }],
                Some(Format::Zip) => zip::extract(&path)?,
                Some(Format::Rar) => rar::extract(&path)?,
                Some(Format::Pdf) => pdf::render(&path)?,
                None => bail!("unsupported page import path {}", path.display()),
            };
            encoded
                .into_iter()
                .map(|source| decode(&path, source))
                .collect()
        })
        .collect::<Result<Vec<_>>>()?;
    let page_count = groups.iter().map(Vec::len).sum();
    let mut pages = Vec::with_capacity(page_count);
    for group in &mut groups {
        pages.append(group);
    }
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level_paths_are_naturally_sorted() {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "koharu-import-order-{}-{timestamp}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("create fixture directory");
        let image = image::RgbaImage::from_pixel(1, 1, image::Rgba([0, 0, 0, 255]));
        let mut encoded = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut encoded, ImageFormat::Png)
            .expect("encode fixture");
        let paths = ["page10.PNG", "page2.png", "page1.png"].map(|name| directory.join(name));
        for path in &paths {
            fs::write(path, encoded.get_ref()).expect("write fixture");
        }

        let pages = import(paths.into()).expect("import fixtures");
        fs::remove_dir_all(&directory).expect("remove fixture directory");
        assert_eq!(
            pages
                .iter()
                .map(|page| page.name.as_str())
                .collect::<Vec<_>>(),
            ["page1.png", "page2.png", "page10.PNG"]
        );
    }

    #[test]
    fn expand_walks_directories_and_keeps_only_supported_files() {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "koharu-import-expand-{}-{timestamp}",
            std::process::id()
        ));
        let nested = directory.join("chapter");
        fs::create_dir_all(&nested).expect("create fixture directory");
        let supported = vec![directory.join("a.png"), nested.join("b.PDF")];
        for path in &supported {
            fs::write(path, b"x").expect("write fixture");
        }
        fs::write(nested.join("notes.txt"), b"x").expect("write fixture");
        // A single file mixed in with the directory is kept; an unsupported one
        // is dropped rather than left for the decoder to reject.
        let single = directory.join("single.png");
        fs::write(&single, b"x").expect("write fixture");

        let mut expanded = expand(vec![
            directory.clone(),
            single.clone(),
            nested.join("notes.txt"),
        ]);
        expanded.sort();
        let mut expected = supported.clone();
        expected.push(single);
        expected.sort();
        assert_eq!(expanded, expected);

        fs::remove_dir_all(&directory).expect("remove fixture directory");
    }
}
