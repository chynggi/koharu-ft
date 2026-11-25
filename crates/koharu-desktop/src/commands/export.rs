//! Export: pages as PNG or PSD files in a folder, and as a CBZ of PNGs or a
//! PDF there, named by a filename pattern.

use std::{
    collections::HashSet,
    io::{Cursor, Write as _},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::Context as _;
use fs_err::File;
use image::{ImageFormat, RgbaImage};
use koharu_psd::Labels;
use koharu_scene::{PageId, Session};
use krilla::{Document, geom::Size, image::Image, page::PageSettings};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{State, Window, ipc::Channel};
use tauri_runtime_cef::CefRuntime;

use self::naming::{Names, Template};
use super::{Result, import::PDF_SCALE};
use crate::app::App;

mod naming;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Png,
    Psd,
    Cbz,
    Pdf,
}

#[derive(Clone, Debug, Deserialize, Type)]
pub struct ExportOptions {
    pub formats: Vec<ExportFormat>,
    /// A file's name without its extension: `{index}`, `{index:04}`, and
    /// `{label}`, the page's name, between literal text.
    pub pattern: String,
    /// PNG and PSD files each in a folder of their own.
    pub subfolders: bool,
}

/// How many of an export's pages are written.
#[derive(Clone, Copy, Debug, Serialize, Type)]
pub struct ExportProgress {
    pub completed: u32,
    pub total: u32,
}

/// The export writing, which `stop_export` stops after the page it's on.
static EXPORTING: Mutex<Option<Arc<AtomicBool>>> = Mutex::new(None);

/// Picks a folder and writes `pages` of the open project there, in project
/// order; `on_progress` hears each page written. `labels` names a PSD's
/// layers the scene names nothing, in the interface's language. The project
/// is the one open when this starts. A stopped export keeps the files it
/// wrote and no unfinished archive.
#[tauri::command]
#[specta::specta]
pub async fn export(
    app: State<'_, Arc<App>>,
    window: Window<CefRuntime>,
    pages: Vec<PageId>,
    options: ExportOptions,
    labels: Labels,
    on_progress: Channel<ExportProgress>,
) -> Result<()> {
    let (name, session) = {
        let project = app.project.lock();
        let project = project.as_ref().context("no project is open")?;
        (project.name.clone(), Arc::clone(&project.session))
    };
    let template = Template::parse(&options.pattern)?;
    if options.formats.is_empty() || pages.is_empty() {
        return Err(anyhow::anyhow!("an export needs a format and a page").into());
    }
    let Some(folder) = rfd::AsyncFileDialog::new()
        .set_parent(&window)
        .pick_folder()
        .await
        .map(|picked| picked.path().to_owned())
    else {
        return Ok(());
    };
    let stop = Arc::new(AtomicBool::new(false));
    if let Some(going) = EXPORTING.lock().replace(Arc::clone(&stop)) {
        going.store(true, Ordering::Relaxed);
    }
    let app = Arc::clone(&app);
    let going = Arc::clone(&stop);
    let written = tokio::task::spawn_blocking(move || {
        let job = Job {
            app: &app,
            session: &session,
            template: &template,
            options: &options,
            labels: &labels,
            stop: &stop,
            on_progress: &on_progress,
        };
        job.write(&name, &pages, &folder)
    })
    .await;
    EXPORTING
        .lock()
        .take_if(|current| Arc::ptr_eq(current, &going));
    Ok(written??)
}

/// Stops the export writing after the page it's on.
#[tauri::command]
#[specta::specta]
pub fn stop_export() {
    if let Some(going) = EXPORTING.lock().take() {
        going.store(true, Ordering::Relaxed);
    }
}

struct Job<'a> {
    app: &'a App,
    session: &'a Mutex<Session>,
    template: &'a Template,
    options: &'a ExportOptions,
    labels: &'a Labels,
    stop: &'a AtomicBool,
    on_progress: &'a Channel<ExportProgress>,
}

impl Job<'_> {
    fn write(&self, name: &str, pages: &[PageId], folder: &Path) -> anyhow::Result<()> {
        let chosen: HashSet<PageId> = pages.iter().copied().collect();
        // Each chosen page with its 1-based place in the project and its name.
        let pages: Vec<(usize, PageId, String)> = {
            let session = self.session.lock();
            let project = session.project();
            for &page in &chosen {
                project.page(page)?;
            }
            project
                .pages()
                .enumerate()
                .filter(|(_, page)| chosen.contains(&page.id))
                .map(|(index, page)| (index + 1, page.id, page.name.clone()))
                .collect()
        };
        let formats = &self.options.formats;
        let png_files = formats.contains(&ExportFormat::Png);
        let psd_files = formats.contains(&ExportFormat::Psd);
        let directory = |extension: &str| -> anyhow::Result<PathBuf> {
            let directory = match self.options.subfolders && png_files && psd_files {
                true => folder.join(extension),
                false => folder.to_owned(),
            };
            fs_err::create_dir_all(&directory)?;
            Ok(directory)
        };
        let png_folder = png_files.then(|| directory("png")).transpose()?;
        let psd_folder = psd_files.then(|| directory("psd")).transpose()?;
        let project_file = |extension: &str| {
            folder.join(match sanitize_filename::sanitize(name.trim()) {
                stem if stem.is_empty() => format!("project.{extension}"),
                stem => format!("{stem}.{extension}"),
            })
        };
        let archive_path = formats
            .contains(&ExportFormat::Cbz)
            .then(|| project_file("cbz"));
        let mut archive = archive_path
            .as_ref()
            .map(|path| anyhow::Ok(::zip::ZipWriter::new(File::create(path)?)))
            .transpose()?;
        let pdf_path = formats
            .contains(&ExportFormat::Pdf)
            .then(|| project_file("pdf"));
        let mut pdf_document = pdf_path.is_some().then(Document::new);
        let total = u32::try_from(pages.len()).unwrap_or(u32::MAX);
        let _ = self.on_progress.send(ExportProgress {
            completed: 0,
            total,
        });
        // Names repeat across folders and the archive, but never within one.
        let (mut png_names, mut psd_names, mut entry_names) =
            (Names::default(), Names::default(), Names::default());
        for (completed, (index, page, label)) in (1u32..).zip(pages) {
            if self.stop.load(Ordering::Relaxed) {
                drop(archive);
                if let Some(path) = archive_path {
                    fs_err::remove_file(path)?;
                }
                return Ok(());
            }
            let stem = self.template.render(index, &label)?;
            if png_folder.is_some() || archive.is_some() || pdf_document.is_some() {
                let frame = self.app.renderer.render(&self.session.lock(), page)?;
                let image = frame.rasterize()?.image;
                if png_folder.is_some() || archive.is_some() {
                    let png = png(&image)?;
                    if let Some(directory) = &png_folder {
                        let file = format!("{}.png", png_names.unique(stem.clone()));
                        fs_err::write(directory.join(file), &png)?;
                    }
                    if let Some(archive) = &mut archive {
                        // Already-compressed PNGs gain nothing from deflate.
                        let options = ::zip::write::SimpleFileOptions::default()
                            .compression_method(::zip::CompressionMethod::Stored);
                        archive.start_file(
                            format!("{}.png", entry_names.unique(stem.clone())),
                            options,
                        )?;
                        archive.write_all(&png)?;
                    }
                }
                if let Some(pdf) = &mut pdf_document {
                    let (width, height) = image.dimensions();
                    // Import's own scale, so a page read back keeps its pixels.
                    let size = Size::from_wh(width as f32 / PDF_SCALE, height as f32 / PDF_SCALE)
                        .context("a page has no pixels")?;
                    pdf.start_page_with(PageSettings::new(size))
                        .surface()
                        .draw_image(Image::from_rgba8(image.into_raw(), width, height), size);
                }
            }
            if let Some(directory) = &psd_folder {
                let psd = koharu_psd::export(
                    &self.session.lock(),
                    &self.app.renderer,
                    page,
                    self.labels,
                )?;
                let file = format!("{}.psd", psd_names.unique(stem));
                fs_err::write(directory.join(file), psd)?;
            }
            let _ = self.on_progress.send(ExportProgress { completed, total });
        }
        if let Some(archive) = archive {
            archive.finish()?;
        }
        if let (Some(path), Some(pdf)) = (pdf_path, pdf_document) {
            fs_err::write(path, pdf.finish()?)?;
        }
        Ok(())
    }
}

fn png(image: &RgbaImage) -> anyhow::Result<Vec<u8>> {
    let mut png = Cursor::new(Vec::new());
    image.write_to(&mut png, ImageFormat::Png)?;
    Ok(png.into_inner())
}
