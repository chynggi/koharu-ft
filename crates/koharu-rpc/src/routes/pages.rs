//! Page listing, selection, ordering, import, and export.
//!
//! Import and export each come in two shapes, and which one a caller wants is
//! decided by **where the files have to live**, not by preference:
//!
//! - `/pages/import` and `/pages/export` take server-side paths. That is right
//!   for the desktop window, where the server and the user are the same machine
//!   and the bytes are already on disk.
//! - `/pages/import/upload` and `/pages/export/download` move the bytes over
//!   HTTP. That is the only thing that can work for a remote browser: the source
//!   images live on the user's machine and the rendered output has to get back
//!   there, so no path either side could name would refer to the same file.

use std::io::Write as _;
use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Context as _;
use axum::Json;
use axum::body::Bytes;
use axum::extract::{ConnectInfo, DefaultBodyLimit, Multipart, Path, State};
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use tokio::io::AsyncWriteExt as _;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, EntityId, PageDraft};
use serde::Deserialize;
use tauri::Manager as _;

use koharu_app::commands::canvas::CanvasChannel;
use koharu_app::commands::editing;
use koharu_app::commands::import;
use koharu_app::commands::lifecycle::{self, PageImportSource, PageSelection};
use koharu_app::commands::output::{self, ExportFormat};
use koharu_app::commands::processing::Processing;
use koharu_app::commands::project::{CurrentProject, Page, PageSummary};

use crate::AppState;
use crate::error::{ApiResult, ipc_bytes};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/pages", get(get_pages))
        .route("/page", get(get_page))
        .route("/page/select", post(select_page))
        .route("/pages/import", post(import_pages))
        // The body limit is lifted rather than raised: a manga import can be a
        // multi-gigabyte CBZ, and there is no honest number between "a few
        // pages" and "a whole series". Memory stays bounded regardless because
        // every field is streamed to disk a chunk at a time; the resource this
        // spends is temp-directory space, and reaching the route at all already
        // requires the API token on any non-loopback bind.
        .route(
            "/pages/import/upload",
            post(import_upload).layer(DefaultBodyLimit::disable()),
        )
        .route("/pages/import/dialog", post(import_dialog))
        .route("/pages/export", post(export_pages))
        .route("/pages/export/dialog", post(export_dialog))
        .route("/pages/export/download", post(export_download))
        .route("/pages/{id}/thumbnail", get(get_thumbnail))
        .route("/page/rename", post(rename_page))
        .route("/pages/delete", post(delete_pages))
        .route("/page/move", post(move_page))
}

async fn get_pages(State(app): State<AppState>) -> ApiResult<Json<Vec<PageSummary>>> {
    Ok(Json(
        lifecycle::get_pages(app.state::<CurrentProject>()).await?,
    ))
}

async fn get_page(State(app): State<AppState>) -> ApiResult<Json<Option<Page>>> {
    Ok(Json(
        lifecycle::get_page(app.state::<CurrentProject>()).await?,
    ))
}

#[derive(Deserialize)]
struct SelectPageRequest {
    page: EntityId,
}

async fn select_page(
    State(app): State<AppState>,
    Json(request): Json<SelectPageRequest>,
) -> ApiResult<Json<PageSelection>> {
    Ok(Json(
        lifecycle::select_page(
            app.state::<koharu_desktop::Desktop>(),
            request.page,
            app.state::<CurrentProject>(),
            app.state::<CanvasChannel>(),
        )
        .await?,
    ))
}

#[derive(Deserialize)]
struct ImportPagesRequest {
    files: Vec<String>,
}

async fn import_pages(
    State(app): State<AppState>,
    Json(request): Json<ImportPagesRequest>,
) -> ApiResult<Json<Vec<PageSummary>>> {
    reject_while_processing(&app)?;
    let files: Vec<PathBuf> = request
        .files
        .into_iter()
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .collect();
    if files.is_empty() {
        return Err(anyhow::anyhow!("none of the provided files exist").into());
    }
    commit_imported_pages(&app, decode_paths(files).await?).await
}

/// Import uploaded bytes by first writing them to a temporary directory and
/// then running the ordinary path-based import over them.
///
/// The detour through disk is deliberate. `import::import` dispatches on the
/// file extension and hands ZIP/CBZ, RAR and PDF sources to extractors that
/// each take a `&Path`. A bytes-based dispatch would therefore be a *second*
/// copy of that four-way branch, free to drift from the first. Writing the
/// upload out costs one pass over bytes that just crossed a network, and buys
/// a single import implementation for every format.
async fn import_upload(
    State(app): State<AppState>,
    mut multipart: Multipart,
) -> ApiResult<Json<Vec<PageSummary>>> {
    reject_while_processing(&app)?;
    let staging = tempfile::tempdir().context("failed to create a staging directory")?;
    let mut files = Vec::new();
    while let Some(mut field) = multipart
        .next_field()
        .await
        .context("failed to read the next uploaded file")?
    {
        // Take only the final component: the name is caller-controlled and is
        // about to become a path, so anything that could climb out of the
        // staging directory has to be dropped here rather than sanitized.
        let Some(name) = field
            .file_name()
            .map(std::path::Path::new)
            .and_then(std::path::Path::file_name)
            .map(|name| name.to_string_lossy().into_owned())
        else {
            continue;
        };
        let path = staging.path().join(&name);
        let mut file = tokio::fs::File::create(&path)
            .await
            .with_context(|| format!("failed to stage the uploaded file {name}"))?;
        // Streamed rather than `field.bytes()`: a single CBZ can be larger than
        // it is reasonable to hold in memory, and the whole point of staging is
        // that it never has to be.
        while let Some(chunk) = field
            .chunk()
            .await
            .with_context(|| format!("failed to read the uploaded file {name}"))?
        {
            file.write_all(&chunk)
                .await
                .with_context(|| format!("failed to stage the uploaded file {name}"))?;
        }
        file.flush()
            .await
            .with_context(|| format!("failed to stage the uploaded file {name}"))?;
        files.push(path);
    }
    if files.is_empty() {
        return Err(anyhow::anyhow!("the upload contained no files").into());
    }
    let pages = decode_paths(files).await?;
    // Held until here so the staging directory outlives the decode; dropping it
    // removes the files.
    drop(staging);
    commit_imported_pages(&app, pages).await
}

#[derive(Deserialize)]
struct ImportDialogRequest {
    source: PageImportSource,
}

/// Import through the desktop window's native file dialog.
///
/// Loopback only, and the guard is not a formality: the dialog opens in the
/// server process, so for a remote caller it would appear on a machine they
/// cannot see — and in the container that display is a headless Xvfb, which
/// means the request would simply never return. Browsers use
/// `/pages/import/upload` instead.
async fn import_dialog(
    State(app): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Json(request): Json<ImportDialogRequest>,
) -> ApiResult<Json<Vec<PageSummary>>> {
    let window = require_local_window(&app, peer)?;
    lifecycle::import_pages(
        request.source,
        window,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<Processing>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(Json(
        lifecycle::get_pages(app.state::<CurrentProject>()).await?,
    ))
}

#[derive(Deserialize)]
struct ExportDialogRequest {
    pages: Vec<EntityId>,
    format: ExportFormat,
}

/// Export through the desktop window's native folder picker. Loopback only,
/// for the same reason as `import_dialog`.
async fn export_dialog(
    State(app): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Json(request): Json<ExportDialogRequest>,
) -> ApiResult<()> {
    let window = require_local_window(&app, peer)?;
    output::export_pages(
        window,
        request.pages,
        request.format,
        app.state::<CurrentProject>(),
        app.state::<koharu_desktop::Desktop>(),
    )
    .await?;
    Ok(())
}

fn require_local_window(
    app: &AppState,
    peer: SocketAddr,
) -> ApiResult<tauri::WebviewWindow<tauri::Cef>> {
    if !peer.ip().is_loopback() {
        return Err(anyhow::anyhow!(
            "a native dialog can only be opened for a caller on this machine"
        )
        .into());
    }
    Ok(app
        .get_webview_window("main")
        .context("there is no desktop window to open a dialog in")?)
}

fn reject_while_processing(app: &AppState) -> ApiResult<()> {
    if !app.state::<Processing>().stops.lock().is_empty() {
        return Err(anyhow::anyhow!("pages cannot be imported while processing is running").into());
    }
    Ok(())
}

async fn decode_paths(files: Vec<PathBuf>) -> ApiResult<Vec<import::Page>> {
    Ok(tokio::task::spawn_blocking(move || import::import(files))
        .await
        .context("page import worker stopped unexpectedly")??)
}

async fn commit_imported_pages(
    app: &AppState,
    pages: Vec<import::Page>,
) -> ApiResult<Json<Vec<PageSummary>>> {
    let app = app.clone();
    let desktop = app.state::<koharu_desktop::Desktop>();
    let canvas_channel = app.state::<CanvasChannel>();
    let (commit, page) = {
        let current = app.state::<CurrentProject>();
        let mut project = current.project.lock().await;
        let project = project
            .as_mut()
            .context("no project is open")?;
        let source = AssetRole::new("source")?;
        let patch = project.snapshot().patch(|edit| {
            for imported in pages {
                let page = edit.add_page(
                    PageDraft::new(
                        imported.name,
                        f64::from(imported.width),
                        f64::from(imported.height),
                    ),
                    At::End,
                )?;
                edit.set_asset(
                    page,
                    &source,
                    AssetInput::new(
                        imported.bytes,
                        imported.format.to_mime_type(),
                        AssetMetadata {
                            width: Some(imported.width),
                            height: Some(imported.height),
                            attributes: Default::default(),
                        },
                    ),
                )?;
            }
            Ok(())
        })?;
        let commit = project.session.commit(patch).await?;
        project.record(vec![commit.revision]);
        project.reconcile_page();
        (commit, project.active_page())
    };
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    canvas_channel.publish(desktop.canvas_state());
    Ok(Json(lifecycle::get_pages(app.state::<CurrentProject>()).await?))
}

#[derive(Deserialize)]
struct RenamePageRequest {
    page: EntityId,
    label: String,
}

async fn rename_page(
    State(app): State<AppState>,
    Json(request): Json<RenamePageRequest>,
) -> ApiResult<()> {
    editing::rename_page(
        request.page,
        request.label,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}

#[derive(Deserialize)]
struct DeletePagesRequest {
    pages: Vec<EntityId>,
}

async fn delete_pages(
    State(app): State<AppState>,
    Json(request): Json<DeletePagesRequest>,
) -> ApiResult<()> {
    editing::delete_pages(
        request.pages,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}

#[derive(Deserialize)]
struct ExportPagesRequest {
    pages: Vec<EntityId>,
    format: ExportFormat,
    directory: String,
}

async fn export_pages(
    State(app): State<AppState>,
    Json(request): Json<ExportPagesRequest>,
) -> ApiResult<()> {
    let directory = PathBuf::from(request.directory);
    if !directory.is_dir() {
        return Err(anyhow::anyhow!("the export directory does not exist").into());
    }
    output::export_pages_to(
        directory,
        request.pages,
        request.format,
        app.state::<CurrentProject>(),
        app.state::<koharu_desktop::Desktop>(),
    )
    .await?;
    Ok(())
}

#[derive(Deserialize)]
struct ExportDownloadRequest {
    pages: Vec<EntityId>,
    format: ExportFormat,
}

/// Render an export into a temporary directory and return it as one ZIP.
///
/// The detour mirrors the one in `import_upload`, for the same reason:
/// `export_pages_to` already renders, names and writes every page, and asking
/// it to yield bytes instead would mean a second copy of that pipeline. One
/// archive rather than one response per page because an export is a set - the
/// browser cannot be made to save twenty files from one gesture, and the
/// `{:04}_` prefixes that carry reading order only mean something together.
async fn export_download(
    State(app): State<AppState>,
    Json(request): Json<ExportDownloadRequest>,
) -> ApiResult<impl IntoResponse> {
    let staging = tempfile::tempdir().context("failed to create a staging directory")?;
    output::export_pages_to(
        staging.path().to_owned(),
        request.pages,
        request.format,
        app.state::<CurrentProject>(),
        app.state::<koharu_desktop::Desktop>(),
    )
    .await?;
    let root = staging.path().to_owned();
    let archive = tokio::task::spawn_blocking(move || archive_directory(&root))
        .await
        .context("export archive worker stopped unexpectedly")??;
    drop(staging);
    Ok((
        [
            (header::CONTENT_TYPE, "application/zip"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"koharu-export.zip\"",
            ),
        ],
        Bytes::from(archive),
    ))
}

/// Pack the flat set of files `export_pages_to` just wrote into a ZIP.
fn archive_directory(root: &std::path::Path) -> anyhow::Result<Vec<u8>> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    // Deflate earns little on PNG and a lot on PSD. One method for both keeps
    // this to a single path; the cost next to rendering the pages is noise.
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    let mut entries: Vec<_> = std::fs::read_dir(root)
        .context("failed to list the rendered export")?
        .collect::<Result<Vec<_>, _>>()
        .context("failed to list the rendered export")?
        .into_iter()
        .filter(|entry| entry.path().is_file())
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    if entries.is_empty() {
        anyhow::bail!("the export produced no files");
    }
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        let bytes = std::fs::read(entry.path())
            .with_context(|| format!("failed to read the rendered page {name}"))?;
        writer
            .start_file(&name, options)
            .with_context(|| format!("failed to add {name} to the export archive"))?;
        writer
            .write_all(&bytes)
            .with_context(|| format!("failed to write {name} into the export archive"))?;
    }
    Ok(writer
        .finish()
        .context("failed to finalize the export archive")?
        .into_inner())
}

async fn get_thumbnail(
    State(app): State<AppState>,
    Path(page): Path<EntityId>,
) -> ApiResult<Vec<u8>> {
    ipc_bytes(output::get_thumbnail(page, app.state::<CurrentProject>()).await?)
}

#[derive(Deserialize)]
struct MovePageRequest {
    page: EntityId,
    index: u32,
}

async fn move_page(
    State(app): State<AppState>,
    Json(request): Json<MovePageRequest>,
) -> ApiResult<()> {
    editing::move_page(
        request.page,
        request.index,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}
