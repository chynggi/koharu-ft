use anyhow::Context as _;
use koharu_desktop::{CanvasState, Desktop};
use koharu_scene::EntityId;
use serde::Deserialize;
use specta::Type;
use tauri::State;

use super::{
    Error,
    canvas::{CanvasChannel, Point},
    project::{CurrentProject, Page, Project, Typography},
};
async fn synchronize_canvas(
    desktop: &Desktop,
    commit: &koharu_scene::Commit,
    page: Option<EntityId>,
) -> anyhow::Result<CanvasState> {
    desktop.synchronize(&commit.snapshot, page, commit).await?;
    Ok(desktop.canvas_state())
}

#[derive(Clone, Debug, Deserialize, Type)]
pub struct GeometryUpdate {
    pub layer: EntityId,
    pub points: Option<Vec<Point>>,
}

#[derive(Clone, Debug, Deserialize, Type)]
pub struct TypographyUpdate {
    pub layer: EntityId,
    pub typography: Typography,
}

#[tauri::command]
#[specta::specta]
pub async fn rename_page(
    page: EntityId,
    label: String,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.rename_page(page, label).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn set_page_chapter_start(
    page: EntityId,
    chapter_start: bool,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, active_page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_page_chapter_start(page, chapter_start).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, active_page).await?;
    canvas_channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn delete_pages(
    pages: Vec<EntityId>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.delete_pages(pages).await?;
        project.record_commit(&commit);
        project.reconcile_page();
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn move_page(
    page: EntityId,
    index: u32,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.move_page(page, index as usize).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn merge_pages(
    pages: Vec<EntityId>,
    delete_originals: bool,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<EntityId, Error> {
    let (commit, merged) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let (commit, merged) = project.merge_pages(pages, delete_originals).await?;
        project.record_commit(&commit);
        project.select_page(merged)?;
        (commit, merged)
    };
    let canvas = synchronize_canvas(&desktop, &commit, Some(merged)).await?;
    canvas_channel.publish(canvas);
    Ok(merged)
}

#[tauri::command]
#[specta::specta]
pub async fn set_source_text(
    layer: EntityId,
    text: String,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_source_text(layer, text).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn set_translation(
    layer: EntityId,
    text: Option<String>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_translation(layer, text).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn set_typography(
    updates: Vec<TypographyUpdate>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_typography(updates).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn set_geometry(
    updates: Vec<GeometryUpdate>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_geometry(updates).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn set_visibility(
    layers: Vec<EntityId>,
    visible: Option<bool>,
    opacity: Option<f32>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.set_visibility(layers, visible, opacity).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn delete_layers(
    layers: Vec<EntityId>,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.delete_layers(layers).await?;
        project.record_commit(&commit);
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn move_layer(
    layer: EntityId,
    parent: EntityId,
    index: u32,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<Page, Error> {
    let (commit, page, view) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.move_layer(layer, parent, index as usize).await?;
        project.record_commit(&commit);
        let page = project.active_page().context("no active page")?;
        let view = Project::page(&commit.snapshot, page)?;
        (commit, page, view)
    };
    desktop
        .synchronize(&commit.snapshot, Some(page), &commit)
        .await?;
    canvas_channel.publish(desktop.canvas_state());
    Ok(view)
}

#[tauri::command]
#[specta::specta]
pub async fn paste_text_layers(
    layers: Vec<EntityId>,
    page: EntityId,
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<Vec<EntityId>, Error> {
    let (commit, added) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let (commit, added) = project.paste_text_layers(page, layers).await?;
        project.record_commit(&commit);
        (commit, added)
    };
    let canvas = synchronize_canvas(&desktop, &commit, Some(page)).await?;
    canvas_channel.publish(canvas);
    Ok(added)
}

#[tauri::command]
#[specta::specta]
pub async fn undo(
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.undo().await?;
        project.reconcile_page();
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn redo(
    desktop: State<'_, Desktop>,
    project: State<'_, CurrentProject>,
    canvas_channel: State<'_, CanvasChannel>,
) -> Result<(), Error> {
    let (commit, page) = {
        let mut project = project.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let commit = project.redo().await?;
        project.reconcile_page();
        (commit, project.active_page())
    };
    let canvas = synchronize_canvas(&desktop, &commit, page).await?;
    canvas_channel.publish(canvas);
    Ok(())
}
