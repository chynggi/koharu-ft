//! Layer editing and undo/redo. Thin delegation to the desktop commands.

use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use koharu_scene::EntityId;
use serde::Deserialize;
use tauri::Manager as _;

use koharu_app::commands::canvas::CanvasChannel;
use koharu_app::commands::editing::{self, GeometryUpdate, TypographyUpdate};
use koharu_app::commands::project::{CurrentProject, Page};

use crate::AppState;
use crate::error::ApiResult;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/layers/source-text", post(set_source_text))
        .route("/layers/translation", post(set_translation))
        .route("/layers/typography", post(set_typography))
        .route("/layers/geometry", post(set_geometry))
        .route("/layers/visibility", post(set_visibility))
        .route("/layers/delete", post(delete_layers))
        .route("/layers/move", post(move_layer))
        .route("/project/undo", post(undo))
        .route("/project/redo", post(redo))
}

#[derive(Deserialize)]
struct SetSourceTextRequest {
    layer: EntityId,
    text: String,
}

async fn set_source_text(
    State(app): State<AppState>,
    Json(request): Json<SetSourceTextRequest>,
) -> ApiResult<()> {
    editing::set_source_text(
        request.layer,
        request.text,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}

#[derive(Deserialize)]
struct SetTranslationRequest {
    layer: EntityId,
    text: Option<String>,
}

async fn set_translation(
    State(app): State<AppState>,
    Json(request): Json<SetTranslationRequest>,
) -> ApiResult<()> {
    editing::set_translation(
        request.layer,
        request.text,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}

#[derive(Deserialize)]
struct SetTypographyRequest {
    updates: Vec<TypographyUpdate>,
}

async fn set_typography(
    State(app): State<AppState>,
    Json(request): Json<SetTypographyRequest>,
) -> ApiResult<()> {
    editing::set_typography(
        request.updates,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}

#[derive(Deserialize)]
struct SetGeometryRequest {
    updates: Vec<GeometryUpdate>,
}

async fn set_geometry(
    State(app): State<AppState>,
    Json(request): Json<SetGeometryRequest>,
) -> ApiResult<()> {
    editing::set_geometry(
        request.updates,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}

#[derive(Deserialize)]
struct SetVisibilityRequest {
    layers: Vec<EntityId>,
    visible: Option<bool>,
    opacity: Option<f32>,
}

async fn set_visibility(
    State(app): State<AppState>,
    Json(request): Json<SetVisibilityRequest>,
) -> ApiResult<()> {
    editing::set_visibility(
        request.layers,
        request.visible,
        request.opacity,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}

#[derive(Deserialize)]
struct DeleteLayersRequest {
    layers: Vec<EntityId>,
}

async fn delete_layers(
    State(app): State<AppState>,
    Json(request): Json<DeleteLayersRequest>,
) -> ApiResult<()> {
    editing::delete_layers(
        request.layers,
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}

#[derive(Deserialize)]
struct MoveLayerRequest {
    layer: EntityId,
    parent: EntityId,
    index: u32,
}

async fn move_layer(
    State(app): State<AppState>,
    Json(request): Json<MoveLayerRequest>,
) -> ApiResult<Json<Page>> {
    Ok(Json(
        editing::move_layer(
            request.layer,
            request.parent,
            request.index,
            app.state::<koharu_desktop::Desktop>(),
            app.state::<CurrentProject>(),
            app.state::<CanvasChannel>(),
        )
        .await?,
    ))
}

async fn undo(State(app): State<AppState>) -> ApiResult<()> {
    editing::undo(
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}

async fn redo(State(app): State<AppState>) -> ApiResult<()> {
    editing::redo(
        app.state::<koharu_desktop::Desktop>(),
        app.state::<CurrentProject>(),
        app.state::<CanvasChannel>(),
    )
    .await?;
    Ok(())
}
