//! Canvas manifest/resource retrieval and paint/erase/transform/inpaint commits.

use axum::extract::{Path, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use koharu_desktop::{Frame, TransformFrame};
use koharu_scene::{EntityId, Revision};
use serde::Deserialize;
use tauri::Manager as _;

use koharu_app::commands::canvas::{
    self, CanvasChannel, CanvasGeneration, CanvasPagePreparation, LayerCommit, PaintBrush, Point,
};
use koharu_app::commands::project::CurrentProject;

use crate::AppState;
use crate::error::{ApiResult, ipc_bytes};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/canvas/manifest/{generation}", get(get_canvas_manifest))
        .route(
            "/canvas/resource/{generation}/{resource}",
            get(get_canvas_resource),
        )
        .route("/pages/{page}/canvas/prepare", post(prepare_canvas_page))
        .route(
            "/pages/{page}/canvas/manifest/{revision}",
            get(get_canvas_page_manifest),
        )
        .route(
            "/pages/{page}/canvas/resource/{revision}/{resource}",
            get(get_canvas_page_resource),
        )
        .route("/canvas/text/point", post(add_point_text))
        .route("/canvas/text/box", post(add_text_box))
        .route("/canvas/commit/paint", post(commit_paint))
        .route("/canvas/commit/erase", post(commit_erase))
        .route("/canvas/commit/transform", post(commit_transform))
        .route("/canvas/commit/inpaint", post(commit_inpaint))
}

async fn get_canvas_manifest(
    State(app): State<AppState>,
    Path(generation): Path<CanvasGeneration>,
) -> ApiResult<Vec<u8>> {
    ipc_bytes(
        canvas::get_canvas_manifest(generation, app.state::<koharu_desktop::Desktop>()).await?,
    )
}

async fn get_canvas_resource(
    State(app): State<AppState>,
    Path((generation, resource)): Path<(CanvasGeneration, String)>,
) -> ApiResult<Vec<u8>> {
    ipc_bytes(
        canvas::get_canvas_resource(generation, resource, app.state::<koharu_desktop::Desktop>())
            .await?,
    )
}

async fn prepare_canvas_page(
    State(app): State<AppState>,
    Path(page): Path<EntityId>,
) -> ApiResult<Json<Option<CanvasPagePreparation>>> {
    Ok(Json(
        canvas::prepare_canvas_page(
            page,
            app.state::<koharu_desktop::Desktop>(),
            app.state::<CurrentProject>(),
        )
        .await?,
    ))
}

async fn get_canvas_page_manifest(
    State(app): State<AppState>,
    Path((page, revision)): Path<(EntityId, Revision)>,
) -> ApiResult<Vec<u8>> {
    ipc_bytes(
        canvas::get_canvas_page_manifest(page, revision, app.state::<koharu_desktop::Desktop>())
            .await?,
    )
}

async fn get_canvas_page_resource(
    State(app): State<AppState>,
    Path((page, revision, resource)): Path<(EntityId, Revision, String)>,
) -> ApiResult<Vec<u8>> {
    ipc_bytes(
        canvas::get_canvas_page_resource(
            page,
            revision,
            resource,
            app.state::<koharu_desktop::Desktop>(),
        )
        .await?,
    )
}

#[derive(Deserialize)]
struct AddPointTextRequest {
    point: Point,
}

async fn add_point_text(
    State(app): State<AppState>,
    Json(request): Json<AddPointTextRequest>,
) -> ApiResult<Json<LayerCommit>> {
    Ok(Json(
        canvas::add_point_text(
            request.point,
            app.state::<koharu_desktop::Desktop>(),
            app.state::<CurrentProject>(),
            app.state::<CanvasChannel>(),
        )
        .await?,
    ))
}

#[derive(Deserialize)]
struct AddTextBoxRequest {
    frame: Frame,
}

async fn add_text_box(
    State(app): State<AppState>,
    Json(request): Json<AddTextBoxRequest>,
) -> ApiResult<Json<LayerCommit>> {
    Ok(Json(
        canvas::add_text_box(
            request.frame,
            app.state::<koharu_desktop::Desktop>(),
            app.state::<CurrentProject>(),
            app.state::<CanvasChannel>(),
        )
        .await?,
    ))
}

#[derive(Deserialize)]
struct CommitPaintRequest {
    expected_revision: Revision,
    layer: Option<EntityId>,
    points: Vec<Point>,
    brush: PaintBrush,
}

async fn commit_paint(
    State(app): State<AppState>,
    Json(request): Json<CommitPaintRequest>,
) -> ApiResult<Json<LayerCommit>> {
    Ok(Json(
        canvas::commit_paint(
            request.expected_revision,
            request.layer,
            request.points,
            request.brush,
            app.state::<koharu_desktop::Desktop>(),
            app.state::<CurrentProject>(),
            app.state::<CanvasChannel>(),
        )
        .await?,
    ))
}

#[derive(Deserialize)]
struct CommitEraseRequest {
    expected_revision: Revision,
    layer: EntityId,
    points: Vec<Point>,
    diameter: f32,
}

async fn commit_erase(
    State(app): State<AppState>,
    Json(request): Json<CommitEraseRequest>,
) -> ApiResult<Json<LayerCommit>> {
    Ok(Json(
        canvas::commit_erase(
            request.expected_revision,
            request.layer,
            request.points,
            request.diameter,
            app.state::<koharu_desktop::Desktop>(),
            app.state::<CurrentProject>(),
            app.state::<CanvasChannel>(),
        )
        .await?,
    ))
}

#[derive(Deserialize)]
struct CommitTransformRequest {
    expected_revision: Revision,
    elements: Vec<TransformFrame>,
}

async fn commit_transform(
    State(app): State<AppState>,
    Json(request): Json<CommitTransformRequest>,
) -> ApiResult<Json<Option<Revision>>> {
    Ok(Json(
        canvas::commit_transform(
            request.expected_revision,
            request.elements,
            app.state::<koharu_desktop::Desktop>(),
            app.state::<CurrentProject>(),
            app.state::<CanvasChannel>(),
        )
        .await?,
    ))
}

#[derive(Deserialize)]
struct CommitInpaintRequest {
    expected_revision: Revision,
    points: Vec<Point>,
    diameter: f32,
}

async fn commit_inpaint(
    State(app): State<AppState>,
    Json(request): Json<CommitInpaintRequest>,
) -> ApiResult<Json<Option<koharu_app::commands::processing::JobId>>> {
    Ok(Json(
        canvas::commit_inpaint(
            request.expected_revision,
            request.points,
            request.diameter,
            app.clone(),
            app.state::<CurrentProject>(),
        )
        .await?,
    ))
}
