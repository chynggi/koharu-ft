//! `POST /window/move-resize` — interactive move or resize of the desktop
//! window. Only Linux needs it: there the runtime's own window dragging cannot
//! see the pointer press, which lands on the CEF window (see
//! `koharu_app::linux_window`).

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;

use crate::AppState;
use crate::error::ApiResult;

pub fn router() -> Router<AppState> {
    Router::new().route("/window/move-resize", post(move_resize))
}

#[cfg(target_os = "linux")]
#[derive(serde::Deserialize)]
struct MoveResizeRequest {
    /// Resize from this edge or corner; absent to move the window.
    direction: Option<koharu_app::linux_window::ResizeDirection>,
}

#[cfg(target_os = "linux")]
async fn move_resize(
    State(app): State<AppState>,
    axum::Json(request): axum::Json<MoveResizeRequest>,
) -> ApiResult<StatusCode> {
    // Reading the native window handle blocks on the event loop thread.
    tokio::task::spawn_blocking(move || {
        koharu_app::linux_window::begin_move_resize(&app, request.direction)
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(not(target_os = "linux"))]
async fn move_resize(_state: State<AppState>) -> ApiResult<StatusCode> {
    Ok(StatusCode::NOT_IMPLEMENTED)
}
