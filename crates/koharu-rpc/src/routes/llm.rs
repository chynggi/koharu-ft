//! Local LLM diagnostics and GGUF registration over HTTP.

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::extract::{ConnectInfo, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use koharu_app::commands::llm::{self, LlmCapabilities};
use tauri::Manager;

use crate::AppState;
use crate::error::ApiResult;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/llm/capabilities", get(get_llm_capabilities))
        .route("/llm/gguf-file", post(pick_gguf_file))
}

async fn get_llm_capabilities(_state: State<AppState>) -> ApiResult<Json<LlmCapabilities>> {
    Ok(Json(llm::get_llm_capabilities().await?))
}

/// Opens the native `.gguf` picker, but **only for a loopback caller**.
///
/// The picker runs in the server process, so the file it returns is a path on
/// the machine running Koharu — which is what llama.cpp needs, since that is
/// the process that loads the weights. For the desktop CEF window the server
/// and the user are the same machine and the dialog is exactly right.
///
/// A remote caller gets `None` instead of a dialog, and the UI falls back to
/// typing the server-side path. Opening it anyway would be worse than useless:
/// in the container the window runs on a headless Xvfb display, so the dialog
/// would appear where nobody can see it and the request would never return.
async fn pick_gguf_file(
    State(app): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
) -> ApiResult<Json<Option<PathBuf>>> {
    if !peer.ip().is_loopback() {
        return Ok(Json(None));
    }
    let Some(window) = app.get_webview_window("main") else {
        tracing::warn!("no main window to parent the GGUF picker to");
        return Ok(Json(None));
    };
    Ok(Json(llm::pick_gguf_file(window).await?))
}
