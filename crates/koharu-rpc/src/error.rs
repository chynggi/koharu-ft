//! Uniform JSON error responses.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use tauri::ipc::{InvokeResponseBody, IpcResponse};

pub struct ApiError(anyhow::Error);

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self(error)
    }
}

impl From<koharu_scene::Error> for ApiError {
    fn from(error: koharu_scene::Error) -> Self {
        Self(error.into())
    }
}

impl From<koharu_app::commands::Error> for ApiError {
    fn from(error: koharu_app::commands::Error) -> Self {
        Self(anyhow::Error::msg(error.to_string()))
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let message = format!("{:#}", self.0);
        tracing::error!(error = %message, "API request failed");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": message })),
        )
            .into_response()
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

/// Unwraps a Tauri binary IPC response (`CanvasBytes`, `ThumbnailBytes`,
/// `FontPreviewBytes`, ...) into raw bytes for an octet-stream HTTP response.
pub fn ipc_bytes<T: IpcResponse>(value: T) -> ApiResult<Vec<u8>> {
    let body = value.body().map_err(anyhow::Error::from)?;
    Ok(match body {
        InvokeResponseBody::Raw(bytes) => bytes,
        InvokeResponseBody::Json(json) => json.into_bytes(),
    })
}
