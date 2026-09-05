//! HTTP API that exposes the running Koharu desktop application.
//!
//! The server embeds into the Tauri process through
//! `koharu_app::extend_setup` and drives the same managed states and
//! command functions as the desktop UI. There is no separate standalone
//! process: without the desktop runtime (projects, renderer, pipeline)
//! there is nothing meaningful to serve.

pub mod api;
pub mod error;
pub mod routes;

use std::path::PathBuf;

use tauri::{AppHandle, Cef};
use tauri::Manager as _;

/// Axum state: the live Tauri application handle.
pub type AppState = AppHandle<Cef>;

/// Serve the API (and, if `static_dir` is given, the exported frontend) on
/// `{host}:{port}`. Binds the listener synchronously so the port is
/// guaranteed to be accepting connections by the time this call returns,
/// then drives the server on a spawned task until the process exits.
///
/// `api_token`, when set, requires every `/api/v1/*` request to present it
/// as `Authorization: Bearer <token>` or a `?token=<token>` query parameter
/// (the latter exists because browsers' `EventSource` cannot set headers).
/// Leave it `None` only for loopback-bound, single-user local use — binding
/// to a non-loopback `host` without a token exposes the API, including
/// provider secrets handled by the config routes, to anyone who can reach it.
pub fn serve(
    app: AppHandle<Cef>,
    host: &str,
    port: u16,
    static_dir: Option<PathBuf>,
    api_token: Option<String>,
) {
    let listener = match std::net::TcpListener::bind((host, port)) {
        Ok(listener) => listener,
        Err(error) => {
            tracing::error!(%error, host, port, "failed to bind the Koharu API server");
            return;
        }
    };
    if let Err(error) = listener.set_nonblocking(true) {
        tracing::error!(%error, "failed to configure the Koharu API listener");
        return;
    }
    let listener = match tokio::net::TcpListener::from_std(listener) {
        Ok(listener) => listener,
        Err(error) => {
            tracing::error!(%error, "failed to adopt the Koharu API listener into tokio");
            return;
        }
    };
    tracing::info!("Koharu API listening on http://{host}:{port}");
    app.manage(routes::pages::ExportStaging::default());
    tokio::spawn(async move {
        // `with_connect_info` so routes can tell a loopback caller (the desktop
        // window) from a remote browser — see `routes::llm::pick_gguf_file`.
        let service = api::router(app, static_dir, api_token)
            .into_make_service_with_connect_info::<std::net::SocketAddr>();
        if let Err(error) = axum::serve(listener, service).await {
            tracing::error!(%error, "the Koharu API server stopped");
        }
    });
}

