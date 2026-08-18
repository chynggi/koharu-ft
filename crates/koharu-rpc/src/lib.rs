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

/// Axum state: the live Tauri application handle.
pub type AppState = AppHandle<Cef>;

/// Serve the API (and, if `static_dir` is given, the exported frontend) on
/// `127.0.0.1:port`. Binds the listener synchronously so the port is
/// guaranteed to be accepting connections by the time this call returns,
/// then drives the server on a spawned task until the process exits.
pub fn serve(app: AppHandle<Cef>, port: u16, static_dir: Option<PathBuf>) {
    let listener = match std::net::TcpListener::bind(("127.0.0.1", port)) {
        Ok(listener) => listener,
        Err(error) => {
            tracing::error!(%error, port, "failed to bind the Koharu API server");
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
    tracing::info!("Koharu API listening on http://127.0.0.1:{port}");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, api::router(app, static_dir)).await {
            tracing::error!(%error, "the Koharu API server stopped");
        }
    });
}

