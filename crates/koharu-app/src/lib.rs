//! Koharu's Tauri-managed application state, commands, and lifecycle.

use tauri::{AppHandle, Cef};

mod app;
pub mod commands;

pub use app::run;

/// Register a hook that runs once during Tauri setup, after the application
/// states are managed and before the main window is shown. Embedders use this
/// to attach side services (for example the HTTP API in `koharu-rpc`) that
/// need an `AppHandle`.
static SETUP_HOOKS: parking_lot::Mutex<Vec<Box<dyn FnOnce(AppHandle<Cef>) + Send>>> =
    parking_lot::Mutex::new(Vec::new());

pub fn extend_setup(hook: impl FnOnce(AppHandle<Cef>) + Send + 'static) {
    SETUP_HOOKS.lock().push(Box::new(hook));
}

pub(crate) fn take_setup_hooks() -> Vec<Box<dyn FnOnce(AppHandle<Cef>) + Send>> {
    std::mem::take(&mut *SETUP_HOOKS.lock())
}
