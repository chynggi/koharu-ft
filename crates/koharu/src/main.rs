#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use clap::Parser as _;
use koharu::panic;
use koharu::sentry;
use koharu_app as app;
use tracing_subscriber::layer::SubscriberExt as _;

#[derive(clap::Parser)]
#[command(version, about)]
struct Cli {}

#[tokio::main]
#[tauri::cef_entry_point]
async fn main() {
    #[cfg(target_os = "windows")]
    {
        // SAFETY: This only requests the existing parent console. It does not allocate one.
        let _ = unsafe {
            windows::Win32::System::Console::AttachConsole(
                windows::Win32::System::Console::ATTACH_PARENT_PROCESS,
            )
        };
    }

    let _cli = Cli::parse();
    let _guard = sentry::initialize();
    panic::install();

    // HTTP API + static frontend, served from the desktop process; the main
    // window loads its UI from here. Port defaults to `DEFAULT_RPC_PORT` but
    // can be overridden with `KOHARU_RPC_PORT`. The static frontend directory
    // defaults to the workspace's exported `packages/koharu/out` (dev
    // workflow) but can be overridden with `KOHARU_STATIC_DIR`.
    // Must match `devUrl` in tauri.conf.json: the CEF window's initial navigation
    // races the async browser-creation callback (window.navigate() silently no-ops
    // if called before it), so the config-baked URL is what actually loads when
    // KOHARU_RPC_PORT isn't overridden.
    const DEFAULT_RPC_PORT: u16 = 47823;
    let port = std::env::var("KOHARU_RPC_PORT")
        .ok()
        .and_then(|port| port.parse().ok())
        .unwrap_or(DEFAULT_RPC_PORT);
    let static_dir = std::env::var("KOHARU_STATIC_DIR").ok().map(std::path::PathBuf::from).unwrap_or_else(|| {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/koharu/out")
    });
    // Defaults to loopback-only, matching the desktop window's own connection.
    // Set KOHARU_RPC_HOST=0.0.0.0 (e.g. inside a container) to accept remote
    // connections; KOHARU_API_TOKEN is then required by the server for every
    // /api/v1/* request (see koharu_rpc::api::require_token) to avoid exposing
    // provider secrets and project data with no authentication.
    let host = std::env::var("KOHARU_RPC_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let api_token = std::env::var("KOHARU_API_TOKEN").ok();
    if host != "127.0.0.1" && api_token.is_none() {
        panic!(
            "KOHARU_RPC_HOST is set to a non-loopback address ({host}) but KOHARU_API_TOKEN is \
             unset; refusing to expose the API without authentication"
        );
    }
    koharu_app::extend_setup(move |handle| {
        koharu_rpc::serve(handle, &host, port, Some(static_dir), api_token)
    });
    let frontend_url: tauri::Url = format!("http://127.0.0.1:{port}/")
        .parse()
        .expect("the generated frontend URL is always valid");

    let filter = tracing_subscriber::filter::EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry()
            .with(filter)
            .with(sentry::tracing_layer())
            .with(koharu_metrics::layer())
            .with(koharu::tracing::TimingLayer::new()),
    )
    .expect("failed to set the global tracing subscriber");
    tokio::task::block_in_place(|| app::run(tauri::generate_context!(), frontend_url))
        .expect("failed to run the desktop application");
}
