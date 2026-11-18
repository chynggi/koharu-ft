//! `koharu-server --bind 0.0.0.0:47823`, with `KOHARU_SERVER_TOKEN` set.
//! Secrets, a hosted translator's key among them, come from the environment,
//! as `KOHARU_SECRET_<NAME>`.

use std::net::SocketAddr;

use anyhow::{Context as _, bail};
use clap::Parser;
use koharu_pipeline::{Pipeline, PipelineConfig};
use tracing_subscriber::EnvFilter;

/// The variable holding the token every request must bring.
const TOKEN: &str = "KOHARU_SERVER_TOKEN";

#[derive(Debug, Parser)]
#[command(about = "Run Koharu's models for the desktop app on another machine")]
struct Cli {
    /// The address to listen on. Anything but this machine needs a token.
    #[arg(long, default_value = "127.0.0.1:47823")]
    bind: SocketAddr,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    let cli = Cli::parse();
    koharu_config::secrets::from_environment();
    let token = std::env::var(TOKEN).ok().filter(|token| !token.is_empty());
    if token.is_none() && !cli.bind.ip().is_loopback() {
        bail!("listening on {} needs {TOKEN} set", cli.bind);
    }
    let app = koharu_server::router(Pipeline::new(PipelineConfig::default()), token);
    let listener = tokio::net::TcpListener::bind(cli.bind)
        .await
        .with_context(|| format!("couldn't listen on {}", cli.bind))?;
    tracing::info!("listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
