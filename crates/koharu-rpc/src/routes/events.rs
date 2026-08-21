//! `GET /events` — a single SSE stream multiplexing the five startup-subscription
//! channels (canvas, job, download, resource, project) as `{"type":..,"data":..}`.

use axum::Router;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::get;
use futures::Stream;
use serde::Serialize;
use serde_json::json;
use tauri::Manager as _;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::UnboundedReceiverStream;

use koharu_app::commands::canvas::CanvasChannel;
use koharu_app::commands::lifecycle::{Download, DownloadState, ModelResources, ProjectChannel};
use koharu_app::commands::processing::JobChannel;

use crate::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/events", get(events))
}

async fn events(
    State(app): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();

    // All managed by `koharu_app`'s Tauri `setup`, which runs to completion
    // before the setup hook that starts this server, so they are always
    // present by the time a request can arrive.
    let mut canvas_rx = app.state::<CanvasChannel>().broadcast.subscribe();
    let mut job_rx = app.state::<JobChannel>().broadcast.subscribe();
    let mut project_rx = app.state::<ProjectChannel>().broadcast.subscribe();
    let mut download_rx = koharu_runtime::downloads::subscribe();

    // The pipeline is the exception, and it gets its own task. It is managed
    // by `koharu_app`'s *asynchronous* initialization, which loads the ML
    // runtime and so finishes long after the server starts accepting
    // connections — while the UI opens this stream the moment the page loads.
    // Reading it with `state()` alongside the channels above therefore panicked
    // ("state() called before manage()") on essentially every startup, killing
    // the connection. Waiting off to the side keeps the other four channels
    // flowing meanwhile, which matters because model downloads are reported
    // during exactly this window.
    let resource_tx = tx.clone();
    tokio::spawn(async move {
        let Some(mut resource_rx) = resource_receiver(&app, &resource_tx).await else {
            return;
        };
        while resource_rx.changed().await.is_ok() {
            let snapshot = resource_rx.borrow_and_update().clone();
            if !send(&resource_tx, "resource", &ModelResources::from(snapshot)) {
                break;
            }
        }
    });

    tokio::spawn(async move {
        loop {
            let sent = tokio::select! {
                result = canvas_rx.recv() => forward(&tx, "canvas", result),
                result = job_rx.recv() => forward(&tx, "job", result),
                result = project_rx.recv() => forward(&tx, "project", result),
                result = download_rx.recv() => match result {
                    Ok(event) => send(&tx, "download", &map_download(event)),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => true,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => false,
                },
            };
            if !sent {
                break;
            }
        }
    });

    Sse::new(UnboundedReceiverStream::new(rx).map(Ok)).keep_alive(KeepAlive::default())
}

/// Waits for the pipeline to be managed and subscribes to its resource
/// snapshots. Gives up, without subscribing, if the client hangs up first —
/// otherwise a page closed during startup would leave this task parked here
/// until initialization finished.
async fn resource_receiver(
    app: &AppState,
    tx: &tokio::sync::mpsc::UnboundedSender<Event>,
) -> Option<tokio::sync::watch::Receiver<koharu_pipeline::ResourceSnapshot>> {
    loop {
        if let Some(pipeline) = app.try_state::<koharu_pipeline::Pipeline>() {
            return Some(pipeline.subscribe_resources());
        }
        if tx.is_closed() {
            return None;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

/// Forwards a broadcast receive result, skipping `Lagged` and stopping on `Closed`.
fn forward<T: Serialize>(
    tx: &tokio::sync::mpsc::UnboundedSender<Event>,
    kind: &str,
    result: Result<T, tokio::sync::broadcast::error::RecvError>,
) -> bool {
    match result {
        Ok(value) => send(tx, kind, &value),
        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => true,
        Err(tokio::sync::broadcast::error::RecvError::Closed) => false,
    }
}

fn send<T: Serialize>(tx: &tokio::sync::mpsc::UnboundedSender<Event>, kind: &str, data: &T) -> bool {
    let Ok(event) = Event::default().json_data(json!({ "type": kind, "data": data })) else {
        return true;
    };
    tx.send(event).is_ok()
}

fn map_download(event: koharu_runtime::downloads::Event) -> Download {
    match event {
        koharu_runtime::downloads::Event::Started { id, name } => Download {
            id,
            state: DownloadState::Running,
            name: Some(name),
            completed: 0,
            total: 0,
            error: None,
        },
        koharu_runtime::downloads::Event::Progress {
            id,
            name,
            completed,
            total,
        } => Download {
            id,
            state: DownloadState::Running,
            name: Some(name),
            completed,
            total,
            error: None,
        },
        koharu_runtime::downloads::Event::Finished { id } => Download {
            id,
            state: DownloadState::Finished,
            name: None,
            completed: 0,
            total: 0,
            error: None,
        },
        koharu_runtime::downloads::Event::Failed { id, name, error } => Download {
            id,
            state: DownloadState::Failed,
            name: Some(name),
            completed: 0,
            total: 0,
            error: Some(error),
        },
    }
}
