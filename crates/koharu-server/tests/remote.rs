//! A desktop pipeline set to run remotely, against this server on this
//! machine.

use std::{collections::BTreeSet, sync::Arc};

use koharu_pipeline::{
    Event, Outcome, Pipeline, PipelineConfig, RemoteConfig, Request, Stage, Target,
};
use koharu_scene::{Edit, Session};
use parking_lot::Mutex;
use serde_json::json;

/// A server on a free port of this machine, and its address.
async fn serve(token: Option<&str>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a port");
    let address = listener.local_addr().expect("an address");
    let router = koharu_server::router(
        Pipeline::new(PipelineConfig::default()),
        token.map(str::to_owned),
    );
    tokio::spawn(async move { axum::serve(listener, router).await });
    format!("http://{address}")
}

/// A project of one blank page, with no scan, so detection loads no model.
fn project() -> (tempfile::TempDir, Arc<Mutex<Session>>) {
    let dir = tempfile::tempdir().expect("a folder");
    let mut session = Session::create(dir.path().join("test.khr")).expect("a project");
    session
        .apply(Edit {
            label: "test".to_owned(),
            ops: vec![
                serde_json::from_value(json!({
                    "op": "add_page", "index": 0, "size": {"width": 100, "height": 100},
                }))
                .expect("an op"),
            ],
        })
        .expect("a page");
    (dir, Arc::new(Mutex::new(session)))
}

/// Runs detection on the project, at `url` or else here, and what the run
/// told, its pages in order of the project's.
async fn detect(url: Option<String>) -> Vec<Event> {
    let (_dir, session) = project();
    let page = session.lock().project().pages().next().map(|page| page.id);
    let pipeline = Pipeline::new(PipelineConfig {
        remote: url.map(|url| RemoteConfig { url }),
        ..PipelineConfig::default()
    });
    let run = pipeline
        .run(
            session,
            Request {
                target: Target::Project,
                stages: BTreeSet::from([Stage::Detect]),
                config: None,
            },
        )
        .expect("the run starts");
    let told = pipeline
        .history(run, std::time::Duration::from_secs(120))
        .await
        .expect("its history");
    assert!(told.iter().all(|event| match event {
        Event::Started { page: at, .. }
        | Event::Finished { page: at, .. }
        | Event::Failed { page: at, .. } => Some(*at) == page,
        _ => true,
    }));
    told
}

#[tokio::test]
async fn a_remote_run_tells_what_a_local_one_does() {
    let url = serve(None).await;
    let remote = detect(Some(url)).await;
    assert!(
        matches!(
            remote.last(),
            Some(Event::Ended {
                outcome: Outcome::Done,
                ..
            })
        ),
        "{remote:?}"
    );
    // Pages differ by id from project to project; their kinds and order don't.
    let shape = |told: &[Event]| -> Vec<String> {
        told.iter()
            .map(|event| match event {
                Event::Queued { steps, .. } => format!("queued {steps}"),
                Event::Started { stage, .. } => format!("started {stage}"),
                Event::Finished { stage, .. } => format!("finished {stage}"),
                Event::Failed { stage, reason, .. } => format!("failed {stage}: {reason}"),
                Event::Ended { outcome, .. } => format!("ended {outcome:?}"),
            })
            .collect()
    };
    assert_eq!(shape(&remote), shape(&detect(None).await));
}

#[tokio::test]
async fn a_server_with_a_token_refuses_a_run_without_it() {
    let url = serve(Some("secret")).await;
    let told = detect(Some(url)).await;
    let Some(Event::Ended {
        outcome: Outcome::Failed { reason, .. },
        ..
    }) = told.last()
    else {
        panic!("the run fails: {told:?}");
    };
    assert!(reason.contains("401"), "{reason}");
}
