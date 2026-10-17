use std::{sync::Arc, time::Duration};

use crate::{Stage, resources::ResourceMonitor, stages::Stages};

pub(crate) struct AcceleratorGate {
    resources: Arc<ResourceMonitor>,
    lane: Option<Arc<tokio::sync::Semaphore>>,
}

pub(crate) struct AcceleratorPermit {
    _lane: Option<tokio::sync::OwnedSemaphorePermit>,
}

impl AcceleratorGate {
    pub(crate) fn new(device: &koharu_ml::Device, resources: Arc<ResourceMonitor>) -> Self {
        let accelerated = device.backend != koharu_ml::Backend::Cpu
            && device.device_type != koharu_ml::DeviceType::Cpu;
        Self {
            resources,
            // Heterogeneous CUDA model pairs took 2.5-4x longer together than
            // back-to-back on the target workload. Keep accelerator work
            // serialized while loaded weights remain resident between pages.
            lane: accelerated.then(|| Arc::new(tokio::sync::Semaphore::new(1))),
        }
    }

    pub(crate) async fn acquire(&self) -> AcceleratorPermit {
        let Some(lane) = self.lane.as_ref() else {
            return AcceleratorPermit::cpu();
        };
        AcceleratorPermit::accelerator(
            lane.clone()
                .acquire_owned()
                .await
                .expect("accelerator lane is never closed"),
        )
    }

    pub(crate) async fn recover(&self, stage: Stage, stages: &Stages) -> AcceleratorPermit {
        let permit = self.acquire().await;
        if self.lane.is_none() {
            return permit;
        }
        if unload_other_models(stage, stages) {
            let mut changed = self.resources.subscribe();
            let _ = tokio::time::timeout(Duration::from_millis(600), changed.changed()).await;
        }
        permit
    }
}

impl AcceleratorPermit {
    fn accelerator(lane: tokio::sync::OwnedSemaphorePermit) -> Self {
        Self { _lane: Some(lane) }
    }

    fn cpu() -> Self {
        Self { _lane: None }
    }
}

/// Stages whose models must be dropped so that only `requested` may hold one.
/// The order follows `Stage::ALL`, which keeps eviction order predictable in
/// logs. The requesting stage is never a target: keeping it resident lets the
/// next page of the same stage reuse its cell without reloading.
fn eviction_targets(requested: Stage) -> impl Iterator<Item = Stage> {
    Stage::ALL.into_iter().filter(move |stage| *stage != requested)
}

fn unload_other_models(requested: Stage, stages: &Stages) -> bool {
    let mut unloaded = false;
    for stage in eviction_targets(requested) {
        if stages.unload(stage) {
            unloaded = true;
            tracing::info!(target: "koharu_metrics", metric = "model_unload", stage = %stage);
            tracing::debug!(%stage, "unloaded a model to free the accelerator");
        }
    }
    unloaded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eviction_targets_exclude_only_the_requested_stage() {
        assert_eq!(
            eviction_targets(Stage::Inpainting).collect::<Vec<_>>(),
            [Stage::Detection, Stage::Ocr, Stage::Translation]
        );
        assert_eq!(
            eviction_targets(Stage::Detection).collect::<Vec<_>>(),
            [Stage::Ocr, Stage::Translation, Stage::Inpainting]
        );
    }

    #[test]
    fn every_stage_evicts_every_other_stage() {
        for requested in Stage::ALL {
            let targets = eviction_targets(requested).collect::<Vec<_>>();
            assert_eq!(targets.len(), Stage::ALL.len() - 1);
            assert!(!targets.contains(&requested));
        }
    }
}
