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
            // serialized; a stage that takes the lane keeps only its own model
            // resident, because `acquire` evicts the rest.
            lane: accelerated.then(|| Arc::new(tokio::sync::Semaphore::new(1))),
        }
    }

    /// Takes the accelerator for `stage`, dropping every other stage's model
    /// before the caller loads its own.
    ///
    /// The eviction happens *after* the permit is held, so on the single lane
    /// no other stage can be mid-run and nothing running gets its memory
    /// pulled out from under it. On CPU there is no lane and two stages may
    /// genuinely overlap.
    ///
    /// The eviction itself is best-effort: `ModelCell::unload` uses `try_lock`,
    /// so a model whose cell is busy stays loaded. A miss shows up only as a
    /// missing `model_unload` metric. Nothing here refuses a load, and a load
    /// that then runs out of memory is caught by the stage runner's retry.
    pub(crate) async fn acquire(&self, stage: Stage, stages: &Stages) -> AcceleratorPermit {
        let permit = self.lane_acquire().await;
        unload_other_models(stage, stages);
        permit
    }

    /// Retries after memory pressure. The eviction already happened in
    /// [`Self::acquire`], so all that is left is to let the driver return the
    /// memory and let the monitor publish a fresh reading, which the next load
    /// sees as its headroom. The wait is unconditional whenever a lane exists:
    /// this path only runs after a failure, so one sample period is cheap.
    pub(crate) async fn recover(&self, stage: Stage, stages: &Stages) -> AcceleratorPermit {
        let permit = self.acquire(stage, stages).await;
        if self.lane.is_some() {
            let mut changed = self.resources.subscribe();
            let _ = tokio::time::timeout(Duration::from_millis(600), changed.changed()).await;
        }
        permit
    }

    async fn lane_acquire(&self) -> AcceleratorPermit {
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

/// Drops every other stage's model so that only `requested` may hold one. The
/// caller cannot tell whether anything was evicted, and does not need to: each
/// drop reports itself as a `model_unload` metric.
fn unload_other_models(requested: Stage, stages: &Stages) {
    for stage in eviction_targets(requested) {
        if stages.unload(stage) {
            tracing::info!(target: "koharu_metrics", metric = "model_unload", stage = %stage);
            tracing::debug!(%stage, "unloaded a model to free the accelerator");
        }
    }
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

    /// The lane serializes accelerator work, and that serialization is what
    /// makes the eviction in `acquire` safe to do without locking. Build the
    /// gate for an accelerated device so the lane exists, and check that the
    /// second caller really does wait for the first permit to drop.
    #[tokio::test]
    async fn a_second_acquire_waits_for_the_first_permit() {
        let device = koharu_ml::Device::cuda(0);
        let resources = crate::resources::ResourceMonitor::new(&device);
        let translator = koharu_translator::Translator::from_config(
            koharu_ml::Device::cpu(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let stages = Stages::new(
            &crate::PipelineConfig::default(),
            translator,
            &koharu_ml::Device::cpu(),
            resources.clone(),
        )
        .unwrap();
        let gate = Arc::new(AcceleratorGate::new(&device, resources));

        let held = gate.acquire(Stage::Inpainting, &stages).await;
        let second = tokio::spawn({
            let gate = gate.clone();
            async move { gate.acquire(Stage::Ocr, &stages).await }
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(50), second)
                .await
                .is_err(),
            "the second acquire must block while the first permit is held"
        );
        drop(held);
        assert!(gate.lane.is_some(), "this test needs the accelerator lane");
    }
}
