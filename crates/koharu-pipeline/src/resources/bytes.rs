//! Byte counts that carry where they came from.
//!
//! A memory figure in this crate is never a bare `u64`. It travels with the
//! source that produced it and, for readings, the scope that source covers, so
//! a measured number and a computed one cannot be mistaken for each other once
//! they have been passed around a few times.

use std::time::Instant;

/// A byte count together with its provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bytes {
    pub value: u64,
    pub provenance: Provenance,
}

/// Where a byte count came from. There is deliberately no `From<u64>` and no
/// raw constructor: naming a provenance is the only way to build a [`Bytes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// Read from a driver or OS API on this machine.
    Measured {
        source: MeasuredSource,
        scope: MemoryScope,
        sampled_at: Instant,
    },
    /// Computed from inputs we hold. `formula` names the calculation so the UI
    /// can say which one rather than showing a bare number.
    Estimated { formula: Estimate },
}

/// How much of the machine a reading covers.
///
/// Named `MemoryScope` because `Scope` is already the canvas bounds type here.
/// The distinction matters: DXGI reports this process only, while NVML and the
/// DRM counters report every process on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryScope {
    Process,
    Device,
    System,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasuredSource {
    Dxgi,
    Nvml,
    DrmSysfs,
    SystemMemory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimate {
    /// The resolved weight file's size on disk. Its error is the CPU/GPU layer
    /// split and metadata overhead, not the quantization.
    FileSize,
    /// llama.cpp's KV cache layout at `n_ctx` positions. Excludes the compute
    /// graph, which depends on batch shape and backend internals.
    KvCache { n_ctx: u32 },
}

/// The size of a resolved weight file on disk.
///
/// Not a guess from a quantization name: it is a real byte count of a real
/// file, so it bounds the weights from above. `None` when the file cannot be
/// stat'd, never a zero.
#[must_use]
pub fn file_size(path: &std::path::Path) -> Option<Bytes> {
    match std::fs::metadata(path) {
        Ok(metadata) => Some(Bytes::estimated(metadata.len(), Estimate::FileSize)),
        Err(error) => {
            tracing::debug!(path = %path.display(), %error, "could not size a model file");
            None
        }
    }
}

/// The coarse quality of a figure, for badging it in the UI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Measured,
    Estimated,
}

impl Bytes {
    #[must_use]
    pub fn measured(
        value: u64,
        source: MeasuredSource,
        scope: MemoryScope,
        sampled_at: Instant,
    ) -> Self {
        Self {
            value,
            provenance: Provenance::Measured {
                source,
                scope,
                sampled_at,
            },
        }
    }

    #[must_use]
    pub fn estimated(value: u64, formula: Estimate) -> Self {
        Self {
            value,
            provenance: Provenance::Estimated { formula },
        }
    }

    #[must_use]
    pub fn tier(&self) -> Tier {
        match self.provenance {
            Provenance::Measured { .. } => Tier::Measured,
            Provenance::Estimated { .. } => Tier::Estimated,
        }
    }

    /// The scope a reading covers. Estimates have none.
    #[must_use]
    pub fn scope(&self) -> Option<MemoryScope> {
        match self.provenance {
            Provenance::Measured { scope, .. } => Some(scope),
            Provenance::Estimated { .. } => None,
        }
    }

    /// Free headroom, which is only meaningful when both figures come from the
    /// same reading. A budget from one sample minus a usage figure from another
    /// describes no moment that ever existed, and on Windows the budget moves
    /// under external pressure, so the pair is rejected rather than subtracted.
    #[must_use]
    pub fn headroom_from(budget: Self, used: Self) -> Option<Self> {
        let (
            Provenance::Measured {
                source,
                scope,
                sampled_at,
            },
            Provenance::Measured {
                source: used_source,
                scope: used_scope,
                sampled_at: used_at,
            },
        ) = (budget.provenance, used.provenance)
        else {
            return None;
        };
        if source != used_source || scope != used_scope || sampled_at != used_at {
            return None;
        }
        Some(Self::measured(
            budget.value.saturating_sub(used.value),
            source,
            scope,
            sampled_at,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(value: u64, at: Instant) -> Bytes {
        Bytes::measured(value, MeasuredSource::Dxgi, MemoryScope::Process, at)
    }

    #[test]
    fn headroom_comes_from_a_single_sample() {
        let at = Instant::now();
        let headroom = Bytes::headroom_from(measured(12, at), measured(5, at)).unwrap();

        assert_eq!(headroom.value, 7);
        assert_eq!(headroom.scope(), Some(MemoryScope::Process));
    }

    #[test]
    fn headroom_across_two_samples_is_refused() {
        let first = Instant::now();
        let second = first + std::time::Duration::from_millis(100);

        assert_eq!(Bytes::headroom_from(measured(12, first), measured(5, second)), None);
    }

    #[test]
    fn headroom_across_two_scopes_is_refused() {
        let at = Instant::now();
        let device = Bytes::measured(12, MeasuredSource::Dxgi, MemoryScope::Device, at);

        assert_eq!(Bytes::headroom_from(device, measured(5, at)), None);
    }

    #[test]
    fn a_used_figure_above_the_budget_saturates_instead_of_wrapping() {
        let at = Instant::now();

        assert_eq!(Bytes::headroom_from(measured(5, at), measured(12, at)).unwrap().value, 0);
    }

    #[test]
    fn a_missing_file_has_no_size_rather_than_a_zero() {
        assert_eq!(file_size(std::path::Path::new("no-such-model.gguf")), None);
    }

    #[test]
    fn a_real_file_is_sized_from_disk() {
        // `file!()` is relative to the workspace root while a test runs from
        // the package root, so anchor on the manifest directory instead.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let sized = file_size(&path).unwrap();

        assert_eq!(sized.tier(), Tier::Estimated);
        assert_eq!(sized.value, std::fs::metadata(&path).unwrap().len());
        assert!(sized.value > 0);
    }

    #[test]
    fn an_estimate_carries_no_scope_and_cannot_produce_headroom() {
        let estimate = Bytes::estimated(4, Estimate::FileSize);

        assert_eq!(estimate.tier(), Tier::Estimated);
        assert_eq!(estimate.scope(), None);
        assert_eq!(Bytes::headroom_from(estimate, estimate), None);
    }
}
