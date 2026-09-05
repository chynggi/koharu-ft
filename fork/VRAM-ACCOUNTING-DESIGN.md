# VRAM accounting — the measured / attributed / estimated split

Design only. No code in this increment. This is the prerequisite the Phase 2
report flagged: brief §39 forbids presenting unverified VRAM numbers as exact
and forbids inferring memory use from a quantization name, so the Budget Manager
(§11) cannot be built until every number it handles carries its own provenance.

Everything below is grounded in the current tree; file:line references are to
0.70.3 plus the Phase 1–2 changes.

---

## 1. What already exists (verified)

Koharu already has live accelerator telemetry. This was not obvious from the
brief and changes the shape of the work considerably.

| piece | where | what it does |
| --- | --- | --- |
| `ResourceMonitor` | `koharu-pipeline/src/resources.rs:28` | 100 ms sampling loop, `tokio::sync::watch` fan-out |
| `DeviceResources` | `resources.rs:12` | `memory_budget_bytes` / `memory_used_bytes` / `utilization_percent`, all `Option` |
| `vram::Monitor` | `resources/vram.rs:43` | platform dispatch, device selection by vendor+name |
| `vram::unavailable` | `resources/vram.rs:99` | already returns `None` rather than a fabricated zero |
| Windows provider | `resources/windows.rs` | DXGI `QueryVideoMemoryInfo` + NVML for utilization |
| Linux provider | `resources/linux.rs` | NVML, else `/sys/class/drm` amdgpu/xe counters |
| UI transport | `koharu-app/src/commands/lifecycle.rs:118` | `DeviceResources` → `ModelResources` channel |
| Unload-on-pressure | `koharu-pipeline/src/accelerator.rs:39` | `recover()` unloads every other stage, waits ≤600 ms for a fresh sample |

So the fork does **not** need to build measurement from scratch. It needs to
stop treating these numbers as interchangeable, and to add the estimate tier.

### 1.1 Two defects found while reading

**(a) `Device::memory_free` is always `0`.**
Every constructor and every probe hardcodes it — `device.rs:71`, `device.rs:106`,
`hardware/mod.rs:38`, `:50`, `hardware/cuda.rs:95`, `hardware/hip.rs:89`,
`runtime/graph.rs:175`. Nothing ever assigns a real value.

Consequence: `flux2_klein/model.rs:55`

```rust
let keep_parameters_resident = use_accelerator && device.memory_free >= 20 * 1024 * 1024 * 1024;
```

is **always false**. The "high-VRAM cards keep both quantized models resident"
path documented in the comment right below it has never run; every accelerator
gets `params_backend = "*=cpu"`. This is exactly the kind of decision the Budget
Manager should own, so it is listed as work here rather than patched blind.

**(b) `Hardware::discover()` is a `OnceLock`** (`hardware/mod.rs:20`). Whatever it
probed at first call is frozen for the process. `Device` is therefore a
*static description*, never a live reading. Only `ResourceMonitor` is live.

---

## 2. The scope problem: `memory_used_bytes` means four different things

This is the single reason a naive `budget - used >= model_size` check is unsafe.

| platform | source | `budget` | `used` | scope of `used` |
| --- | --- | --- | --- | --- |
| Windows | DXGI `QueryVideoMemoryInfo` (`windows.rs:54`) | OS-assigned budget **for this process** | `CurrentUsage` | **this process only** |
| Linux + NVIDIA | NVML `memory_info` (`linux.rs:68`) | device total | device used | **whole device, all processes** |
| Linux + AMD/Intel | `/sys/class/drm/.../vram*_used_bytes` (`linux.rs:106`) | device total | device used | **whole device, all processes** |
| macOS arm64 | system memory (`vram.rs:74`) | system RAM total | system RAM used | **whole machine, unified memory** |

Reading the same field as "how much VRAM is in use" gives:

- On Linux/macOS: **conservative**. Another process's allocation counts against
  us, so a budget check under-promises. Safe, occasionally annoying.
- On Windows: **optimistic in the wrong direction if read naively**, because
  another process's 8 GB does not appear in `CurrentUsage`. It is *not* actually
  unsafe, because DXGI shrinks `Budget` under external pressure — but that only
  works if the code compares against `Budget`, never against physical capacity.

**Design rule 1.** Headroom is always `budget − used` from the *same sample*, and
never a constant derived from `Device::memory_total`. The two are not
interchangeable, and on Windows only the former reflects contention.

**Design rule 2.** The scope is part of the value. A snapshot carries the scope
tag so the UI can say "this device, all applications" vs "this application", and
so the budget logic can pick a different safety margin per scope.

---

## 3. The tiers

Three tiers, plus an explicit unknown. The brief's §39 wording maps onto them
directly.

### Tier M — Measured
A number this build reads from a driver or OS API on this machine, right now.

- device budget / used / utilization → `ResourceMonitor` (§1)
- **`LlamaModel::size()`** (`koharu-llama/src/model.rs:598` → `llama_model_size`)
  — the loaded model's real byte count as llama.cpp reports it. This is a
  measurement of the loaded object, **not** a guess from the quant name, and is
  what §39 demands instead of `"Q4_K_M ≈ 2.5 GB"`.
- `n_params`, `n_layer`, `n_head_kv`, `n_embd`, `n_ctx_train`
  (`model.rs:603`, `:621`, `:636`, `:593`, `:143`) — real GGUF metadata.

Available only **after** a load. Carries a scope tag and a sample timestamp.

### Tier A — Attributed
A Tier-M delta observed across a load or unload boundary and credited to the
model that crossed it.

Feasible here because `AcceleratorGate` (`accelerator.rs:23`) serialises all
accelerator work onto a one-permit semaphore, so nothing else in Koharu is
allocating during the window. The measurement is real; only the *attribution* is
inferred, and it is polluted by other processes on Linux/macOS (device-wide
scope) but not on Windows (process-local scope).

Attribution is therefore **trustworthy on Windows, indicative elsewhere**, and
must be labelled that way rather than averaged into a single confident figure.

### Tier E — Estimated
Computed from inputs we hold, with the formula stated.

- **Weights, pre-load:** the resolved file's size on disk. We always have the
  path — `HuggingFaceFile::resolve` and `ComponentSource::LocalFile` both yield a
  `PathBuf` before anything is loaded. `fs::metadata().len()` is a real byte
  count of a real file, so this is arithmetic on a measurement, not a guess from
  a name. Its error is the GPU/CPU layer split and metadata overhead, both
  bounded and both stateable.
- **KV cache:** `n_layer × n_ctx × (n_embd_k_gqa + n_embd_v_gqa) × bytes_per_element`,
  with `n_ctx` from the Phase 1 dynamic-context calculation and the element size
  from the Phase 1 `kv_cache_type_k` / `_v` settings. Every input is known
  exactly; the formula is the standard llama.cpp layout. This is the highest-
  quality estimate available and should be labelled as a formula, not a reading.
- **Compute graph / scratch buffers:** not estimable. Depends on batch shape and
  backend internals. Reported as unknown, never as zero.

### Tier U — Unknown
No source. `None`, rendered as "—" with a reason string, exactly as
`vram::unavailable` (`vram.rs:99`) and Phase 1's `LlmCapabilities.deferred`
already do. Never substituted with `0`.

---

## 4. Type design

One wrapper carries provenance with the number so it cannot be laundered by
passing it around.

```rust
/// A byte count together with where it came from. Constructing one requires
/// naming the source, so a number cannot lose its provenance in transit.
pub struct Bytes {
    pub value: u64,
    pub provenance: Provenance,
}

pub enum Provenance {
    /// Read from a driver or OS API.
    Measured { source: MeasuredSource, scope: Scope, sampled_at: Instant },
    /// A measured delta credited to a model that was loaded or unloaded.
    Attributed { scope: Scope, confidence: Confidence },
    /// Computed. `formula` names the calculation for the UI tooltip.
    Estimated { formula: Estimate },
}

pub enum Scope { Process, Device, System }

pub enum MeasuredSource { Dxgi, Nvml, DrmSysfs, SystemMemory, LlamaModel }

/// Windows DXGI is process-local, so a delta there is not polluted by other
/// applications. Every other provider is device-wide.
pub enum Confidence { Isolated, Contended }

pub enum Estimate { FileSize, KvCache { n_ctx: u32 }, Sum }
```

`Option<Bytes>` is Tier U. There is deliberately no `Bytes::raw(u64)`
constructor and no `From<u64>`.

Placement, following the Phase 1/2 precedent: the plain types go in
`koharu-pipeline/src/resources/` (which already owns `DeviceResources`), and the
serde/specta mirror goes in `koharu-app/src/commands/lifecycle.rs` (which already
mirrors `DeviceResources` → the `ModelResources` channel). `koharu-runtime` and
`koharu-ml` stay untouched.

---

## 5. What each tier may be used for

| decision | M | A | E | U |
| --- | --- | --- | --- | --- |
| Display a number to the user | yes, with scope label | yes, with confidence label | yes, marked as an estimate | "—" + reason |
| Refuse to load a model ("won't fit") | no — see below | no | no | no |
| Warn before loading | yes | yes | yes | no |
| Choose GPU layer count | yes | yes | yes | no |
| Decide eviction order under pressure | yes | yes | no | no |
| Pick `params_backend` residency (§1.1a) | yes | yes | yes | no |

**Nothing hard-refuses a load.** The estimate cannot see the compute graph, and
`budget` on Windows moves under us. A refusal built on an incomplete model
produces a product that will not run a job the hardware can actually run. The
Budget Manager warns, and it evicts, but the driver stays the authority on OOM —
which today is already handled by `AcceleratorGate::recover` unloading other
stages and retrying.

**Infinite retry is out** (§39). `recover()` already runs once per acquisition,
not in a loop; the Budget Manager must keep that shape.

---

## 6. UI presentation rules

1. A tier badge or icon on every figure. Estimated figures also carry the
   formula name in a tooltip ("file size", "KV cache at 4096 context").
2. Scope in words, not jargon: "8.1 / 12.0 GB — this device, all applications"
   vs "3.4 GB — Koharu only".
3. Estimated totals are shown as a range or with a `~`, never as a bare exact
   figure next to a measured one at the same visual weight.
4. Unknown renders as "—" plus the reason, matching the `deferred` pattern from
   Phase 1's `LlmRuntimePreferences`.
5. Never sum across tiers into one confident number. A total that mixes a
   measured 3.4 GB with an estimated 2.1 GB is presented as
   "3.4 GB measured + ~2.1 GB estimated", not "5.5 GB".
6. No letter grades, no "safe/unsafe" verdicts on hardware — §39 bars unverified
   grading, and the same reasoning applies here.

---

## 7. Per-model-kind coverage

| model | weights pre-load | weights loaded | KV cache | notes |
| --- | --- | --- | --- | --- |
| Local LLM (llama.cpp) | E, file size | **M**, `LlamaModel::size()` | E, exact formula | best coverage; all Phase 1 settings feed the formula |
| FLUX.2 Klein (sd.cpp) | E, sum of the three resolved files | U | n/a | `koharu-diffusion` exposes no size API; adding one is a `koharu-diffusion-sys` change, out of scope here |
| LaMa / AOT / detection / OCR (Torch, ONNX) | E, file size | U | n/a | no size API exposed |
| Cloud translation providers | n/a | n/a | n/a | no local memory |

The asymmetry is the point: the LLM path can be genuinely accurate, and the doc
says plainly that the others cannot be, rather than inventing parity.

---

## 8. Decisions this design does *not* make

Deliberately deferred to the implementation increment, because they need
numbers from a real machine:

- The safety margin per scope (how much headroom before warning).
- Eviction ordering when several models are resident — current behaviour
  (`unload_other_models`, `accelerator.rs:62`) unloads *everything* else, which
  a budget manager may want to soften.
- Whether the `params_backend` threshold should stay a fixed 20 GiB once
  `memory_free` is real, or become budget-relative.
- Whether to add a size API to `koharu-diffusion` for FLUX weight measurement.

---

## 9. Test plan

Pure functions, no GPU needed:

1. `Bytes` cannot be constructed without a provenance (compile-time, by absence
   of a constructor — asserted by a doc test that fails to compile).
2. Headroom is computed from `budget − used` of one sample; a mixed-sample
   computation is unrepresentable because `Measured` carries `sampled_at`.
3. KV-cache formula against hand-computed values for a known
   `n_layer`/`n_head_kv`/`n_embd`/`n_ctx` set, for each `KvCacheChoice`.
4. Scope propagation: a Windows-sourced delta yields `Confidence::Isolated`, an
   NVML- or DRM-sourced delta yields `Confidence::Contended`.
5. Tier-mixing guard: summing a `Measured` and an `Estimated` produces a
   two-part total, never a single `Measured`.
6. Unknown stays unknown — extending `vram.rs`'s existing
   `unknown_metrics_remain_unknown` test to the new type.
7. Serde/specta round-trip of the mirrored config and snapshot types.

---

## 10. Recommended build order

Steps 1–4 are **done** — see `PHASE-3-NOTES.md` (steps 1–3) and
`PHASE-4-NOTES.md` (step 4) for the decisions taken while implementing them.

1. ~~`Bytes` / `Provenance` types + tests. No behaviour change.~~ **done**
2. ~~Rewrap the existing `DeviceResources` fields in `Bytes` and carry the scope
   through to the UI. Still no behaviour change — this is the honesty pass.~~
   **done**; the scope reaches the UI, the source stays internal.
3. ~~Fix `Device::memory_free` (§1.1a) from the live monitor rather than the
   frozen `Hardware` probe, and make the `params_backend` decision consume
   it.~~ **done**; filled per load rather than written into the `OnceLock`.
4. ~~Add the estimate tier: file sizes at resolve time, KV-cache formula from
   the Phase 1 settings.~~ **done**; both estimates are logged, not yet
   displayed — see `PHASE-4-NOTES.md` D5.
5. Only then: budgets, warnings, and eviction policy. **Next.**

Steps 1–3 were worth shipping on their own — step 3 alone fixed a residency path
that had never executed.
