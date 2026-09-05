# HANDOFF — 2026-08-20

Where the fork stands, what is verified, and what was interrupted mid-flight.

Baseline: **0.73.0 plus the koharu-rpc port**, on branch `fork/vram-accounting`
in the `D:\koharu-ft-serve` clone. The fork was written against upstream 0.70.3
(`db439495`) and lived entirely in the working tree of the old
`C:\Users\chyng\koharu-ft` clone; on 2026-08-20 it was committed as-is and
rebased 30 commits forward. §8 records what that cost.

---

## 1. Position in the plan

| phase | scope | state |
| --- | --- | --- |
| Phase 1 | Custom GGUF, LLM advanced settings, dynamic context, model profile | complete, verified |
| Phase 2 | FLUX.2 Klein per-component sources + inference options | complete, verified |
| Phase 3 design | `VRAM-ACCOUNTING-DESIGN.md` — measured/estimated split | complete |
| Phase 3 steps 1–3 | `Bytes`/provenance, scope to the UI, live `memory_free` | complete, verified |
| Phase 3 step 4 | estimate tier — file sizes + KV-cache formula | complete, verified |
| **Phase 3 step 5** | **budgets, warnings, eviction policy** | **next — not started** |

Read in this order to pick the work up: `VRAM-ACCOUNTING-DESIGN.md` §3–§4 and
§10, then `PHASE-3-NOTES.md`, then §3 below.

---

## 2. Verification sweep — run, and it caught something

The step-4 sweep was interrupted once and has since been completed:

```
cargo test -j 6 -p koharu-pipeline --lib      65 passed
cargo test -j 6 -p koharu-ml --lib            61 passed, 16 ignored
cargo check -j 6 --workspace                  clean, no new warnings
```

The first run **failed** on `a_real_file_is_sized_from_disk`: it anchored on
`file!()`, which is relative to the workspace root while a test runs from the
package root. Fixed to anchor on `CARGO_MANIFEST_DIR`. Worth recording as
evidence that the sweep is not a formality.

`-j 6` is mandatory on this machine — unbounded cargo parallelism freezes the
desktop (12 cores).

No frontend work landed in step 4, so `protocol.ts`, vitest, `tsc`, and oxlint
are unaffected by it and remain green from step 3.

---

## 3. Step 4 — what is written

### New: `crates/koharu-ml/src/llm/footprint.rs`

- `ElementSize { bytes_per_block, elements_per_block }` — ggml block layout, kept
  whole so quantized caches stay in integer arithmetic. `ElementSize::F16` is the
  llama.cpp default when no cache type is configured.
- `ElementSize::for_kv_cache_type(KvCacheType) -> Option<Self>` — covers exactly
  the nine types llama.cpp accepts for a KV cache (F32, F16, BF16, Q8_0, Q5_1,
  Q5_0, Q4_1, Q4_0, IQ4_NL), which are exactly the nine `KvCacheChoice` offers.
  `None` for anything else rather than a guessed size.
- `KvGeometry { n_layer, n_head_kv, key_length, value_length }` with
  `read(&LlamaModel)` and `cache_bytes(n_ctx, key, value)`.
  `read` takes the head dimensions from GGUF `{arch}.attention.key_length` /
  `value_length`, falling back to `n_embd / n_head`. **This matters:** the
  fallback is wrong for Gemma and other architectures whose head dimension does
  not divide the embedding, so the stated metadata wins where present, and `read`
  returns `None` when neither source yields a positive length.
- **7 tests, all passing** (`cargo test -j 6 -p koharu-ml --lib llm::footprint`).
  One caught a bad hand-computed literal: Llama-3-8B at 4096 ctx f16 KV is
  512 MiB, not 1 GiB.

### Modified: `crates/koharu-ml/src/llm/mod.rs`, `.../llm/model.rs`

- `Llm::weights_bytes()` → `llama_model_size`, a **measurement** of the loaded
  tensors. This is what §39 demands instead of inferring size from a quant name.
- `Llm::kv_geometry() -> Option<KvGeometry>`.
- `ContextConfig` gained `n_ctx: u32` (it previously only kept `params` and
  `n_batch`, and the estimate needs the resolved dynamic context).
- `Model::log_kv_cache_estimate` runs just before `new_context` and emits a
  `tracing::debug!` with `n_ctx`, `weights_bytes`, `estimated_kv_cache_bytes`.
  **The estimate is logged, not displayed** — see §5.

### Modified: `crates/koharu-ml/src/flux2_klein/mod.rs`

- `pub async fn resolve_paths(&Flux2KleinSource) -> Result<[PathBuf; 3]>`, a thin
  wrapper over the existing private `resolve`, so a caller can size a checkpoint
  before loading it.

### Modified: `crates/koharu-pipeline/src/resources/bytes.rs`

- `Estimate::KvCache { n_ctx }` added alongside `FileSize`.
- `pub fn file_size(&Path) -> Option<Bytes>` — `fs::metadata().len()` wrapped as
  `Estimate::FileSize`. `None` (never `0`) when the file cannot be stat'd.
- 2 new tests (`a_missing_file_has_no_size_rather_than_a_zero`,
  `a_real_file_is_sized_from_disk`), both passing.

### Modified: `crates/koharu-pipeline/src/stages/inpainting.rs`

- `log_weight_estimate(&Flux2KleinSource)` resolves the three component paths and
  logs their summed size before the FLUX load. Resolution is a cache lookup once
  the files are present, so the second resolve costs a stat, not a download.

---

## 4. Two defects found earlier, one fixed

**Fixed in step 3.** `Device::memory_free` was hardcoded `0` in every constructor
and probe (`device.rs:71`, `:106`, `hardware/mod.rs:38`, `:50`, `cuda.rs:95`,
`hip.rs:89`, `runtime/graph.rs:175`). `flux2_klein/model.rs:55`'s
`keep_parameters_resident` could therefore never be true, and the documented
"high-VRAM cards keep both quantized models resident" path had **never executed**.
`inpainting::Processor::device_for_load` now fills `memory_free` from the live
`ResourceMonitor` at each load. See `PHASE-3-NOTES.md` D5.

**Still open, by design.** `Hardware::discover()` is a `OnceLock`
(`hardware/mod.rs:20`), so `koharu_ml::Device` is a frozen first-call probe, not a
live reading. Step 3 works around it per-load rather than changing the `OnceLock`
— writing a real value into it would freeze one sample for the process lifetime.

---

## 5. The one open judgement call

Step 4's estimates currently only reach `tracing::debug!`. They are computed
correctly but **no user sees them**. That was deliberate: the design's step 5
owns the UI surface (warnings, budgets), and shipping a number without the
scope/tier badges §6 of the design specifies would be the exact §39 violation the
whole tier system exists to prevent.

If step 4 should ship a visible surface before step 5, the cheapest honest one is
the custom-GGUF registration flow — `pick_gguf_file` already returns a path, so a
sibling command could return `file_size` for it and the UI could show
"~4.1 GB (file size)" next to the model. That is a decision, not an oversight.

---

## 6. Verification status by phase

Green as of the last run:

| check | result | last run |
| --- | --- | --- |
| `cargo test -j 6 -p koharu-pipeline --lib` | 65/65 | rebase |
| `cargo test -j 6 -p koharu-ml --lib` | 61 passed, 16 ignored | rebase |
| `cargo check -j 6 --workspace` | clean, no new warnings | rebase |
| `tsc --noEmit -p packages/koharu/tsconfig.json` | exit 0 | rebase |
| `oxlint packages/koharu` | clean | rebase |
| `bun run --filter '@koharu/app' test` | 84/84 across 12 files | rebase |

The whole table was re-run after the rebase and matches step 4's numbers. The
vitest count moved 81 → 84 because the 30 upstream commits added three tests,
not because anything here changed.

The only pre-existing warning in the tree is `koharu-torch-sys`'s build-script
`linker_messages` one. It is not ours.

### Never verified, on any phase

**No manual end-to-end run has happened.** This machine has no `.gguf` and no
FLUX checkpoint, so none of the following has been exercised against real
weights:

- registering a custom GGUF and translating a page (Phase 1)
- pointing a FLUX component at a different checkpoint and inpainting (Phase 2)
- `keep_parameters_resident` actually becoming true on a ≥20 GiB card (step 3 D5)
- `KvGeometry::read` against a real GGUF's metadata (step 4)

The first thing a machine with real weights should do is run those four.

---

## 7. Immediate next actions

1. Decide §5 — does the estimate tier ship a visible surface now, or wait for
   step 5's warnings to carry it?
2. Step 5: budgets, warnings, eviction. Design §8 lists what it must decide
   that this design deliberately did not — safety margin per scope, eviction
   ordering (today `unload_other_models` unloads *everything* else), whether the
   FLUX residency threshold stays a fixed 20 GiB now that it is reachable, and
   whether `koharu-diffusion` gets a size API.

Constraints that still govern step 5: nothing hard-refuses a load (the driver
stays the OOM authority), no infinite retry, no summing across tiers into one
confident number, and no letter grades on hardware.

---

## 8. The rebase onto 0.73.0 + koharu-rpc

The 30 intervening commits touched 20 of the fork's 52 files, but only two
conflicted.

**`crates/koharu-app/src/commands/mod.rs`** — upstream turned every
`pub(crate) mod` into `pub mod` so koharu-rpc could call the command functions
directly, and added `import`. Resolved to upstream's list plus `pub mod llm;`.

A second rebase followed on 2026-08-20, after main removed the dead
`bindings()` command table along with `bin/generate.rs` (which regenerated
`protocol.ts` from tauri-specta and would have destroyed the hand-written HTTP
client — see the root `HANDOFF.md`). The fork's two `llm::` entries in that
table went with it; `pub mod llm;` stays, and `routes/llm.rs` calls the two
functions directly, so nothing is lost.

**`packages/bridge/src/protocol.ts`** — this is the one that matters. At 0.70.3
the file was tauri-specta's generated IPC bridge; the RPC port replaced it with
a hand-written fetch/SSE client that keeps the same `commands` shape. Upstream's
version wins outright; the fork's contribution to it is four items, re-added by
hand: the `GpuLayers` and `MemoryScope` types, and the `getLlmCapabilities` /
`pickGgufFile` entries.

**The transport gap this exposed.** `get_llm_capabilities` and `pick_gguf_file`
were Tauri commands with no HTTP route, and `LocalModelPreferences` calls both.
`crates/koharu-rpc/src/routes/llm.rs` now serves them. `GET /llm/capabilities`
is a straight port. `POST /llm/gguf-file` opens the native picker **only for a
loopback caller** and returns `null` otherwise, because the dialog runs in the
server process: in the container that display is a headless Xvfb, so a remote
caller would get a dialog nobody can see and a request that never returns. The
UI falls back to typing a path, which is the right shape regardless — llama.cpp
loads the weights server-side, so the path must be one koharu-rpc can read.

**Still unexercised.** The four items in §6's "never verified" list are
unchanged, and `/llm/gguf-file` has never been hit from a real remote browser.
