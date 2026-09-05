# Phase 3, steps 1–3 — provenance-carrying memory figures

Implements steps 1–3 of `VRAM-ACCOUNTING-DESIGN.md` §10. Steps 4 (estimate tier)
and 5 (budgets, warnings, eviction) are not in this increment.

## D1 — `Bytes` lives in `koharu-pipeline`, not `koharu-runtime`

`koharu-runtime`'s `Device` is a frozen `OnceLock` probe; `koharu-pipeline`'s
`ResourceMonitor` is the only live reading in the tree. Putting the type next to
the live sampler keeps `koharu-runtime` and `koharu-ml` free of it, matching the
Phase 1/2 rule that plain crates gain no new dependencies.

## D2 — `MemoryScope`, not `Scope`

`koharu_pipeline::Scope` already exists as the canvas bounds type. The memory
scope is a separate concept and takes a separate name rather than shadowing it.

## D3 — Headroom is refused across samples, sources, or scopes

`Bytes::headroom_from` returns `None` unless both figures came from the same API,
covering the same scope, at the same instant. A budget from one sample minus a
usage figure from another describes no moment that existed, and on Windows the
DXGI budget moves under external pressure between samples. Refusing is cheaper
than a plausible-looking wrong number.

## D4 — The scope reaches the UI, the source does not (yet)

`DeviceResources.memory_scope` is mirrored to the frontend and rendered in words
("this device, all applications"). `MeasuredSource` stays internal: naming DXGI
or NVML to the user adds nothing they can act on, while the scope changes how
they should read the number. Adding the source later is additive.

## D5 — `memory_free` is filled at load time, not at discovery

`Hardware::discover()` is a `OnceLock`, so writing a real value into it would
freeze one reading for the process. Instead `inpainting::Processor` clones its
device description and fills `memory_free` from the latest monitor sample each
time a model is loaded. The value is a real headroom reading in the provider's
scope; before the first sample, and on platforms with no provider, it stays `0`
and the FLUX residency check behaves exactly as it did before.

Consequence: `flux2_klein/model.rs:55`'s `keep_parameters_resident` check can now
be true. On a card reporting ≥20 GiB of free memory, FLUX no longer passes
`params_backend = "*=cpu"` and keeps both quantized models on the accelerator —
the behaviour the comment there has always described but which had never run.

## D6 — The 20 GiB threshold is unchanged

`VRAM-ACCOUNTING-DESIGN.md` §8 leaves open whether the threshold should become
budget-relative. Changing the number and making it reachable in the same
increment would make a regression impossible to attribute. The threshold stays as
upstream wrote it.

## Compatibility

- `ModelResources`/`DeviceResources` gained one field. The channel payload is
  additive; no config or project format is touched, so nothing needs migration.
- `koharu_pipeline::DeviceResources`'s two memory fields changed from
  `Option<u64>` to `Option<Bytes>`. The crate is not published, and the only
  consumer is `koharu-app`, which was updated.
- Behaviour is unchanged everywhere except D5.

## Not in this increment

| deferred | why |
| --- | --- |
| Estimate tier (file sizes, KV-cache formula) | design §10 step 4 |
| Budgets, warnings, eviction ordering | design §10 step 5; needs numbers from real hardware |
| `Provenance::Attributed` (load-boundary deltas) | needs the load/unload hooks step 5 introduces |
| Showing `MeasuredSource` in the UI | D4 |
| Making the FLUX residency threshold budget-relative | D6, design §8 |
