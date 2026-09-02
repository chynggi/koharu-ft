# Phase 3, step 4 — the estimate tier

Implements step 4 of `VRAM-ACCOUNTING-DESIGN.md` §10. Step 5 (budgets, warnings,
eviction) is not in this increment.

## D1 — Weights are measured for the LLM, estimated for everything else

`llama_model_size` reports the real byte count of the loaded tensors, so
`Llm::weights_bytes()` is Tier M — a measurement of the loaded object. This is
what §39 demands instead of `"Q4_K_M ≈ 2.5 GB"`.

FLUX has no equivalent: `koharu-diffusion` exposes no size API, and adding one is
a `koharu-diffusion-sys` change. Its weights are therefore Tier E, the summed
size of the three resolved files on disk. The asymmetry is stated rather than
papered over.

## D2 — The head dimension comes from GGUF metadata, not from `n_embd / n_head`

The obvious KV formula uses `n_embd / n_head` for the head width. That is wrong
for any architecture whose head dimension does not divide the embedding evenly —
Gemma being the common case. `KvGeometry::read` reads
`{arch}.attention.key_length` and `value_length` first and only falls back to the
division, returning `None` when neither yields a positive length rather than
substituting a plausible number.

## D3 — Cache types are sized by their ggml block, not by their name

A quantized KV cache stores 32 elements per block with shared scales, so
bytes-per-element is fractional. `ElementSize` keeps the block whole
(`bytes_per_block`, `elements_per_block`) and the arithmetic stays in integers,
with partial blocks rounded up. `for_kv_cache_type` covers exactly the nine types
llama.cpp accepts for a cache — which are exactly the nine `KvCacheChoice`
offers — and returns `None` for anything else.

## D4 — The mapping lives in `koharu-ml`, not on `KvCacheType`

`KvCacheType` is upstream's full ggml type enum, most of whose variants can never
be a KV cache. Growing it with a block table would mean covering variants that
have no meaning here. The mapping is a fork-owned function instead, per §37.

## D5 — Estimates are logged, not displayed

Both estimates reach `tracing::debug!` and no further. The design's §6 requires a
tier badge, a scope in words, and a `~` on every estimated figure before one is
shown; that presentation layer belongs to step 5, which owns the warnings the
numbers exist to support. Shipping a bare number ahead of its badge is precisely
the §39 failure the tier system was built to prevent.

If a visible surface is wanted before step 5, the cheapest honest one is the
custom-GGUF registration flow: `pick_gguf_file` already returns a path, so a
sibling command could return its `file_size` and the UI could show
"~4.1 GB (file size)" beside the model.

## D6 — `resolve_paths` rather than threading paths out of the loader

Sizing a FLUX checkpoint needs the three resolved paths, which previously existed
only inside the private `resolve`. Exposing a thin `resolve_paths` wrapper is
smaller than changing `Flux2KleinInpaint::load`'s shape to hand them back. Once
the files are present, resolving is a cache lookup, so the second call costs a
stat and not a download.

## Compatibility

No serde, specta, or config types changed. `protocol.ts` is untouched and no
frontend code was involved. `ContextConfig` gained a private field. Behaviour is
unchanged: nothing in this increment alters a decision, only what gets logged.

## Tests

- `cargo test -j 6 -p koharu-ml --lib` — **61 passed, 16 ignored**, including 7
  new `llm::footprint` tests.
- `cargo test -j 6 -p koharu-pipeline --lib` — **65 passed** (63 from step 3 plus
  2 for `file_size`).
- `cargo check -j 6 --workspace` — clean, no new warnings.

Two mistakes the tests caught, both in the tests themselves rather than the code:

1. A hand-computed literal claimed Llama-3-8B at 4096 context with an f16 KV cache
   costs 1 GiB. It is 512 MiB. The formula was right; the assertion was not.
2. `a_real_file_is_sized_from_disk` anchored on `file!()`, which is relative to
   the workspace root while a test runs from the package root. It now anchors on
   `CARGO_MANIFEST_DIR`.

## Not in this increment

| deferred | why |
| --- | --- |
| Any UI surface for an estimate | D5 |
| `Provenance::Attributed` (load-boundary deltas) | needs the load/unload hooks step 5 introduces |
| A size API for `koharu-diffusion` | D1; design §8 leaves it open |
| Budgets, warnings, eviction ordering | design §10 step 5 |

## Never verified

`KvGeometry::read` has not run against a real GGUF — this machine has no model
files. The formula and the block table are unit-tested; the metadata read is not.
