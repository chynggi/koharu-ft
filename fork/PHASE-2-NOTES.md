# Phase 2 — FLUX.2 Klein configurability (brief §13–14)

Follows `PHASE-1-PLAN.md`. Phase 1's deferral note said this increment needed
its own pass because `TRANSFORMER_WEIGHTS` and friends are compile-time
`const HuggingFaceFile` values, and because the 1 MP working area was hardcoded
at three separate call sites.

## What shipped

1. **Configurable checkpoints (§13).** Each of the three FLUX.2 Klein components
   — transformer, text encoder, VAE — can now come from the pinned repository
   (default), a file already on disk, or any Hugging Face repository.
2. **Inference settings (§14).** Steps, strength, seed, mask-crop padding and the
   working area are configuration, no longer `Flux2KleinInpaintOptions::default()`.
3. **One owner for the working area.** `Flux2ImageProcessor::fit_to_area` replaced
   the three copies of `if w * h > 1024 * 1024 { … }`.

## Design decisions

### D1 — the plain type lives in `koharu-ml`, the serde type in `koharu-pipeline`

`ComponentSource` / `Flux2KleinSource` (`koharu-ml/src/flux2_klein/source.rs`)
carry no serde or specta derives. `ComponentSourceConfig` /
`Flux2KleinSourceConfig` (`koharu-pipeline/src/stages/inpainting.rs`) carry both
and convert via `From`.

This is the shape the crate already had: `Flux2KleinConfig` and
`RoremMixedConfig` were already pipeline-local serde types feeding plain
`koharu-ml` option structs. It also keeps `koharu-ml` off `specta`, which it does
not currently depend on. The cost is one mirrored enum, which is why the mirror
is a direct 1:1 `From` with no logic in it.

Same reasoning as Phase 1's split between `koharu-llama` (plain) and
`koharu-translator` (serde/specta).

### D2 — `kind`-tagged enums again

`ComponentSourceConfig` is `#[serde(tag = "kind")]` for the same reason the
Phase 1 enums are: `koharu_config`'s merge replaces a subtree when a tag field
differs, and `kind` keeps that behaviour predictable and away from the
`model` / `provider` tags the pipeline already uses.

### D3 — overrides are per component, not a model preset

There is no "Klein 9B" entry, and no preset list. A preset asserts that a
checkpoint works; a per-component override only says which file to hand to
stable-diffusion.cpp. Brief §39 forbids shipping a path swap as if it were
verified support, so the fork ships the mechanism and not the claim.

### D4 — validation runs when settings are saved, not mid-inference

`Flux2KleinConfig::validate` is called from `Processor::new` (where the NUL check
already lived) and covers steps, strength, working area, absolute-and-existing
local paths, `owner/name` repository shape, and full-commit-hash revisions.
`ComponentSource::resolve` re-validates before touching the network, because a
config can go stale between saving and loading.

Revisions must be a full 40-character commit hash. `HuggingFaceFile::pinned`
already rejects anything else at resolve time; validating it early turns a
mid-inference failure into a settings error. A branch name is not silently
accepted — leave the revision blank to track the repository head instead.

### D5 — `steps: u32` and `#[specta(type = f64)]` on `seed`

specta refuses to export `usize`/`i64` (BigInt precision). `steps` became `u32`
and is widened at the call site; `seed` keeps `i64` in Rust with an explicit
`f64` TypeScript type, matching the `total_memory` annotation added in Phase 1.

### D6 — the debug binary gained the same overrides

`cargo run -p koharu-ml --bin flux2_klein -- --transformer <file> …` is the only
way to test a swapped checkpoint without a UI round-trip, and §13 requires
verification before any claim of support. It also gained `--max-pixels`.

## Klein 9B — still not shipped, and why

Brief §13 forbids implementing 9B by swapping a path. That has not changed:

- `Model::new` (`flux2_klein/model.rs`) hands three file paths and
  `VaeFormat::Flux2` to stable-diffusion.cpp, which infers the architecture from
  GGUF metadata. Whether a 9B transformer loads is a property of the **pinned
  stable-diffusion.cpp build**, not of Koharu's Rust code.
- `FLUX.2-small-decoder` is paired with the 4B transformer. Whether its latent
  width matches a 9B transformer has not been verified against the upstream
  model card or the pinned backend.

Neither question can be answered by reading Koharu's source, so no 9B option,
preset, or documentation claim ships. A user may point the transformer override
at a 9B GGUF; if the pinned backend cannot build a context for it, `Model::new`
fails with stable-diffusion.cpp's own error rather than producing wrong output.

## Compatibility

- `Flux2KleinConfig` is `#[serde(default)]` and every new field has a default, so
  a `[processor.flux2-klein]` section written before this change still loads.
  Covered by `a_flux_section_written_before_the_settings_existed_still_loads`.
- The defaults reproduce the previous behaviour exactly:
  `Flux2KleinConfig::default().options() == Flux2KleinInpaintOptions::default()`
  and the default source resolves the same three pinned repositories. Covered by
  `flux_defaults_reproduce_the_previous_inference_options`.
- In-repo API break: `Flux2Klein::load` and `Flux2KleinInpaint::load` now take a
  `&Flux2KleinSource`. All three callers (pipeline stage, debug binary, bench)
  were updated in the same change.
- Project files (`.khrproj`) are untouched.

## Not in this increment

| item | reason |
| --- | --- |
| Text CFG / scheduler / sample method | Still hardcoded (`text_cfg: 1.0`, `Flux2`, `Euler`). These are architecture choices from the diffusers reference pipeline, not user preferences; exposing them invites silently broken output. |
| Non-inpaint FLUX settings in the UI | `Flux2Klein` (generation, not inpainting) is only reachable from the debug binary today. |
| VRAM-based residency switch (`memory_free >= 20 GiB`) | Belongs to the VRAM Budget Manager increment, which owns residency policy across every model. |
| Klein 9B | See above. |
