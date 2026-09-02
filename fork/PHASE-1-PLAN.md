# Phase 1 — Power User Foundation: implementation plan

Scope for the **first shipping increment**, per brief §43:

1. Custom GGUF registration
2. LLM Advanced Settings
3. Dynamic Context preserved as the default
4. Model Profile persistence

FLUX advanced settings, VRAM Budget Manager, hardware profiles and pipeline profiles
are Phase 1 items in the brief's §38 list but are explicitly sequenced *after* this
increment is working (§43 final paragraph). They are planned in §7 below, not built yet.

Read `fork/ARCHITECTURE.md` first — this plan assumes its file/line anchors.

---

## 1. Design decisions (and why)

### D1 — New settings live in `[providers.local]`, not a new section

`LocalConfig` (`crates/koharu-translator/src/local/catalog.rs:35`) is
`pub struct LocalConfig {}` — an empty, already-registered, already-serialized
placeholder for the local provider. It reaches the runtime for free: `Translator` holds
`providers: Config<ProvidersConfig>` (`lib.rs:28`), so `Translator::local()` can read it
without new wiring.

Rejected alternative: a new top-level `[llm]` section. It would need its own
`koharu_config::load` call, its own change subscription, and a second path into
`LocalTranslator`. Extra machinery for no benefit — `[providers.local]` is exactly the
scope of this configuration.

`koharu-config`'s deep merge (`lib.rs:437`) makes every field we add automatically
backward compatible: existing `config.toml` files with no `[providers.local]` keys get
defaults, and unknown keys are preserved. **No migration code is needed for Phase 1.**

One caveat from that merge function: it replaces a whole subtree when a `provider` or
`model` **tag** changes (line 440). Our nested tables must therefore not use a bare
`model` key as a discriminant. We use `kind` for tagged enums instead.

### D2 — Context bounds go into `koharu-ml`, context *modes* stay in `koharu-translator`

`context_values` (`crates/koharu-ml/src/llm/model.rs:620`) stays policy-free. It gains
two optional bounds:

```rust
pub n_ctx:     Option<NonZeroU32>,   // existing — fixed size, must be >= required
pub n_ctx_min: Option<NonZeroU32>,   // new — raise a dynamic context to at least this
pub n_ctx_max: Option<NonZeroU32>,   // new — refuse if the dynamic context exceeds this
```

resolving as:

```
required = max(prompt_positions + max_tokens + 1,
               prompt_tokens    + max_tokens + 1,
               batch_tokens     + 1)                       // UNCHANGED

n_ctx = match n_ctx {
    Some(fixed) if fixed >= required => fixed,             // UNCHANGED
    Some(_)                          => error              // UNCHANGED message
    None => {
        let n = max(required, n_ctx_min.unwrap_or(0));
        ensure!(n_ctx_max.is_none_or(|m| n <= m), …);
        n
    }
}
```

With both bounds `None` this is **byte-for-byte the current behaviour**, which is the
§3 requirement. The four UI modes map on top:

| UI mode | `n_ctx` | `n_ctx_min` | `n_ctx_max` |
| --- | --- | --- | --- |
| Dynamic (default) | None | None | None |
| Fixed | `Some(n)` | None | None |
| Dynamic + Minimum | None | `Some(n)` | None |
| Dynamic + Maximum | None | None | `Some(n)` |

`Dynamic + Minimum` and `Dynamic + Maximum` can be combined; the enum carries both
optional bounds so the UI can offer either or both.

**Deliberate decision on `Dynamic + Maximum` overflow:** when `required > max` we
**error** with an actionable message rather than silently lowering `max_tokens`.
Lowering it would truncate the schema-constrained JSON output and surface later as an
opaque "returned N translations for M segments" failure (`prompt.rs:36`). An explicit
"this page needs N positions, your maximum is M — raise the maximum or lower Max Output
Tokens" is the honest failure.

### D3 — `GenerationOptions` loses its unused serde derives

KV-cache type and flash-attention policy must reach `LlamaContextParams`. The natural
types are `koharu_llama::context::params::KvCacheType` (already exists,
`context/params.rs:159`) and a new `LlamaFlashAttentionType` sibling of the existing
`LlamaAttentionType`. Neither can derive `serde` without adding a serde dependency to
`koharu-llama`, which is a `publish = true` FFI-boundary crate with no serde today.

`GenerationOptions` derives `Serialize, Deserialize` (`llm/mod.rs:245`) but **nothing in
the workspace serializes it** (verified: only construction sites exist). Per upstream's
own rule "remove dead abstractions", we drop the derives and use the llama types
directly.

The serde/specta surface is not lost — it moves to where it is genuinely needed: the
config types in `koharu-translator` (`KvCacheChoice`, `FlashAttentionMode`, …), which
need `specta::Type` for the UI anyway and therefore could never have reused
llama-cpp types.

### D4 — Custom models sit beside the static catalog, not inside it

`catalog::MODELS` stays a `&'static [LocalModelDescriptor]` and is not touched.
A new module `local/registry.rs` owns resolution:

```rust
enum LocalModel {
    Builtin(LocalModelDescriptor),   // downloads from Hugging Face
    Custom(CustomModel),             // already on disk, never downloaded or modified
}
```

with one `resolve(&self, selection, config) -> ResolvedLocalModel` producing
`{ model: PathBuf, projector: Option<PathBuf>, generation: ModelGeneration,
target_languages: SupportedLanguages }`.

Built-in ids win on collision, and registering a custom model whose id shadows a
built-in one is rejected at save time with a clear error.

Custom models declare `target_languages: All` and `generation: ModelGeneration::default()`
(every field `None`). We do **not** guess a temperature for a model we know nothing about;
the user's `GenerationConfig` governs entirely. This is the §39 "don't fabricate ratings"
rule applied to sampling defaults.

The existing vision consistency check in `LocalTranslator::load` (`local/mod.rs:44`) —
`llm.capabilities().vision != descriptor.projector.is_some()` — is kept for custom models.
It is what catches "user pointed at an mmproj that isn't one".

### D5 — Settings precedence

```
GenerationOptions::default()                 (koharu-ml)
  ↓ catalog descriptor.generation             (built-in tuning; empty for custom)
  ↓ providers.local.runtime                   (global LLM runtime settings)
  ↓ providers.local.profiles["<model-id>"]    (per-model override)
  ↓ pipeline.translation.generation           (per-run, what the UI edits today)
```

Lower rows win. This is the §34 layering restricted to what Phase 1 owns; hardware and
project layers slot in above `runtime` in Phase 2/3 without re-ordering anything.

`ModelGeneration::options()` (`koharu-translator/src/model.rs:108`) is the single place
this merge happens today and stays the single place afterwards — it grows a
`runtime: &ResolvedRuntime` parameter.

### D6 — Capability detection is reported, never guessed

A new `LlmCapabilities` describes what the *current* build and device actually support:

| capability | source of truth |
| --- | --- |
| `gpu_offload` | `LlamaBackend::supports_gpu_offload()` && `device.backend != Cpu` (the exact condition `model_params` already applies at `llm/model.rs:577`) |
| `gpu_layers` | same as above |
| `flash_attention` | the active `runtime::packages::Llama` variant (CUDA / HIP / Vulkan / Metal); reported as `Supported` / `Unknown`, never as `Unsupported` on a guess |
| `kv_cache_quantization` | `Supported` only when flash attention is available — quantized V cache requires it on the mainstream backends; otherwise `RequiresFlashAttention` with that reason string |
| `threads` | always |

Where we cannot determine support, the value is `Unknown` and the UI enables the control
with a caveat rather than disabling it on a guess. Disabling a working option is worse
than letting llama.cpp report the real error.

---

## 2. File-by-file change list

### New files

| path | contents |
| --- | --- |
| `crates/koharu-translator/src/local/registry.rs` | `LocalConfig` (extended), `LlmRuntimeConfig`, `ContextMode`, `GpuLayers`, `KvCacheChoice`, `FlashAttentionMode`, `CustomModel`, `LocalModel` resolution, validation, precedence merge, unit tests |
| `crates/koharu-app/src/commands/llm.rs` | `get_llm_capabilities`, `inspect_gguf`, `add_custom_model`, `remove_custom_model` |
| `packages/koharu/components/preferences/LlmRuntimePreferences.tsx` | the §4 Advanced Settings panel |
| `packages/koharu/components/preferences/CustomModelDialog.tsx` | the §5 "Add Local Model" form |
| `fork/ARCHITECTURE.md`, `fork/PHASE-1-PLAN.md` | this analysis (already written) |

### Modified files

| path | change | size |
| --- | --- | --- |
| `crates/koharu-llama/src/context/params.rs` | add `LlamaFlashAttentionType { Auto, Enabled, Disabled }` + `From`/`Into` for `llama_flash_attn_type` | ~35 lines, mirrors existing `LlamaAttentionType` |
| `crates/koharu-llama/src/context/params/get_set.rs` | `with_flash_attention_policy` takes the safe enum instead of the raw `-sys` type | ~6 lines |
| `crates/koharu-ml/src/llm/mod.rs` | `GenerationOptions`: drop serde derives, add `n_ctx_min`, `n_ctx_max`, `kv_cache_type_k`, `kv_cache_type_v`, `flash_attention`; re-export `KvCacheType`, `LlamaFlashAttentionType` | ~25 lines |
| `crates/koharu-ml/src/llm/model.rs` | `context_values` applies min/max; `context_config` applies KV types + flash policy; new tests | ~35 lines |
| `crates/koharu-translator/src/local/catalog.rs` | move `LocalConfig` out to `registry.rs`; `MODELS` untouched | ~5 lines |
| `crates/koharu-translator/src/local/mod.rs` | resolve through `registry`; build `LoadOptions` from config (`gpu_layers`, `load_mode`); pass runtime settings into `ModelGeneration::options` | ~60 lines |
| `crates/koharu-translator/src/model.rs` | `ModelGeneration::options(overrides, runtime)` | ~25 lines |
| `crates/koharu-translator/src/lib.rs` | pass `LocalConfig` into `LocalTranslator::load`; include custom models in `models()`; `supports_vision` consults the registry; `LoadedLocal::matches` also compares the runtime revision so settings changes reload | ~35 lines |
| `crates/koharu-app/src/commands/mod.rs` | register the four new commands | 5 lines |
| `crates/koharu-app/src/commands/preferences.rs` | validate `providers.local` on save | ~10 lines |
| `packages/koharu/components/preferences/SettingsPage.tsx` | new "Models" tab | ~15 lines |
| `packages/koharu/components/preferences/TranslationPreferences.tsx` | show custom models in the picker; per-model profile entry point | ~20 lines |
| `packages/bridge/src/protocol.ts` | **regenerated**, never hand-edited | — |
| `packages/koharu/lib/i18n.ts` + locale files | new strings | — |

Nothing in `koharu-pipeline`, `koharu-scene`, `koharu-storage`, `koharu-runtime` changes
in this increment.

---

## 3. Config shape after Phase 1

```toml
[providers.local]

  [providers.local.runtime]
  # Context
  context = { kind = "dynamic" }            # dynamic | fixed | bounded
  # context = { kind = "fixed", size = 8192 }
  # context = { kind = "bounded", minimum = 4096, maximum = 32768 }
  max_output_tokens = 1000                  # feeds GenerationOptions.max_tokens

  # Batching — omitted means Auto (current behaviour)
  # n_batch = 2048
  # n_ubatch = 512

  # Offload
  gpu_layers = { kind = "all" }             # auto | all | custom
  # gpu_layers = { kind = "custom", layers = 24 }
  # n_threads = 8
  # n_threads_batch = 8

  # KV cache / attention — omitted means llama.cpp default
  # kv_cache_type_k = "q8_0"
  # kv_cache_type_v = "q8_0"
  flash_attention = "auto"                  # auto | on | off

  # Per-model overrides, same shape as [providers.local.runtime]
  [providers.local.profiles."custom-qwen-manga"]
  gpu_layers = { kind = "custom", layers = 40 }

  # Registered local GGUF files
  [[providers.local.models]]
  id = "custom-qwen-manga"
  name = "Qwen Manga Translator"
  path = 'D:\AI\Models\model.gguf'
  # projector = 'D:\AI\Models\mmproj-F16.gguf'   # optional, enables vision
```

`kind` is the tag on every nested enum, deliberately avoiding `model`/`provider` so
`koharu_config::merge`'s subtree-replacement rule (`lib.rs:440`) does not fire on these
tables.

---

## 4. Build order (each step is independently checkable)

| # | step | verification |
| --- | --- | --- |
| 1 | `LlamaFlashAttentionType` in `koharu-llama` | `cargo test -p koharu-llama` |
| 2 | `GenerationOptions` fields + `context_values` bounds + `context_config` wiring | `cargo test -p koharu-ml --lib llm` — new tests assert Dynamic is unchanged, Fixed still rejects small values, Min raises, Max errors past the cap |
| 3 | `registry.rs`: config types, validation, precedence merge | `cargo test -p koharu-translator --lib local` — TOML round-trip, id collision, precedence order, path validation |
| 4 | `local/mod.rs` + `lib.rs`: custom model resolution and `LoadOptions` | `cargo test -p koharu-translator` |
| 5 | Tauri commands + `cargo run -p koharu-app --bin generate` | `cargo check -p koharu-app`, `bun run --filter @koharu/bridge typecheck` |
| 6 | React panels | `bun run lint`, `bun run test` |
| 7 | Manual: register a real `.gguf`, translate a page, confirm the Diagnostics readout | app run |

Per upstream `AGENTS.md`, each step runs the **smallest relevant debug-profile check**,
not the whole suite.

---

## 5. Tests to add

**`koharu-ml`** (`llm/model.rs`)
- `dynamic_context_is_unchanged_without_bounds` — pins the §3 formula
- `minimum_context_raises_a_small_requirement`
- `maximum_context_rejects_an_oversized_requirement`
- `minimum_context_never_lowers_the_requirement`
- `kv_cache_and_flash_attention_reach_context_params`

**`koharu-translator`** (`local/registry.rs`)
- `local_config_round_trips_through_toml`
- `defaults_deserialize_from_an_empty_table` (the migration guarantee)
- `custom_model_cannot_shadow_a_builtin_id`
- `custom_model_requires_an_existing_gguf_file`
- `per_model_profile_overrides_global_runtime`
- `per_run_generation_overrides_every_profile`
- `context_mode_maps_to_generation_options`

**`koharu-app`**
- `capabilities_report_unknown_rather_than_unsupported`

---

## 6. Compatibility statement

- **Existing `~/.koharu/config.toml`**: opens unchanged. Every new key has a default and
  `koharu-config` deep-merges defaults under the file's values. No migration step.
- **Existing `.khrproj` projects**: untouched — Phase 1 writes no project data.
- **Existing behaviour with default settings**: identical. `ContextMode::Dynamic`,
  `GpuLayers::All` (= today's `DEFAULT_GPU_LAYERS = 1000`), `n_batch`/`n_ubatch` unset,
  KV types unset, `FlashAttention::Auto` all reduce to the current code path.
- **In-repo API breaks** (`GenerationOptions` serde derives, `with_flash_attention_policy`
  signature): every consumer is updated in the same change, per upstream's no-compat rule.
- **Upstream merge surface**: 2 small edits in `koharu-llama`, 2 in `koharu-ml`, 4 in
  `koharu-translator`, 2 in `koharu-app`, 2 in the frontend. Everything else is new files.

---

## 7. Explicitly deferred (with reasons)

> **Since resolved:** *FLUX.2 configurable models (§13)* and *FLUX inference
> settings (§14)* shipped in `PHASE-2-NOTES.md`. *Klein 9B* remains deferred for
> the reason below.

| item | why not now |
| --- | --- |
| **FLUX.2 configurable models (§13)** | `TRANSFORMER_WEIGHTS` etc. are compile-time `const HuggingFaceFile` (`flux2_klein/mod.rs:23`). Making them runtime values changes `Flux2Klein{,Inpaint}::load`'s signature and touches the diffusion loader. Worth its own increment. |
| **Klein 9B (§13)** | Requires verifying the pinned stable-diffusion.cpp build actually supports a 9B FLUX.2 transformer and that `FLUX.2-small-decoder`'s latent width matches. §39 forbids shipping a path swap without that check. |
| **FLUX inference settings (§14)** | `Flux2KleinInpaintOptions::default()` at `stages/inpainting.rs:237` is a one-line widening, but the 1 MP resize is hardcoded at **three** sites (`mod.rs:80`, `:179`, `:230`) and needs a single owner first. |
| **VRAM Budget Manager (§11)** | Needs per-component attribution that `resources/vram.rs` cannot currently measure. Requires designing the `measured` vs `estimated` split honestly. Phase 1 ships the *diagnostics readout* (§41) of values we already know; budgets come later. |
| **Model residency policy (§12)** | `Pipeline::from_config` rebuilds the whole `StageRunner` on any config change (`pipeline.rs:44`), dropping every model. A residency policy is meaningless until that is addressed. |
| **Hardware / pipeline profiles (§31–32)** | The layered resolver (§34) should be built once, over the full set of layers, rather than twice. |

---

## 7b. Deltas between this plan and what shipped

| planned | shipped | why |
| --- | --- | --- |
| `GpuLayers { Auto, All, Custom }` | `GpuLayers { All, Custom }` | `Auto` would have been byte-identical to `All` until a VRAM budget exists to fit layers to free memory. Offering two identical options would be a lie. `Auto` returns with the VRAM Budget Manager. |
| `add_custom_model` / `remove_custom_model` commands | none | `LocalConfig` is part of `ProvidersConfig`, which `save_preferences` already round-trips. The UI edits the list directly; `ProviderPreferences::into_config` calls `LocalConfig::validate()` so bad entries are rejected at save time. Two fewer commands to keep in sync. |
| `inspect_gguf` command | `pick_gguf_file` | Reading GGUF metadata to pre-fill a chat template is a Phase 2 concern. What Phase 1 needs is a native picker, matching how `import_pages` already uses `rfd`. |
| `LlmCapabilities { gpu_offload, flash_attention, kv_cache_quantization }` as tri-state `Support` | `{ device, backend, gpu_offload: bool, total_memory, deferred: [...] }` | Only GPU offload is genuinely queryable before a context exists. Flash attention and KV quantization are reported as *deferred* with the reason, and their controls stay enabled — per §39, an honest "llama.cpp decides at context creation" beats a fabricated Supported/Unsupported. |
| `LocalConfig` fields non-optional with defaults | every field `Option` | Layering needs to distinguish "unset" from "set to the default value", otherwise a per-model profile could not leave a global setting alone. |

Also added beyond the plan: `LoadSignature`, which limits model reloads to settings that actually affect loading (`gpu_layers`, and the registered file itself). Context, batch, thread, KV and flash-attention changes take effect on the next inference with no reload, because Koharu creates the llama.cpp context per call.

## 8. Open question for the user

**Max Output Tokens default.** The brief's §4 UI shows `4096`. Today the effective
default is `1000` — `ModelGeneration::options` hardcodes `.map_or(1000, …)`
(`model.rs:114`) and every catalog entry sets `max_tokens: Some(1000)`.

Raising the default to 4096 raises `required` in the context formula by ~3k positions
for **every** call, which directly increases KV-cache VRAM on every translation — a
meaningful cost on a 12 GB card, for output that batched bubble translation rarely needs.

Planned unless told otherwise: **keep 1000 as the default**, expose the field in the
Advanced panel with 4096 as a documented suggestion for long-form/novel pages.
