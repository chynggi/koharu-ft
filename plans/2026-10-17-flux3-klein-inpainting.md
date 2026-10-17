# FLUX 3 Image Klein 인페인팅 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FLUX 3 Image Klein을 공식 프롬프트 업샘플러(PUP)와 함께 `flux3-klein` 인페인팅 모델로 추가한다. 페이지 단위 2단계(PUP 캡션 → FLUX 3 인페인팅) 스왑으로 12GB GPU에서 동작하게 한다.

**Architecture:**
- `koharu-ml::flux3_klein`이 두 모델을 모두 소유한다.
  - FLUX 3는 sd.cpp 컨텍스트이고, 파라미터를 항상 RAM에 둔다(`*=cpu`).
  - PUP는 `Llm` + mtmd다. `caption` 호출 한 번 안에서 로드, 캡션 생성, drop까지 끝낸다.
- 파이프라인은 기존 `inpaint_tiled`를 `TilePlan`(계획/크롭/실행)으로 나눈 뒤 `Model::Flux3` 분기에서 두 단계를 차례로 실행한다.
- 측정 게이트(G1–G4)를 통과하기 전에는 파이프라인·UI·문서 작업에 들어가지 않는다.

**Tech Stack:** Rust (`koharu-ml`, `koharu-pipeline`, `koharu-app`, `koharu-diffusion`, `koharu-runtime`), stable-diffusion.cpp `master-968-8cd731e`, llama.cpp `b11552` + mtmd, TypeScript/React (Next.js, vitest), i18next.

**Spec:** `specs/2026-10-17-flux3-klein-inpainting-design.md`

## Global Constraints

- **하위 호환 레이어를 두지 않는다**(`AGENTS.md`). `inpaint_tiled`는 `TilePlan`으로 대체하고 지운다.
- **생성물은 손으로 고치지 않는다.** `crates/koharu-diffusion-sys/include/stable-diffusion.h`는 upstream 태그에서 그대로 받는다. 바인딩은 `build.rs`가 빌드할 때 만든다.
- `packages/bridge/src/protocol.ts`는 **손으로 쓰는** RPC 클라이언트다. 타입은 직접 추가한다.
- 검증은 **debug 프로필**로 관련 테스트만 한 번 실행한다. 전체 테스트 스위트, release 빌드, E2E는 돌리지 않는다.
- 업샘플러를 끄는 설정과 업샘플러 실패 시 폴백은 **만들지 않는다**(스펙 D6). `--no-upsampler`는 디버그 바이너리에만 둔다.
- 내장 체크포인트는 40자 커밋으로 고정한다. 값은 아래 표가 유일한 출처다.

| 상수 | 저장소 @ 커밋 | 파일 |
|---|---|---|
| `TRANSFORMER_WEIGHTS` | `DogukanUrker/FLUX.3-Image-Klein-GGUF` @ `652e3dae6f7f26777c8749c99dcb3bb643a8350c` | `flux3-image-klein-Q4_K.gguf` |
| `TEXT_ENCODER_WEIGHTS` | `Qwen/Qwen3-VL-4B-Instruct-GGUF` @ `1cd86afb9a95c410a6038ab3b40d8b578c892266` | `Qwen3VL-4B-Instruct-Q4_K_M.gguf` |
| `VAE_WEIGHTS` | `Comfy-Org/flux2-klein` @ `5f526678002e43af5551dadb73ce2e8c91b43afe` | `split_files/vae/flux2-vae.safetensors` |
| `UPSAMPLER_WEIGHTS`, `UPSAMPLER_PROJECTOR` | `prithivMLmods/flux-3-image-klein-prompt-upsampler-GGUF` @ `71713afaf9bce90074c078254d07f5b348f9c372` | `flux-3-image-klein-pup.Q4_K_M.gguf`, `flux-3-image-klein-pup.mmproj-q8_0.gguf` |
| `SYSTEM_PROMPTS` | `black-forest-labs/flux-3-image-klein-pup` @ `3f3edb52b2c55de0d49feab28598b68f7f8e4830` | `system_prompts.json` |

## 파일 구조

| 파일 | 상태 | 책임 |
|---|---|---|
| `crates/koharu-runtime/src/runtime/packages/diffusion.rs` | 수정 | sd.cpp 런타임 릴리스 고정 |
| `crates/koharu-diffusion-sys/include/stable-diffusion.h` | 교체 | upstream 헤더(생성물 입력) |
| `crates/koharu-ml/src/flux2_klein/mod.rs`, `processor.rs` | 수정 | `Flux2ImageProcessor`의 가시성만 `pub(crate)`로 넓힘 |
| `crates/koharu-ml/src/flux3_klein/mod.rs` | 신규 | 고정 저장소, 경로 해석, 옵션, `Flux3KleinInpaint` |
| `crates/koharu-ml/src/flux3_klein/source.rs` | 신규 | `Flux3KleinSource`(컴포넌트 5개) |
| `crates/koharu-ml/src/flux3_klein/model.rs` | 신규 | sd.cpp 컨텍스트 조립 |
| `crates/koharu-ml/src/flux3_klein/upsampler.rs` | 신규 | `Flux3PromptUpsampler`(PUP 메시지 조립, 캡션) |
| `crates/koharu-ml/src/bin/flux3_klein.rs` | 신규 | 게이트 G2–G4 측정 도구 |
| `crates/koharu-ml/src/lib.rs` | 수정 | `pub mod flux3_klein;` |
| `crates/koharu-pipeline/src/stages/inpainting.rs` | 수정 | `TilePlan`, `Flux3KleinConfig`, `Model::Flux3` |
| `crates/koharu-pipeline/src/config.rs`, `stages/mod.rs`, `lib.rs`, `bin/run.rs` | 수정 | `flux3-klein` 등록·재노출 |
| `crates/koharu-app/src/commands/preferences.rs` | 수정 | 프로필 기억 |
| `packages/bridge/src/protocol.ts` | 수정 | TS 타입 |
| `packages/koharu/components/preferences/Flux3KleinOptions.tsx` | 신규 | 옵션 패널과 고지 |
| `packages/koharu/components/preferences/models.ts`, `PipelinePreferences.tsx` | 수정 | 모델 목록·패널 연결 |
| `packages/koharu/public/locales/*/translation.json` (9개) | 수정 | i18n |
| `packages/koharu/tests/components/flux3-klein-options.test.tsx` | 신규 | UI 테스트 |
| `packages/docs/{en,ja,zh}/...` | 수정 | 문서(G4 이후) |
| `specs/2026-10-17-flux3-klein-inpainting-design.md` | 수정 | 게이트 측정 기록 |

---

## Phase A — 런타임과 리팩터 (게이트 이전)

### Task 1: sd.cpp 런타임을 `master-968-8cd731e`로 올리기 (게이트 G1)

**Files:**
- Modify: `crates/koharu-runtime/src/runtime/packages/diffusion.rs:15`
- Replace: `crates/koharu-diffusion-sys/include/stable-diffusion.h`

**준비물**: FLUX.2 회귀 비교용 입력이 필요하다. 대표 만화 페이지에서 말풍선 영역을 768×768 이하로 자른 `crop.png`와, 그 크롭의 글자 마스크 `crop-mask.png`(흰색=지울 영역)다. 경로는 환경 변수 `KOHARU_SAMPLE_CROP`, `KOHARU_SAMPLE_MASK`로 넘긴다. 결과물은 스크래치 디렉터리 `$KOHARU_GATE_DIR`에 쓰고 커밋하지 않는다.

- [ ] **Step 1: 업그레이드 전 FLUX.2 기준선을 만든다**

Run:
```bash
cargo run -p koharu-ml --bin flux2_klein -- \
  --input "$KOHARU_SAMPLE_CROP" --mask "$KOHARU_SAMPLE_MASK" \
  --output "$KOHARU_GATE_DIR/flux2-before.png" \
  --prompt "Remove the text and reconstruct the background." --seed 42
```
Expected: 종료 코드 0, `flux2-before.png` 생성.

- [ ] **Step 2: 릴리스 고정값을 바꾼다**

`crates/koharu-runtime/src/runtime/packages/diffusion.rs:15`:
```rust
const RELEASE: &str = "master-968-8cd731e";
```

- [ ] **Step 3: 같은 태그의 헤더를 받는다**

`.agents/skills/runtime/scripts/sync.sh`는 llama 헤더까지 최신으로 바꾸므로 쓰지 않는다. 그 스크립트의 diffusion 줄만 고정 태그로 실행한다.

Run:
```bash
curl -fsSL "https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master-968-8cd731e/include/stable-diffusion.h" \
  -o crates/koharu-diffusion-sys/include/stable-diffusion.h
git diff --stat crates/koharu-diffusion-sys/include/stable-diffusion.h
```
Expected: 헤더 diff가 나온다(FLUX.3 관련 enum이나 필드가 추가됨).

- [ ] **Step 4: 런타임 크레이트를 검사한다**

Run:
```bash
cargo check -p koharu-diffusion-sys -p koharu-diffusion -p koharu-ml -p koharu-pipeline
```
Expected: 성공. 실패하면 헤더 변경으로 안전 API가 깨진 것이다.
- `crates/koharu-diffusion/src/{params.rs,enums.rs,context.rs}`에서 컴파일러가 가리키는 필드와 enum을 새 헤더에 맞춘다.
- 새 enum 값은 `enums.rs`의 기존 매크로 표에 한 줄씩 추가하고, 이름은 헤더의 `sd_*_name` 문자열을 그대로 쓴다.
- 인-리포 소비자를 같은 커밋에서 갱신하고, 호환 별칭은 두지 않는다.

- [ ] **Step 5: 업그레이드 후 FLUX.2를 다시 실행한다**

Run:
```bash
cargo run -p koharu-ml --bin flux2_klein -- \
  --input "$KOHARU_SAMPLE_CROP" --mask "$KOHARU_SAMPLE_MASK" \
  --output "$KOHARU_GATE_DIR/flux2-after.png" \
  --prompt "Remove the text and reconstruct the background." --seed 42
```
Expected: 종료 코드 0. 첫 실행 때 새 런타임 패키지를 내려받는다.

- [ ] **Step 6: 전후 결과를 비교한다**

Run:
```bash
python -I -c "
import math, sys
from PIL import Image, ImageChops
a = Image.open(sys.argv[1]).convert('RGB'); b = Image.open(sys.argv[2]).convert('RGB')
assert a.size == b.size, (a.size, b.size)
d = ImageChops.difference(a, b)
hist = d.histogram(); n = a.size[0] * a.size[1] * 3
mse = sum(v * (i % 256) ** 2 for i, v in enumerate(hist)) / n
print('max', max(c[1] for c in d.getextrema()), 'psnr', 'inf' if mse == 0 else round(10 * math.log10(255 ** 2 / mse), 2))
" "$KOHARU_GATE_DIR/flux2-before.png" "$KOHARU_GATE_DIR/flux2-after.png"
```
Expected: `psnr`가 `inf`이거나 40dB 이상이다. 그보다 낮으면 멈추고, 두 이미지를 사용자에게 보여 판정을 받는다. 결과 수치는 Task 8에서 스펙에 기록하도록 메모해 둔다.

- [ ] **Step 7: 커밋한다**

```bash
git add crates/koharu-runtime/src/runtime/packages/diffusion.rs crates/koharu-diffusion-sys/include/stable-diffusion.h crates/koharu-diffusion
git commit -m "build(runtime): pin stable-diffusion.cpp master-968-8cd731e for FLUX.3

The release carries leejet/stable-diffusion.cpp#2143 (FLUX.3 Image Klein).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `inpaint_tiled`를 `TilePlan`으로 분리

**Files:**
- Modify: `crates/koharu-pipeline/src/stages/inpainting.rs:960-1040` (`inpaint_tiled`), 그리고 호출부 7곳(`Model::run`의 각 분기)
- Test: 같은 파일의 `mod tests`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`mod tests`의 `textured_mask_regions_are_inferred_as_bounded_tiles` 바로 뒤에 추가한다:
```rust
    #[test]
    fn tile_crops_match_the_tiles_run_visits_in_order() {
        let mut image = RgbImage::new(1200, 700);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if (x + y).is_multiple_of(2) {
                Rgb([0, 40, 80])
            } else {
                Rgb([255, 215, 175])
            };
        }
        let mut mask = GrayImage::new(1200, 700);
        mask.put_pixel(10, 10, Luma([u8::MAX]));
        mask.put_pixel(1100, 600, Luma([u8::MAX]));

        let plan = TilePlan::new(&DynamicImage::ImageRgb8(image), &mask, &mask, &[]).unwrap();
        let crops = plan.crops();
        let mut visited = Vec::new();
        plan.run(|index, tile, _| {
            assert_eq!(tile.dimensions(), crops[index].dimensions());
            visited.push(index);
            Ok(tile.clone())
        })
        .unwrap();

        assert_eq!(crops.len(), 2);
        assert_eq!(visited, vec![0, 1]);
    }
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -p koharu-pipeline --lib stages::inpainting::tests::tile_crops_match_the_tiles_run_visits_in_order`
Expected: 컴파일 실패("cannot find type `TilePlan`").

- [ ] **Step 3: `inpaint_tiled`를 `TilePlan`으로 바꾼다**

BallonsTranslator 주석(960–965행)은 그대로 두고, 그 아래의 `fn inpaint_tiled(...) { ... }` 전체를 다음으로 교체한다:
```rust
struct TilePlan {
    output: RgbImage,
    pending_mask: GrayImage,
    labels: ImageBuffer<Luma<u32>, Vec<u32>>,
    tiles: Vec<InpaintTile>,
}

impl TilePlan {
    fn new(
        image: &DynamicImage,
        mask: &GrayImage,
        text_mask: &GrayImage,
        flat_fill_regions: &[FlatFillRegion],
    ) -> Result<Self> {
        ensure!(
            image.dimensions() == mask.dimensions() && mask.dimensions() == text_mask.dimensions(),
            "image and mask dimensions differ: image={:?}, mask={:?}, text_mask={:?}",
            image.dimensions(),
            mask.dimensions(),
            text_mask.dimensions()
        );
        let mut output = image.to_rgb8();
        let mut pending_mask = mask.clone();
        fill_uniform_regions(&mut output, &mut pending_mask, text_mask, flat_fill_regions);
        let (labels, tiles) = inpaint_tiles(&pending_mask);
        Ok(Self {
            output,
            pending_mask,
            labels,
            tiles,
        })
    }

    /// Every tile's image before any tile is inpainted, in the order `run`
    /// visits them. Overlapping tiles are cut from the same untouched output
    /// here, while `run` cuts each one from the output its predecessors left.
    fn crops(&self) -> Vec<DynamicImage> {
        self.tiles
            .iter()
            .map(|tile| DynamicImage::ImageRgb8(crop_tile_image(&self.output, tile)))
            .collect()
    }

    fn run(
        self,
        mut inference: impl FnMut(usize, &DynamicImage, &GrayImage) -> Result<DynamicImage>,
    ) -> Result<DynamicImage> {
        let Self {
            mut output,
            pending_mask,
            labels,
            tiles,
        } = self;
        for (index, tile) in tiles.iter().enumerate() {
            let [left, top, right, bottom] = tile.crop;
            let crop_width = right - left;
            let crop_height = bottom - top;
            let crop_image = crop_tile_image(&output, tile);
            let crop_mask = crop_tile_mask(&pending_mask, tile);

            let generated = inference(index, &DynamicImage::ImageRgb8(crop_image), &crop_mask)?;
            let generated = if generated.dimensions() == (crop_width, crop_height) {
                generated.to_rgb8()
            } else {
                generated
                    .resize_exact(
                        crop_width,
                        crop_height,
                        image::imageops::FilterType::Lanczos3,
                    )
                    .to_rgb8()
            };
            composite_generated(&mut output, &labels, tile, &generated);
        }
        Ok(DynamicImage::ImageRgb8(output))
    }
}

fn crop_tile_image(output: &RgbImage, tile: &InpaintTile) -> RgbImage {
    let [left, top, right, bottom] = tile.crop;
    image::imageops::crop_imm(output, left, top, right - left, bottom - top).to_image()
}
```

- [ ] **Step 4: 호출부 7곳을 옮긴다**

`Model::run`의 각 분기(LaMa, MI-GAN, MangaInpaintor, Aot, Flux, Rorem, PowerPaint)에 있는 다음 패턴을
```rust
                        inpaint_tiled(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                            |image, mask| {
```
아래로 바꾼다:
```rust
                        TilePlan::new(
                            &prepared.image,
                            &prepared.mask,
                            &prepared.text_mask,
                            &prepared.flat_fill_regions,
                        )?
                        .run(|_, image, mask| {
```
그리고 각 클로저 뒤의 닫는 부분 `},\n                        )` 를 `})` 로 맞춘다. 클로저 본문은 바꾸지 않는다.

테스트의 `inpaint_tiled(` 호출 4곳(`uniform_mask_background_is_filled_without_inference`, `uniform_bubbles_are_filled_independently_inside_one_textured_tile`, `connected_mask_crossing_both_page_grid_axes_is_inferred_once`, `textured_mask_regions_are_inferred_as_bounded_tiles`)도 같은 방식으로 옮긴다. 형태는 다음과 같다:
```rust
        let output = TilePlan::new(&DynamicImage::ImageRgb8(image), &mask, &mask, &[])
            .unwrap()
            .run(|_, tile, tile_mask| {
                // 기존 클로저 본문 그대로
            })
            .unwrap()
            .to_rgb8();
```
- 기존 클로저 인자가 `|_, _|`였던 곳은 `|_, _, _|`가 된다.
- `flat_fill_regions` 인자가 있던 테스트는 그 값(`&[rectangle_region(...)]`)을 `TilePlan::new`의 네 번째 인자로 그대로 넘긴다.

- [ ] **Step 5: 관련 테스트를 실행한다**

Run: `cargo test -p koharu-pipeline --lib stages::inpainting::tests`
Expected: 새 테스트를 포함해 모두 PASS. `grep -n "inpaint_tiled" crates/koharu-pipeline/src/stages/inpainting.rs` 출력이 없어야 한다.

- [ ] **Step 6: 커밋한다**

```bash
git add crates/koharu-pipeline/src/stages/inpainting.rs
git commit -m "refactor(pipeline): split tile planning from tile inference

FLUX 3 needs every tile's crop before any tile is inpainted, so the
prompt upsampler can caption a page before the image model runs.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `Flux2ImageProcessor`를 크레이트 안에서 공유

**Files:**
- Modify: `crates/koharu-ml/src/flux2_klein/mod.rs:6-8`
- Modify: `crates/koharu-ml/src/flux2_klein/processor.rs`

- [ ] **Step 1: 모듈과 항목 가시성을 넓힌다**

`crates/koharu-ml/src/flux2_klein/mod.rs`의 `mod processor;`를 다음으로 바꾼다:
```rust
// FLUX 3 Image Klein shares the FLUX.2 VAE and its 16-pixel alignment, so its
// inpainting path reuses this preprocessing unchanged; ownership stays here
// with the diffusers FLUX.2 port it traces.
pub(crate) mod processor;
```
`processor.rs`에서 `pub(super) struct Flux2ImageProcessor;`를 `pub(crate) struct Flux2ImageProcessor;`로 바꾼다. `impl Flux2ImageProcessor` 안의 `pub(super) fn` 9개(`check_max_pixels`, `fit_to_area`, `check_image_input`, `_resize_to_target_area`, `get_crop_region`, `_resize_and_fill`, `_resize_and_crop`, `binarize`, `apply_overlay`)도 모두 `pub(crate) fn`으로 바꾼다.

- [ ] **Step 2: 검사한다**

Run: `cargo test -p koharu-ml --lib flux2_klein::processor`
Expected: `overlay_without_a_crop_preserves_unmasked_pixels` PASS.

- [ ] **Step 3: 커밋한다**

```bash
git add crates/koharu-ml/src/flux2_klein
git commit -m "refactor(ml): share the FLUX.2 image processor inside the crate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Phase B — `koharu-ml::flux3_klein`

### Task 4: 소스, 고정 저장소, 경로 해석

**Files:**
- Create: `crates/koharu-ml/src/flux3_klein/source.rs`
- Create: `crates/koharu-ml/src/flux3_klein/mod.rs` (이 태스크에서는 일부만)
- Modify: `crates/koharu-ml/src/lib.rs:15` 다음 줄

- [ ] **Step 1: 실패하는 테스트와 함께 `source.rs`를 만든다**

```rust
//! Where the FLUX 3 Image Klein components are loaded from.
//!
//! The shipped repositories stay the default. An override only changes which
//! file is handed to stable-diffusion.cpp or llama.cpp; the architecture is
//! still read from the file's own metadata.

use anyhow::{Context as _, Result};

pub use crate::source::ComponentSource;

/// The five checkpoints an inpainting call is assembled from: three for the
/// image model and two for the prompt upsampler that writes its captions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Flux3KleinSource {
    pub transformer: ComponentSource,
    pub text_encoder: ComponentSource,
    pub vae: ComponentSource,
    pub upsampler: ComponentSource,
    pub upsampler_mmproj: ComponentSource,
}

impl Flux3KleinSource {
    pub fn validate(&self) -> Result<()> {
        self.transformer
            .validate()
            .context("FLUX 3 Image Klein transformer")?;
        self.text_encoder
            .validate()
            .context("FLUX 3 Image Klein text encoder")?;
        self.vae.validate().context("FLUX 3 Image Klein VAE")?;
        self.upsampler
            .validate()
            .context("FLUX 3 Image Klein prompt upsampler")?;
        self.upsampler_mmproj
            .validate()
            .context("FLUX 3 Image Klein upsampler projector")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn the_default_source_uses_every_builtin_repository() {
        let source = Flux3KleinSource::default();
        assert_eq!(source.transformer, ComponentSource::Builtin);
        assert_eq!(source.text_encoder, ComponentSource::Builtin);
        assert_eq!(source.vae, ComponentSource::Builtin);
        assert_eq!(source.upsampler, ComponentSource::Builtin);
        assert_eq!(source.upsampler_mmproj, ComponentSource::Builtin);
        source.validate().unwrap();
    }

    #[test]
    fn an_invalid_override_names_its_component() {
        let source = Flux3KleinSource {
            upsampler_mmproj: ComponentSource::LocalFile(PathBuf::from("relative.gguf")),
            ..Flux3KleinSource::default()
        };
        let error = format!("{:#}", source.validate().unwrap_err());
        assert!(error.contains("upsampler projector"), "{error}");
    }
}
```

- [ ] **Step 2: `mod.rs`의 첫 부분을 만든다**

```rust
//! FLUX 3 Image Klein inpainting behind the official prompt upsampler.
//!
//! Component mapping follows stable-diffusion.cpp at:
//! https://github.com/leejet/stable-diffusion.cpp/blob/8cd731e/docs/flux3.md
//! The upsampler message layout follows the model card at:
//! https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/337337621b8526e120f2275d250242f4a6bedd75/README.md

mod source;

pub use self::source::{ComponentSource, Flux3KleinSource};

crate::model_repository!("DogukanUrker/FLUX.3-Image-Klein-GGUF" @ "652e3dae6f7f26777c8749c99dcb3bb643a8350c" {
    TRANSFORMER_WEIGHTS = "flux3-image-klein-Q4_K.gguf"
});
crate::model_repository!("Qwen/Qwen3-VL-4B-Instruct-GGUF" @ "1cd86afb9a95c410a6038ab3b40d8b578c892266" {
    TEXT_ENCODER_WEIGHTS = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
});
// FLUX 3 ships this same FLUX.2 autoencoder as `ae.safetensors` (identical
// SHA-256), but the Black Forest Labs repository is gated; this copy is not.
crate::model_repository!("Comfy-Org/flux2-klein" @ "5f526678002e43af5551dadb73ce2e8c91b43afe" {
    VAE_WEIGHTS = "split_files/vae/flux2-vae.safetensors"
});
crate::model_repository!("prithivMLmods/flux-3-image-klein-prompt-upsampler-GGUF" @ "71713afaf9bce90074c078254d07f5b348f9c372" {
    UPSAMPLER_WEIGHTS = "flux-3-image-klein-pup.Q4_K_M.gguf",
    UPSAMPLER_PROJECTOR = "flux-3-image-klein-pup.mmproj-q8_0.gguf",
});
crate::model_repository!("black-forest-labs/flux-3-image-klein-pup" @ "3f3edb52b2c55de0d49feab28598b68f7f8e4830" {
    SYSTEM_PROMPTS = "system_prompts.json"
});

/// The instruction the prompt upsampler expands into one caption per tile.
pub const DEFAULT_REQUEST: &str =
    "Remove all text and lettering, and reconstruct the background behind it.";
```

`crates/koharu-ml/src/lib.rs`의 `pub mod flux2_klein;` 다음 줄에 `pub mod flux3_klein;`를 추가한다.

- [ ] **Step 3: 테스트를 실행한다**

Run: `cargo test -p koharu-ml --lib flux3_klein::source`
Expected: 테스트 2개 PASS. 상수를 아직 쓰지 않아 `dead_code` 경고가 나는데, Task 5–6에서 사라진다.

- [ ] **Step 4: 커밋한다**

```bash
git add crates/koharu-ml/src/flux3_klein crates/koharu-ml/src/lib.rs
git commit -m "feat(ml): pin the FLUX 3 Image Klein and prompt upsampler checkpoints

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 프롬프트 업샘플러

**Files:**
- Create: `crates/koharu-ml/src/flux3_klein/upsampler.rs`
- Modify: `crates/koharu-ml/src/flux3_klein/mod.rs`

- [ ] **Step 1: 실패하는 테스트와 함께 `upsampler.rs`를 만든다**

```rust
//! The official FLUX 3 Image Klein prompt upsampler.
//!
//! The i2i message layout is `upsample(prompt, images)` in the model card at:
//! https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/337337621b8526e120f2275d250242f4a6bedd75/README.md

use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context as _, Result, bail, ensure};
use image::DynamicImage;

use super::{Flux3KleinSource, SYSTEM_PROMPTS, UPSAMPLER_PROJECTOR, UPSAMPLER_WEIGHTS};
use crate::{
    Device,
    llm::{
        ChatMessage, FinishReason, GenerationOptions, Input, Llm, LoadOptions, MtmdOptions,
        media_marker,
    },
};

/// `max_new_tokens` in the model card's `upsample`.
const MAX_NEW_TOKENS: usize = 3584;

/// Where the upsampler lives, never the loaded model. Each `caption` call loads
/// it, captions every image and drops it before returning, so the accelerator
/// is free for FLUX 3 again; the weights stay in the OS file cache between
/// pages.
#[derive(Clone, Debug)]
pub struct Flux3PromptUpsampler {
    device: Device,
    model: PathBuf,
    projector: PathBuf,
    system_prompt: String,
}

impl Flux3PromptUpsampler {
    /// Resolves (and downloads) the upsampler files and reads the i2i system
    /// prompt. Nothing is loaded onto the accelerator.
    pub async fn load(device: Device, source: &Flux3KleinSource) -> Result<Self> {
        source.validate()?;
        let (model, projector, prompts) = tokio::try_join!(
            source.upsampler.resolve(UPSAMPLER_WEIGHTS.into()),
            source.upsampler_mmproj.resolve(UPSAMPLER_PROJECTOR.into()),
            SYSTEM_PROMPTS.resolve(),
        )
        .context("failed to resolve the FLUX 3 prompt upsampler")?;
        let system_prompt = i2i_system_prompt(
            &std::fs::read_to_string(&prompts).context("failed to read system_prompts.json")?,
        )?;
        Ok(Self {
            device,
            model,
            projector,
            system_prompt,
        })
    }

    /// One caption per image, in order. The model holds the accelerator only
    /// for the duration of this call.
    pub async fn caption(&self, images: Vec<DynamicImage>, request: String) -> Result<Vec<String>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let llm = Llm::load_with_options(
            self.device.clone(),
            self.model.clone(),
            LoadOptions {
                mtmd: Some(MtmdOptions::new(self.projector.clone())),
                ..LoadOptions::default()
            },
        )
        .await
        .context("failed to load the FLUX 3 prompt upsampler")?;
        ensure!(
            llm.capabilities().vision,
            "the FLUX 3 prompt upsampler projector does not support images"
        );
        let system_prompt = self.system_prompt.clone();
        tokio_rayon::spawn(move || {
            images
                .iter()
                .enumerate()
                .map(|(index, image)| {
                    caption_one(&llm, &system_prompt, &request, image)
                        .with_context(|| format!("failed to caption tile {index}"))
                })
                .collect::<Result<Vec<_>>>()
        })
        .await
    }
}

fn caption_one(
    llm: &Llm,
    system_prompt: &str,
    request: &str,
    image: &DynamicImage,
) -> Result<String> {
    let prompt = llm.render_chat_prompt(&messages(system_prompt, request))?;
    let generation = llm.inference(
        &Input::new(&prompt).with_image(image),
        &GenerationOptions {
            max_tokens: MAX_NEW_TOKENS,
            // `do_sample=False` in the model card.
            temperature: 0.0,
            add_special: true,
            ..GenerationOptions::default()
        },
    )?;
    if generation.finish_reason == FinishReason::Length {
        bail!("the caption reached {MAX_NEW_TOKENS} tokens without finishing");
    }
    let caption = generation.text.trim().to_owned();
    ensure!(!caption.is_empty(), "the prompt upsampler returned an empty caption");
    Ok(caption)
}

/// The model card's i2i conversation: the system prompt, then one user turn
/// holding the reference image followed by the request.
fn messages(system_prompt: &str, request: &str) -> [ChatMessage; 2] {
    [
        ChatMessage::system(system_prompt),
        ChatMessage::user(format!("{}{}", media_marker(), user_text(request))),
    ]
}

fn user_text(request: &str) -> String {
    format!(
        "Reference images follow in order (the first image, the second image, ...).\nUser request: {request}\nOutput the caption."
    )
}

fn i2i_system_prompt(json: &str) -> Result<String> {
    let prompts: HashMap<String, String> =
        serde_json::from_str(json).context("system_prompts.json is not a map of strings")?;
    prompts
        .get("i2i")
        .cloned()
        .context("system_prompts.json has no i2i prompt")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_user_turn_reproduces_the_model_card_template() {
        assert_eq!(
            user_text("Erase the lettering."),
            "Reference images follow in order (the first image, the second image, ...).\nUser request: Erase the lettering.\nOutput the caption."
        );
    }

    #[test]
    fn the_image_precedes_the_request_in_the_user_turn() {
        let [system, user] = messages("SYSTEM", "Erase the lettering.");
        assert_eq!(system, ChatMessage::system("SYSTEM"));
        assert_eq!(
            user,
            ChatMessage::user(format!("{}{}", media_marker(), user_text("Erase the lettering.")))
        );
    }

    #[test]
    fn the_i2i_prompt_is_selected_from_the_published_keys() {
        let json = r#"{"t2i":"A","i2i":"B","t2i_bbox":"C","i2i_bbox":"D"}"#;
        assert_eq!(i2i_system_prompt(json).unwrap(), "B");
        assert!(i2i_system_prompt(r#"{"t2i":"A"}"#).is_err());
    }

    /// Requires the llama.cpp runtime and downloads the upsampler (~6 GB).
    #[tokio::test]
    #[ignore = "requires the dynamically loaded llama.cpp runtime and the upsampler checkpoints"]
    async fn the_upsampler_captions_a_crop() {
        crate::init().await.unwrap();
        let upsampler = Flux3PromptUpsampler::load(crate::device(false), &Flux3KleinSource::default())
            .await
            .unwrap();
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            256,
            256,
            image::Rgb([240, 240, 240]),
        ));
        let captions = upsampler
            .caption(vec![image], super::super::DEFAULT_REQUEST.to_owned())
            .await
            .unwrap();
        assert_eq!(captions.len(), 1);
        assert!(!captions[0].is_empty());
    }
}
```

- [ ] **Step 2: `mod.rs`에 연결한다**

`mod source;` 아래에 `mod upsampler;`를 추가하고, `pub use`를 다음으로 바꾼다:
```rust
pub use self::{
    source::{ComponentSource, Flux3KleinSource},
    upsampler::Flux3PromptUpsampler,
};
```

- [ ] **Step 3: 테스트를 실행한다**

Run: `cargo test -p koharu-ml --lib flux3_klein::upsampler`
Expected: 테스트 3개 PASS, 1개 ignored.

- [ ] **Step 4: 커밋한다**

```bash
git add crates/koharu-ml/src/flux3_klein
git commit -m "feat(ml): caption FLUX 3 inpainting tiles with the official upsampler

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: FLUX 3 컨텍스트와 마스크 인페인팅

**Files:**
- Create: `crates/koharu-ml/src/flux3_klein/model.rs`
- Modify: `crates/koharu-ml/src/flux3_klein/mod.rs`

- [ ] **Step 1: `model.rs`를 만든다**

```rust
//! Native FLUX 3 Image Klein model assembly.
//!
//! Component mapping follows stable-diffusion.cpp at:
//! https://github.com/leejet/stable-diffusion.cpp/blob/8cd731e/docs/flux3.md

use std::{path::PathBuf, sync::Mutex};

use anyhow::{Context as _, Result, anyhow, ensure};
use koharu_diffusion::{Context, ContextParams, ImageGenerationParams, RgbImage, VaeFormat};

use crate::Backend;

#[derive(Debug)]
pub(super) struct ModelPaths {
    pub transformer: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
}

#[derive(Debug)]
pub(super) struct Model {
    context: Mutex<Context>,
}

impl Model {
    pub fn new(device: &crate::Device, paths: ModelPaths) -> Result<Self> {
        let context = Context::new(&context_params(device, paths))
            .context("failed to load FLUX 3 Image Klein components")?;
        ensure!(
            context.supports_image_generation(),
            "the loaded FLUX 3 Image Klein context does not support image generation"
        );
        Ok(Self {
            context: Mutex::new(context),
        })
    }

    pub fn forward(&self, params: &ImageGenerationParams) -> Result<Vec<RgbImage>> {
        let mut context = self
            .context
            .lock()
            .map_err(|_| anyhow!("FLUX 3 Image Klein context lock was poisoned"))?;
        context
            .generate_image(params)
            .context("FLUX 3 Image Klein inference failed")
    }
}

fn context_params(device: &crate::Device, paths: ModelPaths) -> ContextParams {
    let use_accelerator = device.backend != Backend::Cpu;
    let use_cuda = device.backend == Backend::Cuda;
    ContextParams {
        diffusion_model_path: Some(paths.transformer),
        llm_path: Some(paths.text_encoder),
        vae_path: Some(paths.vae),
        enable_mmap: true,
        flash_attention: use_cuda,
        diffusion_flash_attention: use_cuda,
        diffusion_conv_direct: use_cuda,
        vae_conv_direct: use_cuda,
        // FLUX 3 Image Klein decodes with the FLUX.2 autoencoder.
        vae_format: VaeFormat::Flux2,
        backend: Some(device.name.clone()),
        // Unlike FLUX.2 Klein, parameters always stay in RAM: the prompt
        // upsampler takes the accelerator between pages, so this context must
        // hand its VRAM back after every call instead of keeping weights there.
        params_backend: use_accelerator.then(|| "*=cpu".to_owned()),
        ..ContextParams::default()
    }
}
```

- [ ] **Step 2: `mod.rs`에 옵션, 해석, `Flux3KleinInpaint`를 추가한다**

`mod source;` 위에 `mod model;`을 추가한다. 파일 맨 위 doc 주석 아래에 다음 `use`를 넣는다:
```rust
use anyhow::{Context as _, Result, ensure};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, RgbImage};
use koharu_diffusion::{
    GuidanceParams, ImageGenerationParams, SampleMethod, SampleParams, Scheduler,
};

use self::model::{Model, ModelPaths};
use crate::flux2_klein::processor::Flux2ImageProcessor;
```
`DEFAULT_REQUEST` 아래에 다음을 추가한다:
```rust
/// The working area every tile is shrunk to before denoising.
pub const DEFAULT_MAX_PIXELS: u32 = 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct Flux3KleinInpaintOptions {
    pub padding_mask_crop: Option<u32>,
    pub strength: f64,
    /// The distilled checkpoint is built for 16 steps.
    pub num_inference_steps: usize,
    pub seed: i64,
    pub max_pixels: u32,
}

impl Default for Flux3KleinInpaintOptions {
    fn default() -> Self {
        Self {
            padding_mask_crop: None,
            strength: 0.8,
            num_inference_steps: 16,
            seed: -1,
            max_pixels: DEFAULT_MAX_PIXELS,
        }
    }
}

async fn resolve(source: &Flux3KleinSource) -> Result<ModelPaths> {
    let (transformer, text_encoder, vae) = tokio::try_join!(
        source.transformer.resolve(TRANSFORMER_WEIGHTS.into()),
        source.text_encoder.resolve(TEXT_ENCODER_WEIGHTS.into()),
        source.vae.resolve(VAE_WEIGHTS.into()),
    )
    .context("failed to resolve FLUX 3 Image Klein model assets")?;
    Ok(ModelPaths {
        transformer,
        text_encoder,
        vae,
    })
}

#[derive(Debug)]
pub struct Flux3KleinInpaint {
    model: Model,
}

impl Flux3KleinInpaint {
    pub async fn load(device: crate::Device, source: &Flux3KleinSource) -> Result<Self> {
        source.validate()?;
        let model = Model::new(&device, resolve(source).await?)?;
        Ok(Self { model })
    }

    /// Inpaints the masked area of `image` from a caption the prompt upsampler
    /// wrote. Mask preparation and the final overlay are the FLUX.2 Klein
    /// inpaint port's; only the sampler settings belong to FLUX 3.
    pub fn inference(
        &self,
        caption: &str,
        image: &DynamicImage,
        mask_image: &DynamicImage,
        options: &Flux3KleinInpaintOptions,
    ) -> Result<DynamicImage> {
        ensure!(
            image.width() == mask_image.width() && image.height() == mask_image.height(),
            "image/mask dimensions differ: image={}x{}, mask={}x{}",
            image.width(),
            image.height(),
            mask_image.width(),
            mask_image.height()
        );
        ensure!(!caption.contains('\0'), "caption contains an interior NUL byte");
        ensure!(
            options.strength > 0.0 && options.strength <= 1.0,
            "FLUX 3 inpaint strength must be greater than zero and at most one"
        );
        ensure!(
            options.num_inference_steps > 0,
            "num_inference_steps must be greater than zero"
        );
        Flux2ImageProcessor::check_max_pixels(options.max_pixels)?;

        let image = Flux2ImageProcessor::fit_to_area(image, options.max_pixels);
        let width = (image.width() / 16) * 16;
        let height = (image.height() / 16) * 16;
        ensure!(width > 0 && height > 0);
        let source = image.to_rgb8();
        let mut image = RgbImage::new(width, height);
        Resizer::new()
            .resize(
                &source,
                &mut image,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                    .use_alpha(false),
            )
            .expect("source and destination images have the same pixel type");
        let source_mask = mask_image.to_luma8();
        let mut mask_image = image::GrayImage::new(width, height);
        Resizer::new()
            .resize(
                &source_mask,
                &mut mask_image,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                    .use_alpha(false),
            )
            .expect("source and destination masks have the same pixel type");

        let crop_coords = options.padding_mask_crop.and_then(|padding| {
            Flux2ImageProcessor::get_crop_region(&mask_image, width, height, padding)
        });
        let (init_image, mut native_mask) = if let Some((x1, y1, x2, y2)) = crop_coords {
            let image_crop = DynamicImage::ImageRgb8(
                image::imageops::crop_imm(&image, x1, y1, x2 - x1, y2 - y1).to_image(),
            );
            let mask_crop = DynamicImage::ImageLuma8(
                image::imageops::crop_imm(&mask_image, x1, y1, x2 - x1, y2 - y1).to_image(),
            );
            (
                Flux2ImageProcessor::_resize_and_fill(&image_crop, width, height).to_rgb8(),
                Flux2ImageProcessor::_resize_and_fill(&mask_crop, width, height).to_luma8(),
            )
        } else {
            (image.clone(), mask_image.clone())
        };
        Flux2ImageProcessor::binarize(&mut native_mask);

        let strength = if options.strength >= 1.0 {
            1.0
        } else {
            let effective_steps = options.num_inference_steps
                - (options.num_inference_steps as f64 * (1.0 - options.strength)).floor() as usize;
            let boundary = effective_steps as f32 / options.num_inference_steps as f32;
            f32::from_bits(boundary.to_bits() - 1)
        };

        let generated = self
            .model
            .forward(&ImageGenerationParams {
                prompt: caption.to_owned(),
                width: i32::try_from(width)?,
                height: i32::try_from(height)?,
                init_image: Some(init_image.clone()),
                reference_images: vec![init_image],
                reference_image_args: Some("resize_before_vae=0".into()),
                mask_image: Some(native_mask),
                sample: SampleParams {
                    guidance: GuidanceParams {
                        text_cfg: 1.0,
                        ..GuidanceParams::default()
                    },
                    // `Auto` with the default flow shift lets stable-diffusion.cpp
                    // apply FLUX.3's own discrete schedule (shift 6.93).
                    scheduler: Scheduler::Auto,
                    sample_method: SampleMethod::Euler,
                    sample_steps: i32::try_from(options.num_inference_steps)?,
                    ..SampleParams::default()
                },
                seed: options.seed,
                batch_count: 1,
                strength,
                ..ImageGenerationParams::default()
            })?
            .into_iter()
            .next()
            .context("FLUX 3 Image Klein returned no inpainted image")?;

        let generated =
            Flux2ImageProcessor::apply_overlay(&mask_image, &image, generated, crop_coords)?;
        Ok(DynamicImage::ImageRgb8(generated))
    }
}
```
`pub use`에 `Flux3KleinInpaint`, `Flux3KleinInpaintOptions`는 이미 같은 모듈에 정의돼 있으므로 따로 노출할 필요가 없다.

- [ ] **Step 3: 기본값 테스트를 추가한다**

`mod.rs` 끝에 다음을 추가한다:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_follow_the_distilled_checkpoint() {
        let options = Flux3KleinInpaintOptions::default();
        assert_eq!(options.num_inference_steps, 16);
        assert_eq!(options.max_pixels, DEFAULT_MAX_PIXELS);
    }
}
```

- [ ] **Step 4: 테스트를 실행한다**

Run: `cargo test -p koharu-ml --lib flux3_klein`
Expected: 이 모듈의 테스트 6개 PASS, 1개 ignored. `dead_code` 경고 없음.

- [ ] **Step 5: 커밋한다**

```bash
git add crates/koharu-ml/src/flux3_klein
git commit -m "feat(ml): inpaint with FLUX 3 Image Klein through stable-diffusion.cpp

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: 측정용 디버그 바이너리

**Files:**
- Create: `crates/koharu-ml/src/bin/flux3_klein.rs`

- [ ] **Step 1: 바이너리를 만든다**

```rust
use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use clap::Parser;
use koharu_ml::flux3_klein::{
    ComponentSource, DEFAULT_MAX_PIXELS, DEFAULT_REQUEST, Flux3KleinInpaint,
    Flux3KleinInpaintOptions, Flux3KleinSource, Flux3PromptUpsampler,
};

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    mask: PathBuf,

    #[arg(short, long, value_name = "FILE", required_unless_present = "caption_only")]
    output: Option<PathBuf>,

    #[arg(long, value_name = "TEXT", default_value = DEFAULT_REQUEST)]
    request: String,

    /// Prints the upsampler caption and exits (gate G2).
    #[arg(long, conflicts_with = "no_upsampler")]
    caption_only: bool,

    /// Sends the request as the caption. Comparison only; the pipeline always
    /// runs the upsampler.
    #[arg(long)]
    no_upsampler: bool,

    #[arg(long, default_value_t = 16)]
    steps: usize,

    #[arg(long, default_value_t = 0.8)]
    strength: f64,

    #[arg(long, default_value_t = -1)]
    seed: i64,

    #[arg(long)]
    padding_mask_crop: Option<u32>,

    #[arg(long, default_value_t = DEFAULT_MAX_PIXELS)]
    max_pixels: u32,

    #[arg(long, value_name = "FILE")]
    transformer: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    text_encoder: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    vae: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    upsampler: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    upsampler_mmproj: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    cpu: bool,
}

fn component(path: Option<&PathBuf>) -> Result<ComponentSource> {
    match path {
        None => Ok(ComponentSource::Builtin),
        Some(path) => Ok(ComponentSource::LocalFile(std::path::absolute(path)?)),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let input = image::open(&cli.input)?;
    let mask = image::open(&cli.mask)?;

    koharu_ml::init().await?;

    let source = Flux3KleinSource {
        transformer: component(cli.transformer.as_ref())?,
        text_encoder: component(cli.text_encoder.as_ref())?,
        vae: component(cli.vae.as_ref())?,
        upsampler: component(cli.upsampler.as_ref())?,
        upsampler_mmproj: component(cli.upsampler_mmproj.as_ref())?,
    };
    source.validate()?;
    let device = koharu_ml::device(cli.cpu);

    let caption = if cli.no_upsampler {
        cli.request.clone()
    } else {
        let upsampler = Flux3PromptUpsampler::load(device.clone(), &source).await?;
        let started = Instant::now();
        let caption = upsampler
            .caption(vec![input.clone()], cli.request.clone())
            .await?
            .into_iter()
            .next()
            .context("the upsampler returned no caption")?;
        eprintln!(
            "phase=caption seconds={:.2} (load + generate + release)",
            started.elapsed().as_secs_f64()
        );
        caption
    };
    println!("{caption}");
    if cli.caption_only {
        return Ok(());
    }

    let started = Instant::now();
    let pipeline = Flux3KleinInpaint::load(device, &source).await?;
    eprintln!("phase=load seconds={:.2}", started.elapsed().as_secs_f64());
    let options = Flux3KleinInpaintOptions {
        num_inference_steps: cli.steps,
        strength: cli.strength,
        seed: cli.seed,
        padding_mask_crop: cli.padding_mask_crop,
        max_pixels: cli.max_pixels,
    };
    // The first call warms the backend; only the second one is measured.
    pipeline.inference(&caption, &input, &mask, &options)?;
    let started = Instant::now();
    let output = pipeline.inference(&caption, &input, &mask, &options)?;
    eprintln!("phase=inpaint seconds={:.2}", started.elapsed().as_secs_f64());
    output.save(cli.output.context("--output is required")?)?;
    Ok(())
}
```

- [ ] **Step 2: 빌드를 확인한다**

Run: `cargo build -p koharu-ml --bin flux3_klein`
Expected: 성공.

- [ ] **Step 3: 커밋한다**

```bash
git add crates/koharu-ml/src/bin/flux3_klein.rs
git commit -m "feat(ml): add a FLUX 3 Image Klein measurement binary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Phase C — 측정 게이트 (사람의 판정이 필요한 체크포인트)

### Task 8: 게이트 G2–G4 실측과 기록

> **이 태스크는 대상 12GB GPU 머신에서 실행하고, 끝나면 반드시 멈춰서 사용자의 판정을 받는다.** 실패한 게이트가 하나라도 있으면 Phase D 이후로 진행하지 않는다.

**Files:**
- Modify: `specs/2026-10-17-flux3-klein-inpainting-design.md` (끝에 `## 게이트 측정 결과` 절 추가)

준비물은 Task 1과 같은 `$KOHARU_SAMPLE_CROP`, `$KOHARU_SAMPLE_MASK`, `$KOHARU_GATE_DIR`에 더해, 서로 다른 페이지에서 고른 크롭·마스크 쌍 3–5개다(각 768×768 이하).

- [ ] **Step 1: G2 — 업샘플러 로드와 캡션**

Run:
```bash
cargo run -p koharu-ml --bin flux3_klein -- \
  --input "$KOHARU_SAMPLE_CROP" --mask "$KOHARU_SAMPLE_MASK" --caption-only
```
Expected: stdout에 여러 문장으로 된 영어 장면 캡션이, stderr에 `phase=caption seconds=…`가 나온다.

실패 판정은 다음과 같다.
- `unknown model architecture: 'qwen3_5'` 같은 로드 오류가 나면 G2 실패다. llama 런타임(`crates/koharu-runtime/src/runtime/packages/llama.rs`의 `RELEASE`) 최신화를 별도 작업으로 분리하자고 사용자에게 보고하고, **여기서 멈춘다.**
- 캡션이 비거나 토큰 한도 오류가 나도 G2 실패다.

- [ ] **Step 2: G3 — 마스크 인페인팅 품질**

크롭 쌍마다(`$i` = 1..N) 세 가지를 실행한다:
```bash
cargo run -p koharu-ml --bin flux3_klein -- --input crop$i.png --mask mask$i.png \
  --output "$KOHARU_GATE_DIR/flux3-$i.png" --seed 42
cargo run -p koharu-ml --bin flux3_klein -- --input crop$i.png --mask mask$i.png \
  --output "$KOHARU_GATE_DIR/flux3-noup-$i.png" --seed 42 --no-upsampler
cargo run -p koharu-ml --bin flux2_klein -- --input crop$i.png --mask mask$i.png \
  --output "$KOHARU_GATE_DIR/flux2-$i.png" --prompt "Remove the text and reconstruct the background." --seed 42
```
마스크 바깥이 원본과 같은지 확인한다:
```bash
python -I -c "
import sys
from PIL import Image
src = Image.open(sys.argv[1]).convert('RGB'); out = Image.open(sys.argv[2]).convert('RGB'); m = Image.open(sys.argv[3]).convert('L')
assert src.size == out.size == m.size
bad = sum(1 for s, o, k in zip(src.getdata(), out.getdata(), m.getdata()) if k == 0 and max(abs(a - b) for a, b in zip(s, o)) > 1)
print('changed_outside_mask', bad)
" crop$i.png "$KOHARU_GATE_DIR/flux3-$i.png" mask$i.png
```
Expected: `changed_outside_mask 0`. 0이 아니면 G3 실패다.

품질 판정: 세 결과(`flux3-$i`, `flux3-noup-$i`, `flux2-$i`)를 크롭별로 나란히 사용자에게 보여주고, "FLUX 3가 FLUX.2보다 나쁘지 않다"는 판정을 받는다. 사용자가 나쁘다고 판정하면 G3 실패다.

- [ ] **Step 3: G4 — 12GB VRAM 피크와 단계별 시간**

VRAM 샘플러를 백그라운드로 켜고 크롭 1개로 전체 실행을 한 다음, 최댓값을 구한다:
```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$KOHARU_GATE_DIR/vram.log" &
SAMPLER=$!
cargo run -p koharu-ml --bin flux3_klein -- --input crop1.png --mask mask1.png \
  --output "$KOHARU_GATE_DIR/flux3-g4.png" --seed 42 2> "$KOHARU_GATE_DIR/phases.log"
kill $SAMPLER
sort -n "$KOHARU_GATE_DIR/vram.log" | tail -1
cat "$KOHARU_GATE_DIR/phases.log" | grep phase=
```
Expected: VRAM 최댓값(MiB)이 12288 이하이고, `phase=caption`, `phase=load`, `phase=inpaint` 시간이 출력된다. 최댓값이 12288을 넘으면 G4 실패다.
- 바이너리는 Phase 1(PUP)과 Phase 2(FLUX 3)를 순서대로 실행하므로, 이 최댓값이 두 단계 피크 중 큰 값이다.
- 측정 전 다른 GPU 프로세스를 끄고, 그 상태의 기준 사용량(`nvidia-smi` 1회)도 함께 기록한다.

같은 실행 중 작업 관리자나 `Get-Counter '\Memory\Available MBytes'`로 시스템 RAM 감소량을 관찰해 기록한다. 최초 다운로드 크기는 측정하지 않는다. 계획 작성 시 Hub API로 고정 파일 6개의 크기를 합한 **12.36GB**(transformer 3.27, text encoder 2.50, VAE 0.34, upsampler 5.63, mmproj 0.62, system prompts 0.00)로 확정돼 있다.

- [ ] **Step 4: 결과를 스펙에 기록한다**

`specs/2026-10-17-flux3-klein-inpainting-design.md` 끝에 다음 형식의 절을 추가하고, 측정한 값을 그대로 채운다:
```markdown
## 게이트 측정 결과 (YYYY-MM-DD)

장치: <GPU 이름, VRAM>, 드라이버 <버전>, OS <버전>, 입력 <크롭 크기들>

| 게이트 | 결과 | 수치 |
|---|---|---|
| G1 | 통과/실패 | FLUX.2 전후 max <값>, PSNR <값> dB |
| G2 | 통과/실패 | 캡션 단계 <초> (로드+생성+해제) |
| G3 | 통과/실패 | 마스크 바깥 변경 픽셀 <값>, 사용자 판정 <요지> |
| G4 | 통과/실패 | VRAM 피크 <MiB> (기준 <MiB>), RAM 추가 <GB>, 다운로드 12.36GB(Hub API), load <초>, inpaint <초>/타일 |
```

- [ ] **Step 5: 커밋하고 멈춘다**

```bash
git add specs/2026-10-17-flux3-klein-inpainting-design.md
git commit -m "docs(specs): record the FLUX 3 Image Klein gate measurements

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
결과 표를 사용자에게 보여주고 Phase D로 진행해도 되는지 확인을 받는다. G2–G4 중 하나라도 실패했다면 여기서 계획을 중단한다.

---

## Phase D — 파이프라인 (G4 통과 후)

### Task 9: `flux3-klein` 설정과 페이지 단위 2단계 흐름

**Files:**
- Modify: `crates/koharu-pipeline/src/stages/inpainting.rs`
- Modify: `crates/koharu-pipeline/src/config.rs`
- Modify: `crates/koharu-pipeline/src/stages/mod.rs:13-16`, `crates/koharu-pipeline/src/lib.rs:34-38`
- Modify: `crates/koharu-pipeline/src/bin/run.rs`
- Modify: `crates/koharu-app/src/commands/preferences.rs:212-224`

- [ ] **Step 1: 실패하는 설정 테스트를 쓴다**

`inpainting.rs`의 `mod tests`에서 `invalid_flux_settings_are_rejected` 다음에 추가한다:
```rust
    #[test]
    fn flux3_defaults_reproduce_the_model_options() {
        let config = Flux3KleinConfig::default();
        assert_eq!(config.request, koharu_ml::flux3_klein::DEFAULT_REQUEST);
        assert_eq!(config.options(), Flux3KleinInpaintOptions::default());
        assert_eq!(
            Flux3KleinSource::from(config.source.clone()),
            Flux3KleinSource::default()
        );
        config.validate().unwrap();
    }

    #[test]
    fn invalid_flux3_settings_are_rejected() {
        for config in [
            Flux3KleinConfig {
                request: String::new(),
                ..Default::default()
            },
            Flux3KleinConfig {
                request: "Erase\0".to_owned(),
                ..Default::default()
            },
            Flux3KleinConfig {
                steps: 0,
                ..Default::default()
            },
            Flux3KleinConfig {
                strength: 1.5,
                ..Default::default()
            },
            Flux3KleinConfig {
                source: Flux3KleinSourceConfig {
                    upsampler: ComponentSourceConfig::LocalFile {
                        path: PathBuf::from("relative.gguf"),
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        ] {
            assert!(config.validate().is_err(), "{config:?} should be rejected");
        }
    }

    #[test]
    fn a_flux3_section_round_trips_through_toml() {
        let config = Flux3KleinConfig {
            request: "Erase the lettering.".to_owned(),
            steps: 12,
            ..Default::default()
        };
        let document = toml::to_string(&config).unwrap();
        assert_eq!(toml::from_str::<Flux3KleinConfig>(&document).unwrap(), config);
        let partial: Flux3KleinConfig = toml::from_str("request = \"Erase.\"").unwrap();
        assert_eq!(partial.options(), Flux3KleinInpaintOptions::default());
    }
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -p koharu-pipeline --lib stages::inpainting::tests::flux3`
Expected: 컴파일 실패("cannot find type `Flux3KleinConfig`").

- [ ] **Step 3: 설정 타입을 추가한다**

`inpainting.rs`의 `use koharu_ml::{...}` 블록에 다음 한 줄을 추가한다:
```rust
    flux3_klein::{
        DEFAULT_REQUEST as FLUX3_DEFAULT_REQUEST, Flux3KleinInpaint, Flux3KleinInpaintOptions,
        Flux3KleinSource, Flux3PromptUpsampler,
    },
```
`impl Flux2KleinConfig { ... }` 블록(231행 근처) 바로 뒤에 다음을 추가한다:
```rust
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct Flux3KleinSourceConfig {
    pub transformer: ComponentSourceConfig,
    pub text_encoder: ComponentSourceConfig,
    pub vae: ComponentSourceConfig,
    pub upsampler: ComponentSourceConfig,
    pub upsampler_mmproj: ComponentSourceConfig,
}

impl From<Flux3KleinSourceConfig> for Flux3KleinSource {
    fn from(value: Flux3KleinSourceConfig) -> Self {
        Self {
            transformer: value.transformer.into(),
            text_encoder: value.text_encoder.into(),
            vae: value.vae.into(),
            upsampler: value.upsampler.into(),
            upsampler_mmproj: value.upsampler_mmproj.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct Flux3KleinConfig {
    /// The instruction the prompt upsampler expands into one caption per tile.
    pub request: String,
    /// Which checkpoints the image model and the upsampler are loaded from.
    pub source: Flux3KleinSourceConfig,
    pub steps: u32,
    pub strength: f64,
    /// `-1` draws a fresh seed for every call.
    #[specta(type = f64)]
    pub seed: i64,
    pub padding_mask_crop: Option<u32>,
    /// The working area every tile is shrunk to before denoising.
    pub max_pixels: u32,
}

impl Default for Flux3KleinConfig {
    fn default() -> Self {
        let defaults = Flux3KleinInpaintOptions::default();
        Self {
            request: FLUX3_DEFAULT_REQUEST.to_owned(),
            source: Flux3KleinSourceConfig::default(),
            steps: u32::try_from(defaults.num_inference_steps)
                .expect("the default step count fits in u32"),
            strength: defaults.strength,
            seed: defaults.seed,
            padding_mask_crop: defaults.padding_mask_crop,
            max_pixels: defaults.max_pixels,
        }
    }
}

impl Flux3KleinConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.request.trim().is_empty(),
            "FLUX 3 upsampler request is empty"
        );
        ensure!(
            !self.request.contains('\0'),
            "FLUX 3 upsampler request contains NUL"
        );
        ensure!(self.steps > 0, "FLUX 3 steps must be greater than zero");
        ensure!(
            self.strength > 0.0 && self.strength <= 1.0,
            "FLUX 3 strength must be greater than zero and at most one"
        );
        ensure!(
            self.max_pixels >= 64 * 64,
            "FLUX 3 max pixels must be at least {} (64x64)",
            64 * 64
        );
        Flux3KleinSource::from(self.source.clone()).validate()
    }

    fn options(&self) -> Flux3KleinInpaintOptions {
        Flux3KleinInpaintOptions {
            padding_mask_crop: self.padding_mask_crop,
            strength: self.strength,
            num_inference_steps: self.steps as usize,
            seed: self.seed,
            max_pixels: self.max_pixels,
        }
    }
}
```

- [ ] **Step 4: `InpaintingModel`에 등록한다 (`config.rs`)**

`config.rs` 6–9행의 `use crate::stages::{...}`에 `Flux3KleinConfig`를 추가한다. 그리고 다음 다섯 곳에서 `Flux2Klein` 줄 바로 뒤에 `Flux3Klein` 줄을 넣는다.
1. `Serialize`의 `let inpainting = match` (77행): `InpaintingModel::Flux3Klein(_) => "flux3-klein",`
2. `Serialize`의 프로필 `match` (103–105행) 뒤:
   ```rust
               InpaintingModel::Flux3Klein(config) => {
                   processor.flux3_klein.get_or_insert_with(|| config.clone());
               }
   ```
3. `Deserialize`의 `"flux2-klein" => {...}` 뒤:
   ```rust
               "flux3-klein" => {
                   InpaintingModel::Flux3Klein(file.processor.flux3_klein.clone().unwrap_or_default())
               }
   ```
4. `PipelineConfig::inpainting`의 `Flux2Klein` 분기 뒤:
   ```rust
               InpaintingModel::Flux3Klein(config) => Ok(InpaintingModel::Flux3Klein(
                   self.processor
                       .flux3_klein
                       .clone()
                       .unwrap_or_else(|| config.clone()),
               )),
   ```
5. `ProcessorConfig`의 `flux2_klein` 필드 뒤:
   ```rust
       #[serde(rename = "flux3-klein")]
       pub flux3_klein: Option<Flux3KleinConfig>,
   ```
그리고 `InpaintingModel` enum의 `Flux2Klein` 뒤에 추가한다:
```rust
    #[serde(rename = "flux3-klein")]
    Flux3Klein(Flux3KleinConfig),
```

- [ ] **Step 5: 재노출하고 다른 소비자를 맞춘다**

- `stages/mod.rs`의 `pub use inpainting::{...}`, `lib.rs`의 `pub use stages::{...}`에 `Flux3KleinConfig, Flux3KleinSourceConfig`를 알파벳 순서 위치에 추가한다.
- `inpainting.rs`
  - `InpaintingModel::validate`의 `Self::Flux2Klein(settings) => settings.validate()?,` 뒤에 `Self::Flux3Klein(settings) => settings.validate()?,`를 넣는다.
  - `StageProcessor::model`의 `Flux2Klein` 줄 뒤에 `InpaintingModel::Flux3Klein(_) => "flux3-klein",`를 넣는다.
- `bin/run.rs`
  - `use`에 `Flux3KleinConfig`를 추가한다.
  - `InpaintingChoice`의 `Flux2Klein,` 뒤에 `#[value(name = "flux3-klein")]\n    Flux3Klein,`를 넣는다.
  - 매핑 `match`에 `InpaintingChoice::Flux3Klein => InpaintingModel::Flux3Klein(Flux3KleinConfig::default()),`를 넣는다.
- `koharu-app/src/commands/preferences.rs`의 `remember_pipeline_profiles`에서 `Flux2Klein` 블록 뒤에 넣는다:
  ```rust
      if let koharu_pipeline::InpaintingModel::Flux3Klein(settings) = &config.inpainting {
          config.processor.flux3_klein = Some(settings.clone());
      }
  ```

- [ ] **Step 6: `Model::Flux3`를 추가한다 (`inpainting.rs`)**

`enum Model`의 `Flux { ... },` 뒤에 추가한다:
```rust
    Flux3 {
        inpaint: Arc<Mutex<Flux3KleinInpaint>>,
        upsampler: Arc<Flux3PromptUpsampler>,
        config: Box<Flux3KleinConfig>,
    },
```
`Model::load`의 `InpaintingModel::Flux2Klein(config) => {...}` 뒤에 추가한다:
```rust
            InpaintingModel::Flux3Klein(config) => {
                let source: Flux3KleinSource = config.source.clone().into();
                let (inpaint, upsampler) = tokio::try_join!(
                    Flux3KleinInpaint::load(device.clone(), &source),
                    Flux3PromptUpsampler::load(device, &source),
                )?;
                Ok(Self::Flux3 {
                    inpaint: Arc::new(Mutex::new(inpaint)),
                    upsampler: Arc::new(upsampler),
                    config: Box::new(config.clone()),
                })
            }
```
`Model::run`의 `Self::Flux { model, config } => {...}` 분기 뒤에 추가한다:
```rust
            Self::Flux3 {
                inpaint,
                upsampler,
                config,
            } => {
                let plan = TilePlan::new(
                    &prepared.image,
                    &prepared.mask,
                    &prepared.text_mask,
                    &prepared.flat_fill_regions,
                )?;
                // Phase 1 holds the accelerator with the upsampler alone and
                // drops it before phase 2 loads FLUX 3, so a 12 GB card never
                // holds both. Captions come from the untouched crops; phase 2
                // still cuts each tile from the output earlier tiles left.
                let captions = upsampler
                    .caption(plan.crops(), config.request.clone())
                    .await?;
                let inpaint = inpaint.clone();
                let options = config.options();
                (
                    "flux3-klein",
                    tokio_rayon::spawn(move || -> Result<DynamicImage> {
                        let model = inpaint
                            .lock()
                            .map_err(|_| anyhow!("FLUX 3 model lock is poisoned"))?;
                        plan.run(|index, image, mask| {
                            model.inference(
                                &captions[index],
                                image,
                                &DynamicImage::ImageLuma8(mask.clone()),
                                &options,
                            )
                        })
                    })
                    .await?,
                )
            }
```

- [ ] **Step 7: 테스트와 검사를 실행한다**

Run:
```bash
cargo test -p koharu-pipeline --lib stages::inpainting::tests
cargo test -p koharu-pipeline --lib config::tests
cargo check -p koharu-app -p koharu-pipeline --bins
```
Expected: 모두 PASS·성공. 컴파일러가 `match`가 빠짐없지 않다고 지적하면, 그 위치에 `Flux3Klein` 분기를 `Flux2Klein`과 같은 방식으로 추가한다.

- [ ] **Step 8: 실제 페이지로 한 번 확인한다 (수동)**

G4 머신에서 `koharu-pipeline`의 `run` 바이너리를 `--help`로 확인하고, 인페인팅 선택을 `flux3-klein`으로 해서 대표 페이지 1장을 처리한다. 확인할 것은 세 가지다.
- 로그에 `model_unload` 없이 페이지가 끝나는지
- 결과 PNG가 만들어지는지
- 처리 시간

시간은 Task 8의 스펙 표에 `페이지 1장 <초>, 타일 <개>` 행으로 덧붙인다.

- [ ] **Step 9: 커밋한다**

```bash
git add crates/koharu-pipeline crates/koharu-app/src/commands/preferences.rs specs/2026-10-17-flux3-klein-inpainting-design.md
git commit -m "feat(pipeline): inpaint pages with FLUX 3 Image Klein in two phases

Each page is captioned by the prompt upsampler first, which is dropped
before FLUX 3 inpaints the same tiles.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Phase E — UI와 문서 (G4 통과 후)

### Task 10: TS 프로토콜 타입

**Files:**
- Modify: `packages/bridge/src/protocol.ts`

- [ ] **Step 1: 타입을 추가한다**

`export type Flux2KleinSourceConfig = {...};` 바로 뒤에 추가한다:
```ts
export type Flux3KleinConfig = {
	/**  The instruction the prompt upsampler expands into one caption per tile. */
	request?: string,
	/**  Which checkpoints the image model and the upsampler are loaded from. */
	source?: Flux3KleinSourceConfig,
	steps?: number,
	strength?: number,
	/**  `-1` draws a fresh seed for every call. */
	seed?: number,
	padding_mask_crop?: number | null,
	/**  The working area every tile is shrunk to before denoising. */
	max_pixels?: number,
};

export type Flux3KleinSourceConfig = {
	transformer?: ComponentSourceConfig,
	text_encoder?: ComponentSourceConfig,
	vae?: ComponentSourceConfig,
	upsampler?: ComponentSourceConfig,
	upsampler_mmproj?: ComponentSourceConfig,
};
```
`InpaintingModel`에서 `model: "flux2-klein",\n} & Flux2KleinConfig` 뒤에 `| { model: "flux3-klein" } & Flux3KleinConfig`를 넣는다. `ProcessorConfig`의 `"flux2-klein"?: Flux2KleinConfig | null,` 뒤에 `"flux3-klein"?: Flux3KleinConfig | null,`를 넣는다.

- [ ] **Step 2: 타입 검사를 한다**

Run: `bun run --filter @koharu/bridge typecheck`
Expected: 성공.

- [ ] **Step 3: 커밋한다**

```bash
git add packages/bridge/src/protocol.ts
git commit -m "feat(bridge): type the flux3-klein inpainting settings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: i18n 키 (9개 로케일)

**Files:**
- Modify: `packages/koharu/public/locales/{en-US,es-ES,ja-JP,ko-KR,pt-BR,ru-RU,tr-TR,zh-CN,zh-TW}/translation.json`

모든 키는 `settings.pipeline.options` 객체 안에 넣는다. 위치 규칙은 다음과 같다.
- `request`와 `requestDescription`은 `"prompt"` 바로 뒤
- `upsampler`와 `upsampler_mmproj`는 `"source"` 객체 안의 `"transformer"` 뒤
- `flux3` 객체는 `"checkpointSource"` 뒤
`{{vram}}`, `{{ram}}`, `{{download}}` 보간자는 모든 로케일에 똑같이 남긴다.

- [ ] **Step 1: 실패하는 스키마 테스트를 확인한다**

먼저 `en-US`에만 추가한다:
```json
"request": "Upsampler request",
"requestDescription": "The official prompt upsampler turns this request into a full caption for every tile.",
```
```json
"upsampler": "Prompt upsampler",
"upsampler_mmproj": "Upsampler projector",
```
```json
"flux3": {
  "license": "FLUX Kommunity License: non-commercial use, and commercial output only for organizations under US$5M annual revenue.",
  "licenseLink": "Read the license",
  "resources": "Needs {{vram}} GB of VRAM, about {{ram}} GB of extra system RAM, and a {{download}} GB first download.",
  "speed": "Slower than FLUX.2 Klein: the upsampler and the image model take turns on the GPU for every page."
},
```
Run: `cd packages/koharu && bun run test -- tests/lib/localization.test.ts`
Expected: FAIL(`defines the same flattened translation schema in every locale`). 나머지 8개 로케일에 키가 없어서다.

- [ ] **Step 2: 나머지 8개 로케일에 추가한다**

| 키 | ko-KR | ja-JP |
|---|---|---|
| request | 업샘플러 지시문 | アップサンプラーへの指示 |
| requestDescription | 공식 프롬프트 업샘플러가 이 지시문으로 타일마다 전체 캡션을 만듭니다. | 公式のプロンプトアップサンプラーが、この指示からタイルごとに完全なキャプションを生成します。 |
| source.upsampler | 프롬프트 업샘플러 | プロンプトアップサンプラー |
| source.upsampler_mmproj | 업샘플러 프로젝터 | アップサンプラーのプロジェクター |
| flux3.license | FLUX Kommunity License: 비상용 이용과, 연매출 US$5M 미만 조직의 상용 출력만 허용됩니다. | FLUX Kommunity License：非商用利用と、年間売上 US$5M 未満の組織による商用出力のみ許可されます。 |
| flux3.licenseLink | 라이선스 보기 | ライセンスを読む |
| flux3.resources | VRAM {{vram}}GB, 추가 시스템 RAM 약 {{ram}}GB, 최초 다운로드 {{download}}GB가 필요합니다. | VRAM {{vram}} GB、追加のシステム RAM 約 {{ram}} GB、初回ダウンロード {{download}} GB が必要です。 |
| flux3.speed | FLUX.2 Klein보다 느립니다. 페이지마다 업샘플러와 이미지 모델이 번갈아 GPU를 씁니다. | FLUX.2 Klein より遅くなります。ページごとにアップサンプラーと画像モデルが交互に GPU を使います。 |

| 키 | zh-CN | zh-TW |
|---|---|---|
| request | 上采样器指令 | 上取樣器指令 |
| requestDescription | 官方提示词上采样器会根据此指令为每个图块生成完整描述。 | 官方提示詞上取樣器會依此指令為每個圖塊產生完整描述。 |
| source.upsampler | 提示词上采样器 | 提示詞上取樣器 |
| source.upsampler_mmproj | 上采样器投影器 | 上取樣器投影器 |
| flux3.license | FLUX Kommunity License：仅允许非商业用途，以及年收入低于 US$5M 的组织进行商业输出。 | FLUX Kommunity License：僅允許非商業用途，以及年營收低於 US$5M 的組織進行商業輸出。 |
| flux3.licenseLink | 阅读许可证 | 閱讀授權條款 |
| flux3.resources | 需要 {{vram}} GB 显存、约 {{ram}} GB 额外系统内存，首次下载 {{download}} GB。 | 需要 {{vram}} GB 顯示記憶體、約 {{ram}} GB 額外系統記憶體，首次下載 {{download}} GB。 |
| flux3.speed | 比 FLUX.2 Klein 慢：每页都会让上采样器和图像模型轮流使用 GPU。 | 比 FLUX.2 Klein 慢：每頁都會讓上取樣器與影像模型輪流使用 GPU。 |

| 키 | es-ES | pt-BR |
|---|---|---|
| request | Instrucción del upsampler | Instrução do upsampler |
| requestDescription | El upsampler de prompts oficial convierte esta instrucción en una descripción completa para cada mosaico. | O upsampler de prompts oficial transforma esta instrução em uma descrição completa para cada bloco. |
| source.upsampler | Upsampler de prompts | Upsampler de prompts |
| source.upsampler_mmproj | Proyector del upsampler | Projetor do upsampler |
| flux3.license | FLUX Kommunity License: uso no comercial y salida comercial solo para organizaciones con ingresos anuales inferiores a 5 M US$. | FLUX Kommunity License: uso não comercial e saída comercial apenas para organizações com receita anual abaixo de US$ 5 milhões. |
| flux3.licenseLink | Leer la licencia | Ler a licença |
| flux3.resources | Requiere {{vram}} GB de VRAM, unos {{ram}} GB de RAM adicional del sistema y una primera descarga de {{download}} GB. | Requer {{vram}} GB de VRAM, cerca de {{ram}} GB de RAM extra do sistema e um primeiro download de {{download}} GB. |
| flux3.speed | Más lento que FLUX.2 Klein: el upsampler y el modelo de imagen se turnan la GPU en cada página. | Mais lento que o FLUX.2 Klein: o upsampler e o modelo de imagem revezam a GPU a cada página. |

| 키 | ru-RU | tr-TR |
|---|---|---|
| request | Инструкция для апсемплера | Upsampler talimatı |
| requestDescription | Официальный апсемплер промптов превращает эту инструкцию в полное описание для каждого фрагмента. | Resmî istem upsampler'ı bu talimatı her döşeme için eksiksiz bir açıklamaya dönüştürür. |
| source.upsampler | Апсемплер промптов | İstem upsampler'ı |
| source.upsampler_mmproj | Проектор апсемплера | Upsampler projektörü |
| flux3.license | FLUX Kommunity License: некоммерческое использование и коммерческий вывод только для организаций с годовым доходом менее 5 млн долларов США. | FLUX Kommunity License: ticari olmayan kullanım ve yalnızca yıllık geliri 5 milyon ABD dolarının altındaki kuruluşlar için ticari çıktı. |
| flux3.licenseLink | Прочитать лицензию | Lisansı oku |
| flux3.resources | Требуется {{vram}} ГБ видеопамяти, около {{ram}} ГБ дополнительной оперативной памяти и {{download}} ГБ при первой загрузке. | {{vram}} GB VRAM, yaklaşık {{ram}} GB ek sistem belleği ve ilk indirme için {{download}} GB gerekir. |
| flux3.speed | Медленнее, чем FLUX.2 Klein: апсемплер и модель изображения по очереди используют GPU на каждой странице. | FLUX.2 Klein'dan yavaştır: upsampler ve görüntü modeli her sayfada GPU'yu sırayla kullanır. |

- [ ] **Step 3: 테스트를 실행한다**

Run: `cd packages/koharu && bun run test -- tests/lib/localization.test.ts`
Expected: PASS.

- [ ] **Step 4: 커밋한다**

```bash
git add packages/koharu/public/locales
git commit -m "feat(i18n): describe the FLUX 3 Image Klein options and notices

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: 옵션 패널과 모델 목록

**Files:**
- Create: `packages/koharu/components/preferences/Flux3KleinOptions.tsx`
- Modify: `packages/koharu/components/preferences/models.ts`
- Modify: `packages/koharu/components/preferences/PipelinePreferences.tsx`
- Test: `packages/koharu/tests/components/flux3-klein-options.test.tsx`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```tsx
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { Flux3KleinOptions } from '@/components/preferences/Flux3KleinOptions'
import { defaultModel, modelNames, modelOptions } from '@/components/preferences/models'

vi.mock('@tauri-apps/plugin-opener', () => ({ openUrl: vi.fn(async () => undefined) }))

describe('FLUX 3 Image Klein preferences', () => {
  it('is offered after FLUX.2 Klein with the backend default request', () => {
    const inpainting = modelOptions.inpainting as readonly string[]
    expect(inpainting.indexOf('flux3-klein')).toBe(inpainting.indexOf('flux2-klein') + 1)
    expect(modelNames['flux3-klein']).toBe('FLUX 3 Image Klein')
    expect(defaultModel('flux3-klein')).toEqual({
      model: 'flux3-klein',
      request: 'Remove all text and lettering, and reconstruct the background behind it.',
    })
  })

  it('discloses the license, resources and speed', () => {
    render(<Flux3KleinOptions value={{}} onChange={vi.fn()} />)
    expect(screen.getByText(/^FLUX Kommunity License/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Read the license' })).toHaveAttribute(
      'href',
      'https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md',
    )
    expect(screen.getByText(/GB of VRAM/)).toBeInTheDocument()
    expect(screen.getByText(/^Slower than FLUX.2 Klein/)).toBeInTheDocument()
  })

  it('edits the request and lists all five checkpoint sources', () => {
    const onChange = vi.fn()
    render(<Flux3KleinOptions value={{}} onChange={onChange} />)
    fireEvent.change(screen.getByRole('textbox', { name: 'Upsampler request' }), {
      target: { value: 'Erase the lettering.' },
    })
    expect(onChange).toHaveBeenCalledWith({ request: 'Erase the lettering.' })
    for (const label of [
      'Transformer',
      'Text encoder',
      'VAE',
      'Prompt upsampler',
      'Upsampler projector',
    ]) {
      expect(screen.getByText(label)).toBeInTheDocument()
    }
  })
})
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cd packages/koharu && bun run test -- tests/components/flux3-klein-options.test.tsx`
Expected: FAIL(모듈 `@/components/preferences/Flux3KleinOptions`를 찾을 수 없음).

- [ ] **Step 3: 패널을 만든다**

`packages/koharu/components/preferences/Flux3KleinOptions.tsx`:
```tsx
'use client'

import { openUrl } from '@tauri-apps/plugin-opener'
import { useTranslation } from 'react-i18next'

import { ComponentSourceField, NumberField, TextField } from '@/components/preferences/PreferenceFields'
import type { Flux3KleinConfig } from '@koharu/bridge/protocol'

const DEFAULT_REQUEST = 'Remove all text and lettering, and reconstruct the background behind it.'
const DEFAULT_STEPS = 16
const DEFAULT_STRENGTH = 0.8
const DEFAULT_SEED = -1
const DEFAULT_MAX_PIXELS = 1024 * 1024

// Measured at gate G4 (specs/2026-10-17-flux3-klein-inpainting-design.md).
// The locale strings interpolate these, so a new measurement changes only here.
const VRAM_GB = 12
const RAM_GB = 12
const DOWNLOAD_GB = 13

const LICENSE_URL =
  'https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md'

const COMPONENTS = [
  'transformer',
  'text_encoder',
  'vae',
  'upsampler',
  'upsampler_mmproj',
] as const

/**
 * Inference settings, checkpoint overrides and the license and resource
 * notices for FLUX 3 Image Klein. Unset fields fall back to the backend's
 * defaults.
 */
export function Flux3KleinOptions({
  value,
  onChange,
}: {
  value: Flux3KleinConfig
  onChange: (changes: Partial<Flux3KleinConfig>) => void
}) {
  const { t } = useTranslation()
  const source = value.source ?? {}
  return (
    <div className='grid gap-3'>
      <div className='grid gap-1 rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-[11px] text-amber-700 dark:text-amber-300'>
        <p>
          {t('settings.pipeline.options.flux3.license')}{' '}
          <a
            href={LICENSE_URL}
            className='underline'
            onClick={(event) => {
              event.preventDefault()
              void openUrl(LICENSE_URL).catch(() => undefined)
            }}
          >
            {t('settings.pipeline.options.flux3.licenseLink')}
          </a>
        </p>
        <p>
          {t('settings.pipeline.options.flux3.resources', {
            vram: VRAM_GB,
            ram: RAM_GB,
            download: DOWNLOAD_GB,
          })}
        </p>
        <p>{t('settings.pipeline.options.flux3.speed')}</p>
      </div>
      <div className='grid gap-1'>
        <TextField
          label={t('settings.pipeline.options.request')}
          value={value.request ?? DEFAULT_REQUEST}
          onChange={(request) => onChange({ request })}
        />
        <p className='text-[10px] text-muted-foreground'>
          {t('settings.pipeline.options.requestDescription')}
        </p>
      </div>
      <div className='grid grid-cols-3 gap-2'>
        <NumberField
          label={t('settings.pipeline.options.steps')}
          value={value.steps ?? DEFAULT_STEPS}
          min={1}
          step={1}
          onChange={(steps) => onChange({ steps: steps ?? DEFAULT_STEPS })}
        />
        <NumberField
          label={t('settings.pipeline.options.strength')}
          value={value.strength ?? DEFAULT_STRENGTH}
          min={0.05}
          max={1}
          step={0.05}
          onChange={(strength) => onChange({ strength: strength ?? DEFAULT_STRENGTH })}
        />
        <NumberField
          label={t('settings.pipeline.options.seed')}
          value={value.seed ?? DEFAULT_SEED}
          step={1}
          onChange={(seed) => onChange({ seed: seed ?? DEFAULT_SEED })}
        />
        <NumberField
          label={t('settings.pipeline.options.maxPixels')}
          value={value.max_pixels ?? DEFAULT_MAX_PIXELS}
          min={64 * 64}
          step={65536}
          onChange={(max_pixels) => onChange({ max_pixels: max_pixels ?? DEFAULT_MAX_PIXELS })}
        />
        <NumberField
          label={t('settings.pipeline.options.paddingMaskCrop')}
          value={value.padding_mask_crop ?? null}
          min={0}
          step={1}
          onChange={(padding_mask_crop) => onChange({ padding_mask_crop })}
        />
      </div>
      <div className='grid gap-2'>
        <p className='text-[10px] text-muted-foreground'>
          {t('settings.pipeline.options.sourceDescription')}
        </p>
        {COMPONENTS.map((component) => (
          <ComponentSourceField
            key={component}
            label={t(`settings.pipeline.options.source.${component}`)}
            value={source[component] ?? { kind: 'builtin' }}
            onChange={(replacement) =>
              onChange({ source: { ...source, [component]: replacement } })
            }
          />
        ))}
      </div>
    </div>
  )
}
```
`DOWNLOAD_GB = 13`은 확정값이다(12.36GB를 올림). `VRAM_GB`와 `RAM_GB`는 Task 8 표의 G4 실측값을 GB 단위 올림 정수로 맞춘다. 위의 12는 스펙의 추정치이므로, 실측이 다르면 실측값을 쓴다.

- [ ] **Step 4: 모델 목록과 패널을 연결한다**

`models.ts`:
- `modelOptions.inpainting`의 `'flux2-klein',` 뒤에 `'flux3-klein',`을 넣는다.
- `modelNames`의 `'flux2-klein': 'FLUX.2 Klein',` 뒤에 `'flux3-klein': 'FLUX 3 Image Klein',`을 넣는다.
- `defaultModel`의 `case 'flux2-klein': ...` 뒤에 넣는다:
  ```ts
      case 'flux3-klein':
        return {
          model,
          request: 'Remove all text and lettering, and reconstruct the background behind it.',
        }
  ```
- `replaceStage`의 `'inpainting'` 분기에서 `flux2-klein` 스프레드 뒤에 넣는다:
  ```ts
            ...(model.model === 'flux3-klein'
              ? {
                  'flux3-klein': {
                    request: model.request ?? undefined,
                    source: model.source ?? undefined,
                    steps: model.steps ?? undefined,
                    strength: model.strength ?? undefined,
                    seed: model.seed ?? undefined,
                    padding_mask_crop: model.padding_mask_crop ?? undefined,
                    max_pixels: model.max_pixels ?? undefined,
                  },
                }
              : {}),
  ```

`PipelinePreferences.tsx`:
- `import { Flux2KleinOptions } ...` 아래에 `import { Flux3KleinOptions } from '@/components/preferences/Flux3KleinOptions'`를 넣는다.
- `case 'flux2-klein': ...` 블록 뒤에 넣는다:
  ```tsx
      case 'flux3-klein':
        return (
          <Flux3KleinOptions
            value={model}
            onChange={(changes) => onChange({ ...model, ...changes })}
          />
        )
  ```

- [ ] **Step 5: 테스트를 실행한다**

Run:
```bash
cd packages/koharu && bun run test -- tests/components/flux3-klein-options.test.tsx
cd packages/koharu && bun x tsc --noEmit && bun run lint
```
Expected: 테스트 3개 PASS. `koharu` 패키지에는 `typecheck` 스크립트가 없으므로 `tsc --noEmit`을 직접 실행하고, 타입 검사와 lint 모두 성공해야 한다.

- [ ] **Step 6: 커밋한다**

```bash
git add packages/koharu/components/preferences packages/koharu/tests/components/flux3-klein-options.test.tsx
git commit -m "feat(ui): offer FLUX 3 Image Klein with its license and resource notices

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: 문서

**Files:**
- Modify: `packages/docs/{en,ja,zh}/models/vision.mdx`, `guides/cleanup.mdx`, `hardware.mdx`

- [ ] **Step 1: `models/vision.mdx`**

`**FLUX.2 Klein** accepts a reconstruction prompt.`로 시작하는 줄(en 36, ja 36, zh 35) 바로 뒤에 빈 줄을 하나 두고 다음 단락을 넣는다. 들여쓰기는 앞 줄과 같은 4칸이다. `<VRAM>`, `<RAM>`, `<DL>`은 Task 12에서 확정한 `VRAM_GB`, `RAM_GB`, `DOWNLOAD_GB` 값으로 바꾼다.
- en: `**FLUX 3 Image Klein** runs the official prompt upsampler on every tile before inpainting, so it takes a removal request instead of a prompt. It needs <VRAM> GB of VRAM, about <RAM> GB of extra system RAM, and a <DL> GB first download, and it is released under the FLUX Kommunity License (non-commercial use, plus commercial output for organizations under US$5M annual revenue).`
- ja: `**FLUX 3 Image Klein** はインペインティングの前にタイルごとに公式のプロンプトアップサンプラーを実行するため、プロンプトではなく除去の指示を受け取ります。VRAM <VRAM> GB、追加のシステム RAM 約 <RAM> GB、初回ダウンロード <DL> GB が必要で、FLUX Kommunity License（非商用利用と、年間売上 US$5M 未満の組織による商用出力）で提供されます。`
- zh: `**FLUX 3 Image Klein** 在修复前会对每个图块运行官方提示词上采样器，因此接受的是去除指令而不是提示词。它需要 <VRAM> GB 显存、约 <RAM> GB 额外系统内存和 <DL> GB 首次下载，采用 FLUX Kommunity License（非商业用途，以及年收入低于 US$5M 的组织的商业输出）。`

- [ ] **Step 2: `guides/cleanup.mdx`**

생성형 모델을 나열하는 문장(en 38, ja 34, zh 34)을 다음과 같이 바꾼다.
- en: `**LaMa** and **AOT** provide direct inpainting. **FLUX.2 Klein**, **FLUX 3 Image Klein**, and **RORem Mixed** are generative alternatives with prompt controls and larger resource requirements; FLUX 3 Image Klein is the slowest because it captions every tile first. See [vision and inpainting models](/en/models/vision).`
- ja: `**LaMa** と **AOT** は直接型、**FLUX.2 Klein**、**FLUX 3 Image Klein**、**RORem Mixed** はプロンプトを使う生成型です。生成型はより多くのリソースを必要とし、FLUX 3 Image Klein は先にタイルごとにキャプションを作るため最も遅くなります。[画像処理モデル](/ja/models/vision)を参照してください。`
- zh: `**LaMa** 和 **AOT** 是直接修复模型；**FLUX.2 Klein**、**FLUX 3 Image Klein** 和 **RORem Mixed** 是带提示词的生成式选择，资源要求更高；FLUX 3 Image Klein 会先为每个图块生成描述，因此最慢。详见[视觉模型](/zh/models/vision)。`

- [ ] **Step 3: `hardware.mdx`**

Vulkan 탭의 모델 나열에서 `FLUX.2 Klein` 바로 뒤에 FLUX 3를 넣는다.
- en(34행): `FLUX.2 Klein, and RORem` → `FLUX.2 Klein, FLUX 3 Image Klein, and RORem`
- ja(30행): `FLUX.2 Klein、RORem` → `FLUX.2 Klein、FLUX 3 Image Klein、RORem`
- zh(28행): `FLUX.2 Klein 和 RORem` → `FLUX.2 Klein、FLUX 3 Image Klein 和 RORem`

- [ ] **Step 4: 커밋한다**

```bash
git add packages/docs
git commit -m "docs: describe FLUX 3 Image Klein inpainting

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## 범위 밖 (이 계획에서 하지 않는 것)

- 공용 VRAM↔RAM 레지던시 매니저(스펙 D8, 후속 조사)
- llama 런타임 최신화(G2가 실패할 때만 별도 작업으로 분리)
- 캡션 캐시, 업샘플러 끄기 UI, Klein Base, bbox 프롬프트, 다중 참조 편집
