# Manga OCR CPT-flex 옵션 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `muscgab/manga-ocr-base-CPT-flex`를 기존 `manga-ocr` 옆에 새 OCR 옵션 `manga-ocr-cpt-flex`로 추가한다.

**Architecture:** `crates/koharu-ml/src/manga_ocr_cpt/`에 독립 모듈(config, processor, model, mod)을 만든다. 네트워크는 업스트림 `model.py`의 파라미터 경로(`enc.*`, `dec.*`)와 실행 순서를 그대로 따르고, 전처리는 `preprocess.py`의 `aspect` 경로를 따른다. 파이프라인 `OcrModel`, CLI, 프론트엔드 옵션, 문서에 한 항목씩 추가한다.

**Tech Stack:** Rust, `koharu-torch`(VarStore, SDPA, `upsample_bicubic2d`), `fast_image_resize`, `image`, `serde_json`, TypeScript(React).

**설계 문서:** `specs/2026-10-17-manga-ocr-cpt-flex-design.md`

**권위 구현:** `https://github.com/muscgab/manga-ocr-CPT/tree/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt`
**가중치:** `muscgab/manga-ocr-base-CPT-flex` @ `9cc15742f8dc37cb5222a6569839bc774dfc0392`

---

## 계획 작성 시 확정한 사실

설계 이후 직접 확인했다. 구현 중 다시 찾지 않는다.

| 항목 | 값 | 확인 방법 |
|---|---|---|
| 체크포인트 텐서 수 / 파라미터 | 260 / 110,935,974 | safetensors 헤더 |
| 어휘 파일 | `{"character_count": 6817, "characters": [...], ...}`. serde_json으로 문제없이 파싱됨 | 고정 리비전 파일을 Rust로 파싱 |
| Pillow `convert("L")` | `(R*19595 + G*38470 + B*7471 + 0x8000) >> 16`. 흰 255, 검정 0, 빨강 76, 초록 150, 파랑 29, `(10,200,90)` 131 | Pillow 12.3과 대조 |
| `np.median(...).astype(uint8)` | 짝수 개는 가운데 두 값 평균 후 버림: `[1,2,3,4]` → 2, `[10,11]` → 10 | numpy 2.5 |
| 업스트림 `grid` | `1024×1024` → 격자 14×14, 크기 224×224 / `120×900` → 5×38, 600×80 / `900×120` → 38×5, 80×600 / `4×4` → 14×14, 224×224 / `30×2000` → 2×98, 1568×24 | 업스트림 함수를 그대로 옮긴 Python |
| `title.png`(280×53) 캔버스 | 격자 6×32, 리사이즈 507×96, 채움값 255, 캔버스 512×96, 픽셀 합 10,422,116 | 같은 방법 |
| **G4 기대값** | `title.png` → `"対策委員会です!"`, 토큰 id `[1767, 4323, 1601, 1238, 653, 377, 363, 5]` | 업스트림 `model.py`·`preprocess.py`(고정 커밋)와 고정 가중치를 CPU fp32 plain 경로로 실행 |

설계 문서와 다른 점:
- 영어 문서는 upstream 병합(#1211) 뒤 `packages/docs/models/vision.mdx`로 옮겨졌다. 일본어는 `packages/docs/ja/models/vision.mdx`, 중국어는 `packages/docs/zh/models/vision.mdx`다.
- 설계의 테스트 표에 있던 "파이프라인 등록 문자열 테스트"는 넣지 않는다. upstream 병합으로 들어온 AGENTS.md가 "registration string을 고정하는 테스트를 추가하지 말라"고 정한다. G2 컴파일과 G5 타입 검사가 등록 누락을 잡는다.

## 파일 구조

| 파일 | 책임 | 변경 |
|---|---|---|
| `crates/koharu-ml/src/manga_ocr_cpt/config.rs` | 아키텍처·입력 상수, `config.json` 검증 | 생성 |
| `crates/koharu-ml/src/manga_ocr_cpt/processor.rs` | 흑백·격자·캔버스·패치화, 어휘 디코딩 | 생성 |
| `crates/koharu-ml/src/manga_ocr_cpt/model.rs` | ViT 인코더, post-LN BERT 디코더, greedy 생성 | 생성 |
| `crates/koharu-ml/src/manga_ocr_cpt/mod.rs` | 공개 수명주기, 체크포인트 고정 | 생성 |
| `crates/koharu-ml/src/lib.rs` | 모듈 공개 | 1줄 |
| `crates/koharu-pipeline/src/config.rs` | `OcrModel` 등록 | 3곳 |
| `crates/koharu-pipeline/src/stages/ocr.rs` | stage 모델 분기 | 5곳 |
| `crates/koharu-pipeline/src/bin/run.rs` | CLI 선택지 | 2곳 |
| `packages/bridge/src/protocol.ts` | `OcrModel` 유니온 | 1곳 |
| `packages/koharu/components/preferences/models.ts` | 옵션·표시 이름·기본값 | 3곳 |
| `packages/koharu/components/preferences/PipelinePreferences.tsx` | 설정 없음 분기 | 1곳 |
| `packages/docs/models/vision.mdx`, `packages/docs/{ja,zh}/models/vision.mdx` | 모델 표 | 각 1줄 |

---

## Task 1: 상수와 설정 검증

**Files:**
- Create: `crates/koharu-ml/src/manga_ocr_cpt/config.rs`
- Create: `crates/koharu-ml/src/manga_ocr_cpt/mod.rs`(이 Task에서는 모듈 선언만)
- Modify: `crates/koharu-ml/src/lib.rs`

- [ ] **Step 1: 모듈 골격을 만든다**

`crates/koharu-ml/src/manga_ocr_cpt/mod.rs`:

```rust
mod config;
```

`crates/koharu-ml/src/lib.rs`의 `pub mod manga_ocr;` 바로 아래에 추가한다.

```rust
pub mod manga_ocr_cpt;
```

- [ ] **Step 2: 실패하는 테스트를 담은 `config.rs`를 쓴다**

```rust
//! Manga OCR CPT-flex constants and checkpoint configuration.
//!
//! Upstream hardcodes the network constants (`D`, `HEADS`, `EPS`, `MAX_NEW`)
//! and the input budget instead of reading them from `config.json`:
//! https://github.com/muscgab/manga-ocr-CPT/blob/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt/model.py
//! https://github.com/muscgab/manga-ocr-CPT/blob/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt/preprocess.py
//! `config.json` is therefore only checked against them.

use std::{fs, path::Path};

use anyhow::{Result, ensure};
use serde::Deserialize;

#[cfg(test)]
mod tests {
    use super::ModelConfig;

    /// `config.json` of the pinned checkpoint, reduced to the fields Koharu reads.
    const RELEASED: &str = r#"{
        "vocab": 6817,
        "special_tokens": {"pad": 0, "unk": 1, "start": 2, "end": 3, "mask": 4, "first_char": 5},
        "input": {"grey": true, "fit": "aspect", "max_patches": 196, "patch": 16,
                  "resample": "bilinear", "fill": "median border value", "mean": 0.5, "std": 0.5},
        "max_new_tokens": 65
    }"#;

    #[test]
    fn the_released_config_matches_the_ported_constants() {
        let config: ModelConfig = serde_json::from_str(RELEASED).unwrap();
        config.validate().unwrap();
    }

    #[test]
    fn the_square_input_variant_is_rejected() {
        let config: ModelConfig =
            serde_json::from_str(&RELEASED.replace("\"aspect\"", "\"square\"")).unwrap();
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("aspect"), "{error}");
    }
}
```

- [ ] **Step 3: 실패를 확인한다**

Run: `cargo test -p koharu-ml --lib manga_ocr_cpt::config`
Expected: 컴파일 실패 — `unresolved import super::ModelConfig`

- [ ] **Step 4: 상수와 `ModelConfig`를 구현한다**

`config.rs`의 `use` 블록과 `#[cfg(test)]` 사이에 넣는다.

```rust
pub(super) const HIDDEN_SIZE: i64 = 768;
pub(super) const NUM_HEADS: i64 = 12;
pub(super) const HEAD_DIM: i64 = HIDDEN_SIZE / NUM_HEADS;
pub(super) const INTERMEDIATE_SIZE: i64 = 3072;
pub(super) const ENCODER_LAYERS: usize = 12;
pub(super) const DECODER_LAYERS: usize = 2;
/// Every LayerNorm, encoder and decoder alike, uses BERT's 1e-12.
pub(super) const LAYER_NORM_EPS: f64 = 1e-12;
pub(super) const PATCH_SIZE: i64 = 16;
/// Side of the learned position table of the original 224x224 ViT.
pub(super) const POSITION_GRID: i64 = 14;
pub(super) const MAX_PATCHES: i64 = POSITION_GRID * POSITION_GRID;
/// Three equal grey channels times one 16x16 patch.
pub(super) const PATCH_FEATURES: i64 = 3 * PATCH_SIZE * PATCH_SIZE;
pub(super) const DECODER_POSITIONS: i64 = 512;
pub(super) const CHARACTERS: i64 = 6817;
/// Ids 0..5 are pad, unk, start, end, and mask; character `i` is id `5 + i`.
pub(super) const SPECIAL_TOKENS: i64 = 5;
pub(super) const OUTPUT_TOKENS: i64 = CHARACTERS + SPECIAL_TOKENS;
pub(super) const START_TOKEN: i64 = 2;
pub(super) const END_TOKEN: i64 = 3;
/// At most 64 characters plus the end token.
pub(super) const MAX_NEW_TOKENS: i64 = 65;

#[derive(Debug, Deserialize)]
pub(super) struct ModelConfig {
    vocab: i64,
    special_tokens: SpecialTokens,
    input: InputConfig,
    max_new_tokens: i64,
}

#[derive(Debug, Deserialize)]
struct SpecialTokens {
    pad: i64,
    unk: i64,
    start: i64,
    end: i64,
    mask: i64,
    first_char: i64,
}

#[derive(Debug, Deserialize)]
struct InputConfig {
    grey: bool,
    fit: String,
    max_patches: i64,
    patch: i64,
    mean: f64,
    std: f64,
}

impl ModelConfig {
    pub(super) fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.vocab == CHARACTERS,
            "Manga OCR CPT-flex expects {CHARACTERS} characters, the config has {}",
            self.vocab
        );
        let tokens = &self.special_tokens;
        ensure!(
            (tokens.pad, tokens.unk, tokens.start, tokens.end, tokens.mask, tokens.first_char)
                == (0, 1, START_TOKEN, END_TOKEN, 4, SPECIAL_TOKENS),
            "Manga OCR CPT-flex special token ids differ from the ported layout"
        );
        ensure!(
            self.input.fit == "aspect",
            "Manga OCR CPT-flex expects the aspect-preserving input, the config has fit = {:?}",
            self.input.fit
        );
        ensure!(self.input.grey, "Manga OCR CPT-flex expects grey input");
        ensure!(
            self.input.max_patches == MAX_PATCHES && self.input.patch == PATCH_SIZE,
            "Manga OCR CPT-flex expects at most {MAX_PATCHES} patches of {PATCH_SIZE} px"
        );
        ensure!(
            self.input.mean == 0.5 && self.input.std == 0.5,
            "Manga OCR CPT-flex expects mean and std of 0.5"
        );
        ensure!(
            self.max_new_tokens == MAX_NEW_TOKENS,
            "Manga OCR CPT-flex expects at most {MAX_NEW_TOKENS} generated tokens"
        );
        Ok(())
    }
}
```

- [ ] **Step 5: 통과를 확인한다**

Run: `cargo test -p koharu-ml --lib manga_ocr_cpt::config`
Expected: 2개 통과. 상수의 dead_code 경고는 Task 3·4에서 사라진다.

- [ ] **Step 6: 커밋한다**

```bash
cargo fmt -p koharu-ml
git add crates/koharu-ml/src/lib.rs crates/koharu-ml/src/manga_ocr_cpt
git commit -m "feat(manga-ocr-cpt): add the CPT-flex constants and config checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 2: 전처리와 어휘

**Files:**
- Create: `crates/koharu-ml/src/manga_ocr_cpt/processor.rs`
- Modify: `crates/koharu-ml/src/manga_ocr_cpt/mod.rs`

- [ ] **Step 1: 모듈을 선언한다**

`mod.rs`를 다음으로 바꾼다.

```rust
mod config;
mod processor;
```

- [ ] **Step 2: 실패하는 테스트를 담은 `processor.rs`를 쓴다**

```rust
//! Manga OCR CPT-flex image and text processing.
//!
//! Port of the `fit="aspect"` path of the upstream CPU preprocessing and of
//! `MangaOcrCPT._text`:
//! https://github.com/muscgab/manga-ocr-CPT/blob/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt/preprocess.py
//! https://github.com/muscgab/manga-ocr-CPT/blob/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt/__init__.py

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use image::{GrayImage, Luma};

    use super::*;

    #[test]
    fn luma_matches_pillow_convert_l() {
        assert_eq!(luma([255, 255, 255]), 255);
        assert_eq!(luma([0, 0, 0]), 0);
        assert_eq!(luma([255, 0, 0]), 76);
        assert_eq!(luma([0, 255, 0]), 150);
        assert_eq!(luma([0, 0, 255]), 29);
        assert_eq!(luma([10, 200, 90]), 131);
    }

    #[test]
    fn grid_matches_the_upstream_binary_search() {
        // (height, width) -> (grid h, grid w, resized width, resized height),
        // all taken from the upstream `grid` and resize extents.
        for ((height, width), (cells_h, cells_w, resized_w, resized_h)) in [
            ((1024, 1024), (14, 14, 224, 224)),
            ((120, 900), (5, 38, 600, 80)),
            ((900, 120), (38, 5, 80, 600)),
            ((4, 4), (14, 14, 224, 224)),
            ((30, 2000), (2, 98, 1568, 24)),
        ] {
            let (scale, grid_h, grid_w) = grid(height, width);
            assert_eq!((grid_h, grid_w), (cells_h, cells_w), "{height}x{width}");
            assert!(grid_h * grid_w <= MAX_PATCHES);
            assert_eq!(resized_extent(width, scale, grid_w), resized_w, "{height}x{width}");
            assert_eq!(resized_extent(height, scale, grid_h), resized_h, "{height}x{width}");
        }
    }

    #[test]
    fn resized_extent_rounds_half_to_even_like_python() {
        assert_eq!(resized_extent(5, 0.5, 14), 2);
        assert_eq!(resized_extent(7, 0.5, 14), 4);
        assert_eq!(resized_extent(1, 0.01, 14), 1);
        assert_eq!(resized_extent(1000, 1.0, 2), 32);
    }

    #[test]
    fn border_median_averages_then_truncates_like_numpy() {
        // A 2x2 image has border [1, 2, 3, 4, 1, 3, 2, 4] (corners twice).
        let image = GrayImage::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(border_median(&image), 2);
        let image = GrayImage::from_raw(2, 1, vec![10, 11]).unwrap();
        assert_eq!(border_median(&image), 10);
        assert_eq!(border_median(&GrayImage::from_pixel(3, 3, Luma([7]))), 7);
    }

    #[test]
    fn the_fixture_canvas_matches_upstream() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/fixtures/ocr/title.png");
        let canvas = canvas(&grey(&image::open(path).unwrap())).unwrap();
        assert_eq!(canvas.grid, [6, 32]);
        assert_eq!(canvas.image.dimensions(), (512, 96));
        // Right of the 507-wide resize is the median fill.
        assert_eq!(canvas.image.get_pixel(511, 0)[0], 255);
        // Pillow sums the canvas to 10,422,116; `fast_image_resize` may round a
        // few coefficients differently, so allow one level per pixel on average.
        let sum: u64 = canvas.image.as_raw().iter().map(|&value| u64::from(value)).sum();
        assert!(sum.abs_diff(10_422_116) <= 512 * 96, "sum {sum}");
    }

    #[test]
    fn decoding_skips_special_tokens_and_offsets_characters() {
        let vocabulary = Vocabulary {
            characters: vec!["a".into(), "b".into(), "c".into()],
        };
        assert_eq!(vocabulary.decode(&[5, 0, 7, 1, 4, 6]), "acb");
        assert_eq!(vocabulary.decode(&[]), "");
    }
}
```

- [ ] **Step 3: 실패를 확인한다**

Run: `cargo test -p koharu-ml --lib manga_ocr_cpt::processor`
Expected: 컴파일 실패 — `cannot find function luma` 등

- [ ] **Step 4: 전처리와 어휘를 구현한다**

`processor.rs`의 모듈 doc 주석 아래, `#[cfg(test)]` 위에 넣는다.

```rust
use std::{fs, path::Path};

use anyhow::{Result, ensure};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, GrayImage, Luma};
use koharu_torch::{Device, Kind, Tensor};
use serde::Deserialize;

use super::config::{MAX_PATCHES, PATCH_FEATURES, PATCH_SIZE, SPECIAL_TOKENS};

#[derive(Debug)]
pub(super) struct ImageProcessor {
    device: Device,
}

/// One preprocessed crop.
#[derive(Debug)]
pub(super) struct ImageInput {
    /// Patch features `(1, MAX_PATCHES, PATCH_FEATURES)` in `[-1, 1]`, zero
    /// after the valid patches.
    pub(super) patches: Tensor,
    /// Valid-patch mask `(1, MAX_PATCHES)`.
    pub(super) valid: Tensor,
    /// Patch grid `(height, width)` of the valid patches.
    pub(super) grid: [i64; 2],
}

impl ImageProcessor {
    pub(super) fn new(device: Device) -> Self {
        Self { device }
    }

    pub(super) fn preprocess(&self, image: &DynamicImage) -> Result<ImageInput> {
        if image.width() == 0 || image.height() == 0 {
            anyhow::bail!("cannot recognize an empty image");
        }
        let canvas = canvas(&grey(image))?;
        let [grid_h, grid_w] = canvas.grid;
        let count = grid_h * grid_w;
        // (gh*16, gw*16) -> (gh*gw, channel, row, column), the upstream
        // `transpose(0, 2, 4, 1, 3)` order; the three channels are equal.
        let patches = Tensor::from_slice(canvas.image.as_raw())
            .to_device(self.device)
            .to_kind(Kind::Float)
            .view([grid_h, PATCH_SIZE, grid_w, PATCH_SIZE])
            .permute([0, 2, 1, 3])
            .reshape([count, 1, PATCH_SIZE * PATCH_SIZE])
            .expand([count, 3, PATCH_SIZE * PATCH_SIZE], false)
            .reshape([count, PATCH_FEATURES]);
        let patches = (patches / 255.0 - 0.5) / 0.5;
        let padding = Tensor::zeros(
            [MAX_PATCHES - count, PATCH_FEATURES],
            (Kind::Float, self.device),
        );
        let patches = Tensor::cat(&[patches, padding], 0).unsqueeze(0);
        let valid = Tensor::arange(MAX_PATCHES, (Kind::Int64, self.device))
            .lt(count)
            .unsqueeze(0);
        Ok(ImageInput {
            patches,
            valid,
            grid: [grid_h, grid_w],
        })
    }
}

/// Pillow `Image.convert("L")` from RGB: ITU-R 601-2 luma in 16-bit fixed
/// point. The `image` crate's luma uses Rec. 709 weights and would not match
/// the training input.
fn luma([red, green, blue]: [u8; 3]) -> u8 {
    ((u32::from(red) * 19595 + u32::from(green) * 38470 + u32::from(blue) * 7471 + 0x8000) >> 16)
        as u8
}

/// Upstream `_rgb` drops alpha without compositing, as `to_rgb8` does.
fn grey(image: &DynamicImage) -> GrayImage {
    let rgb = image.to_rgb8();
    GrayImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        Luma([luma(rgb.get_pixel(x, y).0)])
    })
}

/// Upstream `grid`: the largest uniform scale whose rounded-up patch grid
/// stays within the budget, found by 60 bisection steps over `[1e-4, 1000]`.
fn grid(height: u32, width: u32) -> (f64, i64, i64) {
    let cells = |size: u32, scale: f64| (f64::from(size) * scale / PATCH_SIZE as f64).ceil() as i64;
    let (mut low, mut high) = (1e-4, 1000.0);
    for _ in 0..60 {
        let middle = (low + high) / 2.0;
        if cells(height, middle) * cells(width, middle) <= MAX_PATCHES {
            low = middle;
        } else {
            high = middle;
        }
    }
    (low, cells(height, low), cells(width, low))
}

/// Resized side inside the grid canvas: `max(1, min(cells * 16, round(size * scale)))`.
/// Python's `round` is round-half-to-even.
fn resized_extent(size: u32, scale: f64, cells: i64) -> u32 {
    (f64::from(size) * scale)
        .round_ties_even()
        .clamp(1.0, (cells * PATCH_SIZE) as f64) as u32
}

/// `np.median(border).astype(uint8)` over the first and last row and column,
/// corners counted twice. NumPy averages the two middle values of an even
/// count in floating point and `astype` truncates.
fn border_median(image: &GrayImage) -> u8 {
    let (width, height) = image.dimensions();
    let mut border = Vec::with_capacity(2 * (width + height) as usize);
    border.extend((0..width).map(|x| image.get_pixel(x, 0)[0]));
    border.extend((0..width).map(|x| image.get_pixel(x, height - 1)[0]));
    border.extend((0..height).map(|y| image.get_pixel(0, y)[0]));
    border.extend((0..height).map(|y| image.get_pixel(width - 1, y)[0]));
    border.sort_unstable();
    let middle = border.len() / 2;
    if border.len() % 2 == 1 {
        border[middle]
    } else {
        ((f64::from(border[middle - 1]) + f64::from(border[middle])) / 2.0) as u8
    }
}

/// The resized crop at the top left of a whole-patch canvas filled with its
/// border median.
#[derive(Debug)]
struct Canvas {
    image: GrayImage,
    grid: [i64; 2],
}

fn canvas(grey: &GrayImage) -> Result<Canvas> {
    let (scale, grid_h, grid_w) = grid(grey.height(), grey.width());
    let width = resized_extent(grey.width(), scale, grid_w);
    let height = resized_extent(grey.height(), scale, grid_h);
    let small = resize_bilinear(grey, width, height)?;
    let mut image = GrayImage::from_pixel(
        (grid_w * PATCH_SIZE) as u32,
        (grid_h * PATCH_SIZE) as u32,
        Luma([border_median(&small)]),
    );
    image::imageops::replace(&mut image, &small, 0, 0);
    Ok(Canvas {
        image,
        grid: [grid_h, grid_w],
    })
}

/// Upstream resizes with Pillow `BILINEAR`; `fast_image_resize` uses the same
/// triangle convolution but can differ at coefficient rounding.
fn resize_bilinear(image: &GrayImage, width: u32, height: u32) -> Result<GrayImage> {
    if image.dimensions() == (width, height) {
        return Ok(image.clone());
    }
    let mut resized = GrayImage::new(width, height);
    Resizer::new()
        .resize(
            image,
            &mut resized,
            &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
        )
        .map_err(|error| anyhow::anyhow!("failed to resize Manga OCR CPT-flex input: {error}"))?;
    Ok(resized)
}

#[derive(Debug, Deserialize)]
struct VocabularyFile {
    character_count: usize,
    characters: Vec<String>,
}

#[derive(Debug)]
pub(super) struct Vocabulary {
    characters: Vec<String>,
}

impl Vocabulary {
    pub(super) fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let file: VocabularyFile = serde_json::from_str(&fs::read_to_string(path)?)?;
        ensure!(
            file.characters.len() == file.character_count,
            "vocabulary lists {} characters but declares {}",
            file.characters.len(),
            file.character_count
        );
        Ok(Self {
            characters: file.characters,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.characters.len()
    }

    /// Upstream `_text` without the end cut, which generation already makes:
    /// ids below `SPECIAL_TOKENS` (pad, unk, start, end, mask) are dropped and
    /// character `i` is id `SPECIAL_TOKENS + i`.
    pub(super) fn decode(&self, ids: &[i64]) -> String {
        ids.iter()
            .filter_map(|&id| usize::try_from(id - SPECIAL_TOKENS).ok())
            .filter_map(|index| self.characters.get(index))
            .map(String::as_str)
            .collect()
    }
}
```

- [ ] **Step 5: 통과를 확인한다**

Run: `cargo test -p koharu-ml --lib manga_ocr_cpt::processor`
Expected: 6개 통과. `the_fixture_canvas_matches_upstream`이 합계 오차로 실패하면 출력된 합계를 기록하고, 격자·크기·채움값이 맞는지 먼저 본다. 합계 차이가 허용치를 넘으면 리사이즈 단계를 Pillow와 행 단위로 비교한다. 허용치를 늘리지는 않는다.

- [ ] **Step 6: 커밋한다**

```bash
cargo fmt -p koharu-ml
git add crates/koharu-ml/src/manga_ocr_cpt
git commit -m "feat(manga-ocr-cpt): port the aspect-preserving preprocessing and vocabulary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 3: 네트워크

**Files:**
- Create: `crates/koharu-ml/src/manga_ocr_cpt/model.rs`
- Modify: `crates/koharu-ml/src/manga_ocr_cpt/mod.rs`

- [ ] **Step 1: 모듈을 선언한다**

```rust
mod config;
mod model;
mod processor;
```

- [ ] **Step 2: 실패하는 모델 트리 테스트를 담은 `model.rs`를 쓴다**

```rust
//! Inference-only Manga OCR CPT-flex network.
//!
//! Port of `MangaOcrNet`: a ViT-B/16 encoder and a 2-layer post-LN BERT
//! decoder with cross-attention whose output layer is tied to the input
//! embedding. Parameter paths follow the checkpoint (`enc.*`, `dec.*`).
//! https://github.com/muscgab/manga-ocr-CPT/blob/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt/model.py

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use koharu_torch::Device;

    use super::Model;

    #[tokio::test]
    #[ignore = "requires the dynamically loaded LibTorch runtime"]
    async fn model_tree_matches_checkpoint_tensor_shapes() -> Result<()> {
        crate::init().await?;
        let model = Model::new(Device::Cpu);
        let variables = model.vs.variables();
        // The pinned checkpoint holds 260 tensors.
        assert_eq!(variables.len(), 260);
        for (name, shape) in [
            ("enc.patch.weight", vec![768, 768]),
            ("enc.cls", vec![768]),
            ("enc.pos", vec![197, 768]),
            ("enc.11.f1.weight", vec![3072, 768]),
            ("enc.ln.bias", vec![768]),
            ("dec.word", vec![6822, 768]),
            ("dec.pos", vec![512, 768]),
            ("dec.type0", vec![768]),
            ("dec.1.ck.weight", vec![768, 768]),
            ("dec.1.ln3.weight", vec![768]),
            ("dec.head.weight", vec![768, 768]),
            ("dec.head_ln.bias", vec![768]),
            ("dec.out_bias", vec![6822]),
        ] {
            assert_eq!(variables[name].size(), shape, "{name}");
        }
        Ok(())
    }
}
```

- [ ] **Step 3: 실패를 확인한다**

Run: `cargo test -p koharu-ml --lib -- --ignored model_tree_matches_checkpoint_tensor_shapes`
Expected: 컴파일 실패 — `unresolved import super::Model`. 같은 이름의 테스트가 다른 모델에도 있지만 컴파일 단계에서 멈춘다.

- [ ] **Step 4: 네트워크를 구현한다**

`model.rs`의 모듈 doc 주석 아래, `#[cfg(test)]` 위에 넣는다.

```rust
use std::path::Path;

use anyhow::Result;
use koharu_torch::{
    Device, Kind, Tensor,
    nn::{self, Module},
};

use super::{
    config::{
        DECODER_LAYERS, DECODER_POSITIONS, ENCODER_LAYERS, END_TOKEN, HEAD_DIM, HIDDEN_SIZE,
        INTERMEDIATE_SIZE, LAYER_NORM_EPS, MAX_NEW_TOKENS, MAX_PATCHES, NUM_HEADS, OUTPUT_TOKENS,
        PATCH_FEATURES, POSITION_GRID, START_TOKEN,
    },
    processor::ImageInput,
};

#[derive(Debug)]
pub(super) struct Model {
    vs: nn::VarStore,
    encoder: Encoder,
    decoder: Decoder,
}

impl Model {
    pub(super) fn new(device: Device) -> Self {
        let mut vs = nn::VarStore::new(device);
        crate::backend::set_precision(&mut vs);
        let root = vs.root();
        let encoder = Encoder::new(&(&root / "enc"));
        let decoder = Decoder::new(&(&root / "dec"));
        vs.freeze();
        Self {
            vs,
            encoder,
            decoder,
        }
    }

    pub(super) fn load(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.vs.load(path)?;
        // Deliberate divergence: upstream keeps the decoder in fp32 on CUDA
        // and only runs encoder matmuls in fp16. Koharu applies its shared
        // precision policy to the whole store; the equivalence gate runs on
        // the CPU in fp32.
        crate::backend::set_precision(&mut self.vs);
        Ok(())
    }

    /// Upstream `greedy` for one crop. Returns the ids before the end token.
    pub(super) fn generate(&self, input: &ImageInput) -> Result<Vec<i64>> {
        let kind = self.vs.kind();
        let device = input.patches.device();
        let (memory, memory_mask) =
            self.encoder
                .forward(&input.patches.to_kind(kind), &input.valid, input.grid);
        let cross = self.decoder.cross_key_values(&memory);
        let mut cache: Vec<Option<(Tensor, Tensor)>> = (0..DECODER_LAYERS).map(|_| None).collect();
        let mut token = START_TOKEN;
        let mut tokens = Vec::new();
        for position in 0..MAX_NEW_TOKENS {
            token = self
                .decoder
                .step(token, position, &mut cache, &cross, &memory_mask, device);
            if token == END_TOKEN {
                break;
            }
            tokens.push(token);
        }
        Ok(tokens)
    }
}

fn layer_norm(path: nn::Path<'_>) -> nn::LayerNorm {
    nn::layer_norm(
        path,
        vec![HIDDEN_SIZE],
        nn::LayerNormConfig {
            eps: LAYER_NORM_EPS,
            ..Default::default()
        },
    )
}

fn linear(path: nn::Path<'_>, input: i64, output: i64) -> nn::Linear {
    nn::linear(path, input, output, Default::default())
}

/// `(batch, sequence, hidden)` -> `(batch, heads, sequence, head_dim)`.
fn heads(x: &Tensor) -> Tensor {
    let size = x.size();
    x.view([size[0], size[1], NUM_HEADS, HEAD_DIM]).transpose(1, 2)
}

fn attend(query: &Tensor, key: &Tensor, value: &Tensor, mask: Option<&Tensor>) -> Tensor {
    let attended = Tensor::scaled_dot_product_attention(
        query,
        key,
        value,
        mask,
        0.0,
        false,
        (HEAD_DIM as f64).powf(-0.5),
        false,
    );
    let size = attended.size();
    attended
        .transpose(1, 2)
        .contiguous()
        .reshape([size[0], size[2], HIDDEN_SIZE])
}

/// Additive SDPA bias over keys: `false` entries become forbidden.
fn additive_mask(keys: &Tensor, kind: Kind) -> Tensor {
    (keys.logical_not().to_kind(Kind::Float) * f32::MIN as f64)
        .to_kind(kind)
        .view([1, 1, 1, -1])
}

#[derive(Debug)]
struct Encoder {
    patch: nn::Linear,
    cls: Tensor,
    pos: Tensor,
    layers: Vec<EncoderLayer>,
    norm: nn::LayerNorm,
}

impl Encoder {
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            patch: linear(path / "patch", PATCH_FEATURES, HIDDEN_SIZE),
            cls: path.var("cls", &[HIDDEN_SIZE], nn::Init::Const(0.0)),
            pos: path.var("pos", &[MAX_PATCHES + 1, HIDDEN_SIZE], nn::Init::Const(0.0)),
            layers: (0..ENCODER_LAYERS)
                .map(|index| EncoderLayer::new(&(path / index)))
                .collect(),
            norm: layer_norm(path / "ln"),
        }
    }

    /// Upstream `pe`: the 14x14 learned table bicubically resized to the patch
    /// grid, zero on padded patches.
    fn position_embeddings(&self, [grid_h, grid_w]: [i64; 2]) -> Tensor {
        let device = self.pos.device();
        let table = self
            .pos
            .narrow(0, 1, MAX_PATCHES)
            .to_kind(Kind::Float)
            .reshape([POSITION_GRID, POSITION_GRID, HIDDEN_SIZE])
            .permute([2, 0, 1])
            .unsqueeze(0);
        let table = if [grid_h, grid_w] == [POSITION_GRID, POSITION_GRID] {
            table
        } else {
            table.upsample_bicubic2d([grid_h, grid_w], false, None::<f64>, None::<f64>)
        };
        let rows = table.squeeze_dim(0).flatten(1, 2).transpose(0, 1);
        let padding = Tensor::zeros(
            [MAX_PATCHES - grid_h * grid_w, HIDDEN_SIZE],
            (Kind::Float, device),
        );
        Tensor::cat(&[rows, padding], 0).to_kind(self.pos.kind())
    }

    /// Returns the memory `(1, 197, hidden)` and its additive key mask.
    fn forward(&self, patches: &Tensor, valid: &Tensor, grid: [i64; 2]) -> (Tensor, Tensor) {
        let device = patches.device();
        let patches =
            self.patch.forward(patches) + self.position_embeddings(grid).unsqueeze(0);
        let cls = (&self.cls + self.pos.select(0, 0)).view([1, 1, HIDDEN_SIZE]);
        let mut hidden = Tensor::cat(&[cls, patches], 1);
        let keys = Tensor::cat(
            &[Tensor::ones([1, 1], (Kind::Bool, device)), valid.shallow_clone()],
            1,
        );
        let mask = additive_mask(&keys, hidden.kind());
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &mask);
        }
        (self.norm.forward(&hidden), mask)
    }
}

/// Pre-LN ViT layer.
#[derive(Debug)]
struct EncoderLayer {
    ln1: nn::LayerNorm,
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
    o: nn::Linear,
    ln2: nn::LayerNorm,
    f1: nn::Linear,
    f2: nn::Linear,
}

impl EncoderLayer {
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            ln1: layer_norm(path / "ln1"),
            q: linear(path / "q", HIDDEN_SIZE, HIDDEN_SIZE),
            k: linear(path / "k", HIDDEN_SIZE, HIDDEN_SIZE),
            v: linear(path / "v", HIDDEN_SIZE, HIDDEN_SIZE),
            o: linear(path / "o", HIDDEN_SIZE, HIDDEN_SIZE),
            ln2: layer_norm(path / "ln2"),
            f1: linear(path / "f1", HIDDEN_SIZE, INTERMEDIATE_SIZE),
            f2: linear(path / "f2", INTERMEDIATE_SIZE, HIDDEN_SIZE),
        }
    }

    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Tensor {
        let normed = self.ln1.forward(hidden);
        let attended = attend(
            &heads(&self.q.forward(&normed)),
            &heads(&self.k.forward(&normed)),
            &heads(&self.v.forward(&normed)),
            Some(mask),
        );
        let hidden = hidden + self.o.forward(&attended);
        // Upstream `F.gelu` is the exact erf form.
        let ffn = self
            .f2
            .forward(&self.f1.forward(&self.ln2.forward(&hidden)).gelu("none"));
        &hidden + ffn
    }
}

#[derive(Debug)]
struct Decoder {
    word: Tensor,
    pos: Tensor,
    type0: Tensor,
    norm: nn::LayerNorm,
    layers: Vec<DecoderLayer>,
    head: nn::Linear,
    head_norm: nn::LayerNorm,
    out_bias: Tensor,
}

impl Decoder {
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            word: path.var("word", &[OUTPUT_TOKENS, HIDDEN_SIZE], nn::Init::Const(0.0)),
            pos: path.var("pos", &[DECODER_POSITIONS, HIDDEN_SIZE], nn::Init::Const(0.0)),
            type0: path.var("type0", &[HIDDEN_SIZE], nn::Init::Const(0.0)),
            norm: layer_norm(path / "ln"),
            layers: (0..DECODER_LAYERS)
                .map(|index| DecoderLayer::new(&(path / index)))
                .collect(),
            head: linear(path / "head", HIDDEN_SIZE, HIDDEN_SIZE),
            head_norm: layer_norm(path / "head_ln"),
            out_bias: path.var("out_bias", &[OUTPUT_TOKENS], nn::Init::Const(0.0)),
        }
    }

    /// Upstream `cross_kv`: each layer's cross-attention keys and values,
    /// computed once per crop.
    fn cross_key_values(&self, memory: &Tensor) -> Vec<(Tensor, Tensor)> {
        self.layers
            .iter()
            .map(|layer| {
                (
                    heads(&layer.ck.forward(memory)),
                    heads(&layer.cv.forward(memory)),
                )
            })
            .collect()
    }

    /// Upstream `step`: feeds `token` at `position` and returns the argmax id.
    fn step(
        &self,
        token: i64,
        position: i64,
        cache: &mut [Option<(Tensor, Tensor)>],
        cross: &[(Tensor, Tensor)],
        memory_mask: &Tensor,
        device: Device,
    ) -> i64 {
        let id = Tensor::from_slice(&[token]).to_device(device);
        let embedding =
            self.word.index_select(0, &id) + self.pos.select(0, position) + &self.type0;
        let mut x = self.norm.forward(&embedding.unsqueeze(0));
        for ((layer, cache), (key, value)) in self.layers.iter().zip(cache.iter_mut()).zip(cross) {
            x = layer.forward(&x, cache, key, value, memory_mask);
        }
        let z = self.head_norm.forward(&self.head.forward(&x).gelu("none"));
        // The output layer is tied to the input embedding.
        let logits = z.matmul(&self.word.transpose(0, 1)) + &self.out_bias;
        logits.argmax(-1, false).int64_value(&[0, 0])
    }
}

/// Post-LN BERT layer with cross-attention.
#[derive(Debug)]
struct DecoderLayer {
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
    o: nn::Linear,
    ln1: nn::LayerNorm,
    cq: nn::Linear,
    ck: nn::Linear,
    cv: nn::Linear,
    co: nn::Linear,
    ln2: nn::LayerNorm,
    f1: nn::Linear,
    f2: nn::Linear,
    ln3: nn::LayerNorm,
}

impl DecoderLayer {
    fn new(path: &nn::Path<'_>) -> Self {
        Self {
            q: linear(path / "q", HIDDEN_SIZE, HIDDEN_SIZE),
            k: linear(path / "k", HIDDEN_SIZE, HIDDEN_SIZE),
            v: linear(path / "v", HIDDEN_SIZE, HIDDEN_SIZE),
            o: linear(path / "o", HIDDEN_SIZE, HIDDEN_SIZE),
            ln1: layer_norm(path / "ln1"),
            cq: linear(path / "cq", HIDDEN_SIZE, HIDDEN_SIZE),
            ck: linear(path / "ck", HIDDEN_SIZE, HIDDEN_SIZE),
            cv: linear(path / "cv", HIDDEN_SIZE, HIDDEN_SIZE),
            co: linear(path / "co", HIDDEN_SIZE, HIDDEN_SIZE),
            ln2: layer_norm(path / "ln2"),
            f1: linear(path / "f1", HIDDEN_SIZE, INTERMEDIATE_SIZE),
            f2: linear(path / "f2", INTERMEDIATE_SIZE, HIDDEN_SIZE),
            ln3: layer_norm(path / "ln3"),
        }
    }

    /// One token. The self-attention cache holds only past and current
    /// positions, so it needs no causal mask.
    fn forward(
        &self,
        x: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
        cross_key: &Tensor,
        cross_value: &Tensor,
        memory_mask: &Tensor,
    ) -> Tensor {
        let key = heads(&self.k.forward(x));
        let value = heads(&self.v.forward(x));
        let (key, value) = match cache.take() {
            Some((past_key, past_value)) => (
                Tensor::cat(&[past_key, key], 2),
                Tensor::cat(&[past_value, value], 2),
            ),
            None => (key, value),
        };
        let attended = attend(&heads(&self.q.forward(x)), &key, &value, None);
        *cache = Some((key, value));
        let x = self.ln1.forward(&(self.o.forward(&attended) + x));
        let attended = attend(
            &heads(&self.cq.forward(&x)),
            cross_key,
            cross_value,
            Some(memory_mask),
        );
        let x = self.ln2.forward(&(self.co.forward(&attended) + &x));
        self.ln3
            .forward(&(self.f2.forward(&self.f1.forward(&x).gelu("none")) + &x))
    }
}
```

- [ ] **Step 5: 모델 트리 테스트를 통과시킨다(G3)**

Run: `cargo test -p koharu-ml --lib -- --ignored manga_ocr_cpt::model::tests::model_tree_matches_checkpoint_tensor_shapes`
Expected: 통과. 변수 수가 260이 아니면 경로 이름을 체크포인트 헤더와 대조한다. 로더는 고치지 않는다.

- [ ] **Step 6: 커밋한다**

```bash
cargo fmt -p koharu-ml
git add crates/koharu-ml/src/manga_ocr_cpt
git commit -m "feat(manga-ocr-cpt): port the CPT-flex encoder-decoder network

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 4: 공개 수명주기와 체크포인트 동등성

**Files:**
- Modify: `crates/koharu-ml/src/manga_ocr_cpt/mod.rs`

- [ ] **Step 1: 실패하는 체크포인트 테스트를 추가한다**

`mod.rs` 맨 끝에 추가한다.

```rust
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::MangaOcrCptFlex;

    #[tokio::test]
    #[ignore = "downloads the checkpoint and requires the LibTorch runtime"]
    async fn checkpoint_matches_upstream_character_sequence() -> anyhow::Result<()> {
        crate::init().await?;
        let model = MangaOcrCptFlex::load(crate::Device::cpu()).await?;
        let input =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/fixtures/ocr/title.png");
        let text = model.inference(&image::open(input)?)?;
        // Upstream plain path (CPU, fp32) on the identical crop produced ids
        // [1767, 4323, 1601, 1238, 653, 377, 363, 5] before the end token.
        assert_eq!(text, "対策委員会です!");
        Ok(())
    }
}
```

- [ ] **Step 2: 실패를 확인한다**

Run: `cargo test -p koharu-ml --lib -- --ignored manga_ocr_cpt::tests::checkpoint_matches_upstream_character_sequence`
Expected: 컴파일 실패 — `unresolved import super::MangaOcrCptFlex`

- [ ] **Step 3: 공개 타입을 구현한다**

`mod.rs`의 모듈 선언 아래, `#[cfg(test)]` 위에 넣는다.

```rust
use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use koharu_torch::Device;

use crate::backend::TryIntoDevice;

use self::{
    config::{CHARACTERS, ModelConfig},
    model::Model,
    processor::{ImageProcessor, Vocabulary},
};

crate::model_repository!("muscgab/manga-ocr-base-CPT-flex" @ "9cc15742f8dc37cb5222a6569839bc774dfc0392" {
    CONFIG = "config.json",
    WEIGHTS = "model.safetensors",
    VOCABULARY = "nar_vocabulary_6817.json",
});

/// Japanese manga text-box OCR: mangaOCR's ViT-B/16 encoder and 2-layer BERT
/// decoder after continued pre-training, with an aspect-preserving input of at
/// most 196 patches and a 6,817-character vocabulary.
///
/// The weights are CC BY-NC-SA 4.0 (non-commercial). Koharu downloads them on
/// first use and does not redistribute them.
#[derive(Debug)]
pub struct MangaOcrCptFlex {
    model: Model,
    processor: ImageProcessor,
    vocabulary: Vocabulary,
}

impl MangaOcrCptFlex {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let (config_path, weights_path, vocabulary_path) = tokio::try_join!(
            async {
                CONFIG
                    .resolve()
                    .await
                    .context("failed to resolve Manga OCR CPT-flex config")
            },
            async {
                WEIGHTS
                    .resolve()
                    .await
                    .context("failed to resolve Manga OCR CPT-flex weights")
            },
            async {
                VOCABULARY
                    .resolve()
                    .await
                    .context("failed to resolve Manga OCR CPT-flex vocabulary")
            },
        )?;

        ModelConfig::from_file(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?
            .validate()?;
        let vocabulary = Vocabulary::from_file(&vocabulary_path)
            .with_context(|| format!("failed to read {}", vocabulary_path.display()))?;
        ensure!(
            vocabulary.len() as i64 == CHARACTERS,
            "Manga OCR CPT-flex vocabulary has {} characters but the decoder expects {CHARACTERS}",
            vocabulary.len()
        );

        let mut model = Model::new(device);
        model
            .load(&weights_path)
            .with_context(|| format!("failed to load {}", weights_path.display()))?;

        Ok(Self {
            model,
            processor: ImageProcessor::new(device),
            vocabulary,
        })
    }

    pub fn inference(&self, image: &DynamicImage) -> Result<String> {
        koharu_torch::no_grad(|| {
            let input = self.processor.preprocess(image)?;
            let ids = self.model.generate(&input)?;
            Ok(self.vocabulary.decode(&ids))
        })
    }
}
```

- [ ] **Step 4: 체크포인트 동등성을 통과시킨다(G4)**

Run: `cargo test -p koharu-ml --lib -- --ignored manga_ocr_cpt::tests::checkpoint_matches_upstream_character_sequence`
Expected: 통과. 실패하면 이 순서로 본다.
1. 전처리: Task 2의 캔버스 테스트가 통과하는지 확인한다.
2. 위치 임베딩: `(6, 32)` 격자에서 bicubic 리사이즈 결과의 shape `(192, 768)`와 패딩 4행이 0인지 확인한다.
3. 마스크: 인코더와 cross-attention 모두 키 197개 중 cls + 유효 192개만 허용하는지 확인한다.
4. 디코더: post-LN 순서(`ln1(o(y) + x)`)와 출력층 가중치 공유(`dec.word`)를 업스트림 `step`과 대조한다.

- [ ] **Step 5: 단위 테스트 전체(G1)를 한 번 돌린다**

Run: `cargo test -p koharu-ml --lib manga_ocr_cpt`
Expected: config 2개, processor 6개 통과. ignored 2개 건너뜀. 경고 없음.

- [ ] **Step 6: 커밋한다**

```bash
cargo fmt -p koharu-ml
git add crates/koharu-ml/src/manga_ocr_cpt/mod.rs
git commit -m "feat(manga-ocr-cpt): load the pinned CPT-flex checkpoint and match upstream

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 5: 파이프라인 등록

**Files:**
- Modify: `crates/koharu-pipeline/src/config.rs`
- Modify: `crates/koharu-pipeline/src/stages/ocr.rs`
- Modify: `crates/koharu-pipeline/src/bin/run.rs`

- [ ] **Step 1: `OcrModel`에 변형을 추가한다**

`config.rs`의 enum에서

```rust
    #[serde(rename = "manga-ocr")]
    MangaOcr,
```

바로 아래에 추가한다.

```rust
    #[serde(rename = "manga-ocr-cpt-flex")]
    MangaOcrCptFlex,
```

`Serialize` 매치의 `OcrModel::MangaOcr => "manga-ocr",` 아래에 추가한다.

```rust
            OcrModel::MangaOcrCptFlex => "manga-ocr-cpt-flex",
```

역직렬화 매치의 `"manga-ocr" => OcrModel::MangaOcr,` 아래에 추가한다.

```rust
            "manga-ocr-cpt-flex" => OcrModel::MangaOcrCptFlex,
```

- [ ] **Step 2: OCR stage에 분기를 추가한다**

`ocr.rs`의 import에서 `manga_ocr::MangaOcr,` 아래에 `manga_ocr_cpt::MangaOcrCptFlex,`를 추가한다.

`model()`의 `OcrModel::MangaOcr => "manga-ocr",` 아래에 추가한다.

```rust
            OcrModel::MangaOcrCptFlex => "manga-ocr-cpt-flex",
```

`enum Model`의 `Manga(Arc<Mutex<MangaOcr>>),` 아래에 추가한다.

```rust
    MangaCptFlex(Arc<Mutex<MangaOcrCptFlex>>),
```

`Model::load`의 `OcrModel::MangaOcr` 분기 아래에 추가한다.

```rust
            OcrModel::MangaOcrCptFlex => Ok(Self::MangaCptFlex(Arc::new(Mutex::new(
                MangaOcrCptFlex::load(device).await?,
            )))),
```

`run`의 이름 매치 `Self::Manga(_) => "manga-ocr",` 아래에 추가한다.

```rust
            Self::MangaCptFlex(_) => "manga-ocr-cpt-flex",
```

`run`의 결과 매치 `Self::Manga(model) => { ... }` 분기 아래에 추가한다.

```rust
            Self::MangaCptFlex(model) => {
                infer_text(model.clone(), targets, |model, image| {
                    model.inference(image)
                })
                .await?
            }
```

- [ ] **Step 3: CLI 선택지를 추가한다**

`run.rs`의 `OcrChoice`에서 `MangaOcr,` 아래에 추가한다.

```rust
    #[value(name = "manga-ocr-cpt-flex")]
    MangaOcrCptFlex,
```

`pipeline_config`의 `OcrChoice::MangaOcr => OcrModel::MangaOcr,` 아래에 추가한다.

```rust
                OcrChoice::MangaOcrCptFlex => OcrModel::MangaOcrCptFlex,
```

- [ ] **Step 4: 컴파일을 확인한다(G2)**

Run: `cargo check -p koharu-ml -p koharu-pipeline -p koharu-app --bins`
Expected: 성공, 경고 없음. 매치 누락은 여기서 `non-exhaustive patterns`로 드러난다.

- [ ] **Step 5: 커밋한다**

```bash
cargo fmt -p koharu-pipeline
git add crates/koharu-pipeline
git commit -m "feat(pipeline): offer Manga OCR CPT-flex as an OCR model

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 6: 프론트엔드와 문서

**Files:**
- Modify: `packages/bridge/src/protocol.ts`
- Modify: `packages/koharu/components/preferences/models.ts`
- Modify: `packages/koharu/components/preferences/PipelinePreferences.tsx`
- Modify: `packages/docs/models/vision.mdx`, `packages/docs/ja/models/vision.mdx`, `packages/docs/zh/models/vision.mdx`

- [ ] **Step 1: 프로토콜 유니온을 넓힌다**

`protocol.ts`의 `| { model: "manga-ocr" }` 바로 뒤에 `| { model: "manga-ocr-cpt-flex" }`를 넣는다.

```ts
} & PaddleOcrVl1_6MangaLoraConfig | { model: "manga-ocr" } | { model: "manga-ocr-cpt-flex" } | { model: "baberu-ocr" } | { model: "hayai-ocr" };
```

- [ ] **Step 2: 설정 화면 옵션을 추가한다**

`models.ts`:
- `modelOptions.ocr`에서 `'manga-ocr',` 아래에 `'manga-ocr-cpt-flex',`
- `modelNames`에서 `'manga-ocr': 'Manga OCR',` 아래에 `'manga-ocr-cpt-flex': 'Manga OCR CPT-flex',`
- `defaultModel`에서 `case 'manga-ocr':` 아래에 `case 'manga-ocr-cpt-flex':`

`PipelinePreferences.tsx`의 `case 'manga-ocr':` 아래에 `case 'manga-ocr-cpt-flex':`를 추가한다(설정 항목 없음, `return null`).

- [ ] **Step 3: 문서 표에 한 줄씩 넣는다**

`packages/docs/models/vision.mdx`의 OCR 표에서 `Manga OCR` 행 아래:

```markdown
| Manga OCR CPT-flex | Japanese manga text; weights are CC BY-NC-SA 4.0 (non-commercial) |
```

`packages/docs/ja/models/vision.mdx`:

```markdown
| Manga OCR CPT-flex | 日本語のマンガ文字。重みは CC BY-NC-SA 4.0(非商用) |
```

`packages/docs/zh/models/vision.mdx`:

```markdown
| Manga OCR CPT-flex | 日语漫画文字；权重为 CC BY-NC-SA 4.0（非商业） |
```

표 열 너비가 맞지 않으면 다음 Step의 포맷터가 정렬한다.

- [ ] **Step 4: 포맷하고 타입 검사를 돌린다(G5)**

```bash
bunx oxfmt packages/koharu/components/preferences/models.ts packages/koharu/components/preferences/PipelinePreferences.tsx packages/docs/models/vision.mdx packages/docs/ja/models/vision.mdx packages/docs/zh/models/vision.mdx
bunx tsc --noEmit -p packages/koharu/tsconfig.json
```

Expected: 타입 오류 없음. `modelNames`가 `Record<ModelName, string>`이므로 이름이 빠지면 여기서 드러난다.

- [ ] **Step 5: 커밋한다**

```bash
git add packages/bridge/src/protocol.ts packages/koharu/components/preferences packages/docs
git commit -m "feat(ui): list Manga OCR CPT-flex and document its license

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Task 7: 게이트 기록

**Files:**
- Modify: `specs/2026-10-17-manga-ocr-cpt-flex-design.md`

- [ ] **Step 1: 결과를 기록한다**

`## 검증` 표 아래에 측정값만 적는다.

```markdown
### 결과 (YYYY-MM-DD)

| # | 결과 | 비고 |
|---|---|---|
| G1 | 통과 / 실패 | 테스트 수, 캔버스 합계 |
| G2 | 통과 / 실패 | |
| G3 | 통과 / 실패 | 변수 260개 |
| G4 | 통과 / 실패 | 실제 출력 |
| G5 | 통과 / 실패 | |
```

- [ ] **Step 2: 커밋한다**

```bash
git add specs/2026-10-17-manga-ocr-cpt-flex-design.md
git commit -m "docs(specs): record the Manga OCR CPT-flex gate results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
