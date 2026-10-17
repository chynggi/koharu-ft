# Manga OCR CPT-flex 옵션 추가 설계

작성일: 2026-10-17

## 배경

Koharu의 `manga-ocr` 옵션은 `mayocream/manga-ocr` @ `4380edba990b959c508752350955350c1c80c31c`
체크포인트와 HF `VisionEncoderDecoder` 이름을 따르는 포트(`crates/koharu-ml/src/manga_ocr/`)를 쓴다.
`muscgab/manga-ocr-base-CPT-flex`는 같은 mangaOCR 아키텍처를 새 데이터로 계속 사전학습한 모델이다.
JMangaBench_Read v1.3(7,789 boxes)에서 원본 mangaOCR이 EM 78.25%(패키지 기본값)인 데 비해 91.42%를 낸다.

이 문서는 CPT-flex를 기존 `manga-ocr`과 **병존하는 새 OCR 옵션** `manga-ocr-cpt-flex`로 추가하는
목표 상태를 정의한다. 기존 `manga-ocr`은 바꾸지 않는다.

## 확인된 외부 사실

조사 시점: 2026-10-17.

- 가중치: `muscgab/manga-ocr-base-CPT-flex` @ `9cc15742f8dc37cb5222a6569839bc774dfc0392`.
  파일은 `config.json`, `model.safetensors`(F32, 110,935,974 파라미터, 444MB),
  `nar_vocabulary_6817.json`이다.
- 권위 구현: GitHub `muscgab/manga-ocr-CPT` @ `2f6ceefbb3f43a9591f05480b4b17cd54453cc81`.
  네트워크는 `manga_ocr_cpt/model.py`, 전처리는 `manga_ocr_cpt/preprocess.py`,
  어휘 디코딩은 `manga_ocr_cpt/__init__.py`(`_text`)에 있다. 이 저장소 코드는 Apache-2.0이다.
- **라이선스**: 가중치는 CC BY-NC-SA 4.0(비상업)이다. Koharu 코드는 MIT OR Apache-2.0이다.
  가중치를 저장소에 넣지 않고 첫 실행에서 내려받으므로 코드 라이선스는 바뀌지 않는다.
  사용자는 비상업 조건을 감수하기로 결정했다.
- **기존 포트로는 로드할 수 없다.**
  - 파라미터 이름이 HF 표준(`encoder.embeddings.patch_embeddings.projection.*`)이 아니라
    자체 이름(`enc.patch`, `enc.{i}.q`, `dec.{i}.ck`, `dec.out_bias` …)이다.
  - 패치 임베딩이 conv가 아니라 `[768, 768]` linear이고, ViT pooler가 없다.
  - 어휘는 `vocab.txt`가 아니라 JSON(`characters` 배열)이다. 출력 크기는 6,822(특수 5 + 문자 6,817)다.
  - `config.json`은 HF 스키마가 아니라 자체 스키마(`input.fit`, `input.max_patches` …)다.
- 체크포인트 텐서(층 인덱스는 `N`으로 접음):

  | 경로 | shape |
  |---|---|
  | `enc.patch.{weight,bias}` | `[768, 768]`, `[768]` |
  | `enc.cls` / `enc.pos` | `[768]` / `[197, 768]` |
  | `enc.N.{ln1,ln2}.{weight,bias}` | `[768]` |
  | `enc.N.{q,k,v,o}.{weight,bias}` | `[768, 768]`, `[768]` |
  | `enc.N.f1` / `enc.N.f2` | `[3072, 768]` / `[768, 3072]` (+bias) |
  | `enc.ln.{weight,bias}` | `[768]` |
  | `dec.word` / `dec.pos` / `dec.type0` | `[6822, 768]` / `[512, 768]` / `[768]` |
  | `dec.ln.{weight,bias}` | `[768]` |
  | `dec.N.{q,k,v,o,cq,ck,cv,co}.{weight,bias}` | `[768, 768]`, `[768]` |
  | `dec.N.f1` / `dec.N.f2` | `[3072, 768]` / `[768, 3072]` (+bias) |
  | `dec.N.{ln1,ln2,ln3}.{weight,bias}` | `[768]` |
  | `dec.head.{weight,bias}` / `dec.head_ln.{weight,bias}` | `[768, 768]` / `[768]` |
  | `dec.out_bias` | `[6822]` |

  인코더는 12층(`enc.0`–`enc.11`), 디코더는 2층(`dec.0`–`dec.1`)이다.

## 결정 사항

| # | 결정 | 근거 |
|---|---|---|
| D1 | `koharu-ml`에 **독립 모듈 `manga_ocr_cpt`**을 만든다. 기존 `manga_ocr`과 코드를 공유하지 않는다 | 이름·패치 임베딩·전처리·어휘가 모두 다르다. 공유하려면 키 변환 계층과 변형 분기가 필요한데, 이는 AGENTS.md의 "로더를 고치기 전에 이름 불일치를 의심하라"와 "호환 계층 금지"에 어긋난다 |
| D2 | 옵션 id는 `manga-ocr-cpt-flex`, Rust 이름은 `OcrModel::MangaOcrCptFlex`/`MangaOcrCptFlex`, 표시 이름은 `Manga OCR CPT-flex`로 한다 | 기존 `manga-ocr`과 구별되고 업스트림 모델 이름을 그대로 드러낸다 |
| D3 | 파라미터 경로는 체크포인트 이름(`enc.*`, `dec.*`)을 그대로 쓴다 | AGENTS.md: checkpoint-affecting names 보존 |
| D4 | `config.json`은 **검증에만** 쓴다. 아키텍처 상수는 `config.rs`에 고정한다 | 업스트림 `model.py`도 상수(`D, HEADS, EPS, MAX_NEW`)를 하드코딩한다. 설정 파일은 `fit`·`vocab`·`max_patches`·`patch`·`mean`·`std`·특수 토큰이 기대값과 같은지 확인하는 용도다 |
| D5 | 흑백 변환은 Pillow `convert("L")`의 정수 공식을 직접 구현한다 | `image` 크레이트의 luma는 Rec.709 계수를 써서 값이 다르다. 학습 입력과 맞추려면 Pillow 공식이어야 한다 |
| D6 | 정밀도는 기존 `crate::backend::set_precision` 경로를 따른다 | AGENTS.md: 확립된 VarStore 로딩 경로 사용. 업스트림 CUDA 경로는 인코더만 fp16이고 디코더는 fp32다. 로컬에서는 디코더도 반정밀도가 될 수 있다는 점을 의도적 발산으로 코드 옆에 남긴다. 동등성 게이트는 CPU fp32에서 본다 |
| D7 | 크롭은 한 장씩 처리한다. 업스트림의 배치·CUDA graph·GPU 전처리 경로는 옮기지 않는다 | Koharu OCR stage는 크롭을 하나씩 넘긴다. 업스트림 README도 두 경로의 출력이 같다고 밝힌다 |
| D8 | 기존 `manga-ocr` 옵션과 기본 OCR 모델은 바꾸지 않는다 | 사용자 결정(교체가 아닌 옵션 추가) |
| D9 | 문서의 모델 표에 비상업 라이선스를 명시한다. UI에는 별도 고지를 두지 않는다 | 라이선스 조건이 기존 모델들과 다른 유일한 OCR이다. 설정 화면의 표시 이름은 다른 모델과 같은 형식을 유지한다 |

## 백엔드 구성 — `crates/koharu-ml/src/manga_ocr_cpt/`

기존 모델 모듈과 같은 4파일 구조다.

| 파일 | 책임 |
|---|---|
| `mod.rs` | 공개 수명주기(`load`/`inference`), 체크포인트 고정, 어휘 크기 검증 |
| `config.rs` | 아키텍처·입력 상수, `config.json` 검증 |
| `model.rs` | 네트워크, 가중치 로딩, greedy 디코딩 |
| `processor.rs` | 흑백 변환, 종횡비 맞춤, 패치화, 어휘 디코딩 |

`crates/koharu-ml/src/lib.rs`에 `pub mod manga_ocr_cpt;` 한 줄이 늘어난다.

### `mod.rs`

```rust
crate::model_repository!("muscgab/manga-ocr-base-CPT-flex" @ "9cc15742f8dc37cb5222a6569839bc774dfc0392" {
    CONFIG = "config.json",
    WEIGHTS = "model.safetensors",
    VOCABULARY = "nar_vocabulary_6817.json",
});

pub struct MangaOcrCptFlex { model, processor, vocabulary, device }

impl MangaOcrCptFlex {
    pub async fn load(device: crate::Device) -> Result<Self>;
    pub fn inference(&self, image: &DynamicImage) -> Result<String>;
}
```

`load`는 세 파일을 병렬로 resolve하고, `config.json`을 검증하고,
`vocabulary.len() + SPECIAL_TOKENS == OUTPUT_TOKENS`(6,817 + 5 = 6,822)를 `ensure!`한 뒤 가중치를 로드한다.
`inference`는 `koharu_torch::no_grad` 안에서 전처리 → `generate` → 어휘 디코딩을 한다.

### `config.rs`

```rust
pub(super) const HIDDEN_SIZE: i64 = 768;
pub(super) const NUM_HEADS: i64 = 12;
pub(super) const HEAD_DIM: i64 = HIDDEN_SIZE / NUM_HEADS;
pub(super) const INTERMEDIATE_SIZE: i64 = 3072;
pub(super) const ENCODER_LAYERS: usize = 12;
pub(super) const DECODER_LAYERS: usize = 2;
pub(super) const LAYER_NORM_EPS: f64 = 1e-12;
pub(super) const PATCH_SIZE: i64 = 16;
/// Learned square position grid of the original 224x224 ViT.
pub(super) const POSITION_GRID: i64 = 14;
pub(super) const MAX_PATCHES: i64 = 196;
pub(super) const DECODER_POSITIONS: i64 = 512;
pub(super) const OUTPUT_TOKENS: i64 = 6822;
pub(super) const SPECIAL_TOKENS: i64 = 5;
pub(super) const START_TOKEN: i64 = 2;
pub(super) const END_TOKEN: i64 = 3;
/// At most 64 characters plus the end token.
pub(super) const MAX_NEW_TOKENS: i64 = 65;
```

`ModelConfig::from_file`은 `config.json`에서 `vocab`, `special_tokens`, `input.{grey, fit, max_patches,
patch, mean, std}`, `max_new_tokens`만 역직렬화하고, `validate`가 각 값이 위 상수와
`fit == "aspect"`, `grey == true`, `mean == std == 0.5`에 맞는지 확인한다.

### `model.rs`

업스트림 `MangaOcrNet`을 구조 그대로 옮긴다. 층 이름은 체크포인트 경로와 같다.

**인코더**(`encode`):

1. `x = enc.patch(P) + pe`. `P`는 `(1, 196, 768)`, `pe`는 유효 패치에만 위치 행이 있고 나머지는 0이다.
2. `h = cat([enc.cls + enc.pos[0], x], 1)` → `(1, 197, 768)`.
3. 키 마스크 `[true, V…]`(cls 포함 197)를 SDPA 가산 마스크로 만든다.
4. 12층 pre-LN: `h += o(attn(ln1(h)))`, `h += f2(gelu(f1(ln2(h))))`. GELU는 exact(`F.gelu` 기본)다.
5. `enc.ln(h)`와 마스크를 돌려준다.

**위치 임베딩**(`position_embeddings(gh, gw)`): `enc.pos[1:]`을 `(1, 768, 14, 14)`로 바꿔
`upsample_bicubic2d([gh, gw], align_corners=false)`로 맞춘다. `(14, 14)`이면 테이블을 그대로 쓴다.
결과를 `(gh*gw, 768)`로 펴서 196행 버퍼 앞에 놓고 나머지는 0으로 둔다.

**디코더 한 스텝**(`step`): post-LN BERT 2층이다.

1. `x = dec.ln(dec.word[tok] + dec.pos[t] + dec.type0)`.
2. 층마다:
   - self-attention: `q,k,v` → 캐시에 `k,v`를 이어 붙인다 → `x = ln1(o(y) + x)`.
   - cross-attention: `cq(x)`와 인코더 메모리에서 미리 계산한 `ck, cv`, 인코더 마스크 → `x = ln2(co(y) + x)`.
   - FFN: `x = ln3(f2(gelu(f1(x))) + x)`.
3. `z = dec.head_ln(gelu(dec.head(x)))`, `logits = z · dec.wordᵀ + dec.out_bias`(입력 임베딩과 가중치 공유).

**생성**(`generate`): `START_TOKEN`으로 시작해 최대 `MAX_NEW_TOKENS`번 argmax한다.
`END_TOKEN`이 나오면 멈추고 END는 결과에 넣지 않는다. cross-attention의 `ck, cv`는 층마다 한 번만 계산한다.
self-attention 캐시는 기존 포트들처럼 `Tensor::cat`으로 늘린다.
업스트림의 정적 캐시와 `live` 마스크는 CUDA graph용이므로 옮기지 않는다(D7). 출력은 같다.

### `processor.rs`

업스트림 `preprocess.py`의 `fit="aspect"` 경로다.

1. **흑백**: 픽셀마다 `L = (R*19595 + G*38470 + B*7471 + 0x8000) >> 16`(Pillow `convert("L")`, D5).
   이후 단일 채널 `GrayImage`로 다룬다. 업스트림은 `L → RGB`로 세 채널을 복제하지만 값이 같으므로
   패치화 단계에서 채널을 복제한다.
2. **격자**(`grid(h, w)`): `lo = 1e-4`, `hi = 1000.0`에서 60회 이진 탐색해
   `ceil(h*s/16) * ceil(w*s/16) <= 196`을 만족하는 최대 `s`를 찾는다. `(s, gh, gw)`를 돌려준다.
3. **리사이즈 크기**: `rw = max(1, min(gw*16, round(w*s)))`, `rh`도 같다.
   Python `round`는 짝수 반올림이므로 `f64::round_ties_even`을 쓴다.
4. **리사이즈**: `fast_image_resize`의 `Convolution(Bilinear)`. Pillow `BILINEAR`와 같은 삼각 필터 합성곱이다.
   계수 반올림에서 미세한 차이가 날 수 있다(기존 Hayai 포트와 같은 판단).
5. **캔버스**: `(gh*16, gw*16)` 캔버스를 테두리 중앙값으로 채우고 리사이즈 결과를 좌상단에 놓는다.
   테두리는 첫 행, 마지막 행, 첫 열, 마지막 열을 이어 붙인 것으로, 모서리가 두 번 들어간다.
   중앙값은 `np.median`처럼 짝수 개일 때 가운데 두 값의 평균이고, `astype(uint8)`처럼 버림한다.
6. **정규화·패치화**: `(v / 255 - 0.5) / 0.5`. 패치 특징 순서는 `(channel, row, column)`이다
   (`transpose(0, 2, 4, 1, 3)`). 유효 패치 `gh*gw`개 뒤를 0으로 채워 `(1, 196, 768)`을 만들고,
   유효 마스크 `V`와 `(gh, gw)`를 함께 돌려준다.
7. **어휘 디코딩**: `ids[t] >= 5`인 토큰만 `characters[t - 5]`로 잇는다. pad·unk·mask는 버린다.

## 소비자

| 소비자 | 변경 |
|---|---|
| `crates/koharu-ml/src/lib.rs` | `pub mod manga_ocr_cpt;` |
| `crates/koharu-pipeline/src/config.rs` | `OcrModel::MangaOcrCptFlex`(`#[serde(rename = "manga-ocr-cpt-flex")]`), `Serialize` 매치, 역직렬화 매치 |
| `crates/koharu-pipeline/src/stages/ocr.rs` | `Model::MangaCptFlex`, `load`·`run`·`model()` 분기 |
| `crates/koharu-pipeline/src/bin/run.rs` | `OcrChoice::MangaOcrCptFlex`(`manga-ocr-cpt-flex`)와 매핑 |
| `packages/bridge/src/protocol.ts` | `OcrModel` 유니온에 `{ model: "manga-ocr-cpt-flex" }`(손으로 쓰는 파일이다) |
| `packages/koharu/components/preferences/models.ts` | `modelOptions.ocr`, `modelNames`, `defaultModel` |
| `packages/koharu/components/preferences/PipelinePreferences.tsx` | 설정 항목 없음(`return null` 분기) |
| `packages/docs/{en,ja,zh}/models/vision.mdx` | 표에 한 줄, CC BY-NC-SA 4.0 명시(D9) |

기존 `normalize_ocr_text`(전부 대체 문자인 시퀀스를 `…`로 바꾸는 처리)는 모든 OCR 결과에 공통으로 적용되므로
그대로 둔다. OCR stage의 모델 상주 정책(`specs/2026-10-17-sequential-model-residency-design.md`)도 그대로 따른다.

## 오류 처리

| 상황 | 처리 |
|---|---|
| 빈 이미지 | `processor`가 `cannot recognize an empty image`로 실패 |
| `config.json`이 기대 입력 설정과 다름 | `validate`가 어떤 필드가 다른지 밝히며 실패 |
| 어휘 크기 불일치 | `load`가 `ensure!`로 실패 |
| 가중치 이름·shape 불일치 | 기존 VarStore 로더가 실패. 로더를 고치기 전에 경로·shape를 체크포인트와 대조한다 |
| 64자 초과 박스 | 업스트림과 같이 65토큰에서 잘린다 |
| 다운로드 실패 | 기존 `model_repository!` 경로 |

## 검증

| # | 게이트 | 명령 | 통과 조건 |
|---|---|---|---|
| G1 | 단위 테스트(장치 불필요) | `cargo test -p koharu-ml manga_ocr_cpt` | 아래 단위 테스트 전부 통과 |
| G2 | 컴파일 | `cargo check -p koharu-ml -p koharu-pipeline -p koharu-app --bins` | 성공 |
| G3 | 모델 트리 | `cargo test -p koharu-ml -- --ignored manga_ocr_cpt::model::tests::model_tree_matches_checkpoint_tensor_shapes` | 대표 경로의 shape가 체크포인트 표와 같다 |
| G4 | 체크포인트 동등성 | `cargo test -p koharu-ml -- --ignored manga_ocr_cpt::tests::checkpoint_matches_upstream_character_sequence` | `benches/fixtures/ocr/title.png`에서 업스트림 plain 경로(CPU fp32)와 같은 문자열 |
| G5 | 프론트엔드 타입 | `bunx tsc --noEmit -p packages/koharu/tsconfig.json` | 성공. `ModelName` 유니온이 늘어도 `modelNames`·`defaultModel`·`PipelinePreferences`의 망라 검사가 통과 |

**G4의 기대 문자열은 가정하지 않는다.** 구현 전에 업스트림 `MangaOcrCPT("flex", device="cpu",
backend="plain")`을 같은 픽스처에 한 번 돌려 얻은 문자열을 테스트에 넣는다. 실패하면 전처리
(격자, 리사이즈 크기, 중앙값, luma)부터 업스트림 `preprocess` 출력과 텐서 단위로 비교하고,
그다음 인코더 마스크와 위치 임베딩, 디코더 실행 순서를 본다.

## 테스트

| 계층 | 내용 | 위치 |
|---|---|---|
| 단위 | `grid`: 정사각(`1024×1024` → `14×14`), 가로로 긴 박스, 세로로 긴 박스에서 `gh*gw <= 196`이고 업스트림과 같은 격자 | `processor.rs` |
| 단위 | 리사이즈 크기의 짝수 반올림(`round_ties_even`)과 `[1, gw*16]` 클램프 | `processor.rs` |
| 단위 | 테두리 중앙값: 짝수 개 평균 후 버림, 모서리 중복 포함 | `processor.rs` |
| 단위 | Pillow luma: 흰색 255, 검정 0, 순수 빨강 76, 순수 초록 150, 순수 파랑 29 | `processor.rs` |
| 단위 | 어휘 디코딩: 특수 토큰 0–4를 건너뛰고 `t - 5`로 인덱싱 | `processor.rs` |
| 단위 | `config.json` 검증: 기대값 통과, `fit = "square"` 거부 | `config.rs` |
| 장치 | 모델 트리 shape(G3) | `model.rs` |
| 체크포인트 | 업스트림 문자열 동등성(G4) | `mod.rs` |
| 파이프라인 | `run.rs`의 `--ocr manga-ocr-cpt-flex` 선택, `config.rs`의 직렬화 왕복 | `koharu-pipeline` |

## 범위 밖

| 항목 | 사유 |
|---|---|
| `manga-ocr-base-CPT`(224 정사각 입력) | 업스트림이 대조군이라고 밝힌다. CPT-flex만 추가한다 |
| 기존 `manga-ocr` 교체 | 옵션 추가로 결정했다(D8) |
| 배치·CUDA graph·GPU 전처리 | 크롭 단위 처리다(D7) |
| `koharu-ml`의 bin·bench | 게이트 G3·G4로 충분하다. 필요해지면 따로 추가한다 |
| 업스트림 `accel/`(Hayai·PaddleOCR 가속) | 이 옵션과 무관하다 |
| UI 라이선스 고지 | 문서에만 명시한다(D9) |

## 참고

- `https://huggingface.co/muscgab/manga-ocr-base-CPT-flex` @ `9cc15742…`
- `https://github.com/muscgab/manga-ocr-CPT/tree/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/manga_ocr_cpt`
- `https://github.com/muscgab/manga-ocr-CPT/blob/2f6ceefbb3f43a9591f05480b4b17cd54453cc81/TECHNICAL_REPORT.md`
- `specs/2026-10-17-sequential-model-residency-design.md` — OCR stage 상주 정책
