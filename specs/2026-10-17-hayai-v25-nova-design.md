# Hayai OCR v2.5 Nova 지원 설계

작성일: 2026-10-17

## 배경

Koharu의 `hayai-ocr`는 현재 Hayai OCR v2 체크포인트를 쓴다
(`crates/koharu-ml/src/hayai_ocr/`, `JustANormalTinkerer/hayai-ocr-v2` @
`4a4ce477c9a8841f208b94e1d9ed5c0938965e05`). 그 뒤에 v2.5 Nova가 나왔고,
`ternurarl/koharu-docker` 포크가 이식 가능한 형태로 이미 구현해 두었다.

이 문서는 **같은 `hayai-ocr` 엔트리를 v2.5 Nova로 교체**하는 목표 상태를 정의한다.
변형을 병존시키지 않는다(AGENTS.md "Never add backward compatibility").

## 확인된 외부 사실

조사 시점: 2026-10-17.

- 체크포인트: `JustANormalTinkerer/hayai-ocr-v2.5-nova` @
  `2803d0dc005fa4396ab1e7c5a472d2c450a247f7`. `config.json`, `model.safetensors`,
  `tokenizer.json` 세 파일 구조는 v2와 같다.
- 참조 구현: `ternurarl/koharu-docker`의 `crates/koharu-ml/src/hayai_ocr/`
  (`mod.rs`, `config.rs`, `model.rs`, `processor.rs`). v2.5 Nova를 완전히 이식해 두었다.
  이 포크는 AGENTS.md가 요구하는 "commit-pinned authoritative implementation"의 자리가
  아니다. 권위는 모델 저장소의 `modeling_hayai.py`이고, 포크는 그 이식 결과를 비교하는
  대상으로 쓴다.
- v2 대비 체크포인트가 바뀌는 지점:
  - 투영기가 `proj.0`/`proj.2`(bias 있음)에서 `norm`/`mlp.0`/`mlp.2`(bias 없음)/`out_norm`으로
    바뀐다. 이른바 `DSCProjector`다.
  - `DSCProjector`는 2×2 픽셀 언셔플로 패치 4개를 디코더 토큰 1개로 합친다.
  - 그래서 2D 시각 RoPE가 **압축된 격자** 위에서 돈다. 시각 접두부는
    `ceil(H/2) × ceil(W/2)`개 위치를 갖고 **패딩 행이 없다**.
- v2와 같아서 바뀌지 않는 것: SigLIP2 NaFlex 비전 인코더 상수, 디코더 헤드 레이아웃
  (8 query / 2 kv / 64 head_dim), `attn_res_scale`·`ffn_res_scale`, 정적 KV 캐시,
  1D 텍스트 RoPE, `MAX_NEW_TOKENS = 256`, `REPETITION_PENALTY = 1.0`,
  `vocab_size = 16004`, `d_model = 512`, `d_vision = 768`, `d_ffn`, `n_layers = 12`.
  **이들은 Nova의 새 기능이 아니다.** 로컬 v2가 이미 갖고 있다.
- 새로 생기는 후처리: 출력을 NFKC 정규화하고 CJK 문자 사이 공백을 제거한다
  (`normalize_text`). 이것은 순수 추가가 아니라 **반환 문자열을 바꾸는 변경**이다.
- 텍스트 정규화에 NFKC가 필요하므로 `icu_normalizer`가 필요하다. 이미 워크스페이스
  의존성이다(`Cargo.toml`의 `icu_normalizer = { version = "2.3.0", … }`).

## 결정 사항

| # | 결정 | 근거 |
|---|---|---|
| D1 | `hayai-ocr` id와 `HayaiOcr` 타입을 그대로 두고 **체크포인트만 교체**한다. 별도 변형 엔트리·별칭·분기 플래그를 만들지 않는다 | 교체로 결정했다. 두 변형을 동시에 유지하면 로더에 죽은 분기가 생긴다. AGENTS.md의 변경 정책과 일치한다 |
| D2 | `MAX_NUM_PATCHES`를 256에서 **512**로 올린다 | CER 3.10% / EM 80.68%로 측정된 품질 우선 지점이다. 384는 CER 3.65%로 v2.1을 앞서지 못한다. 지연 비용 약 34%를 감수한다 |
| D3 | `Projector`를 제자리에서 **`DSCProjector`로 교체**한다. 새 타입을 추가하고 분기하지 않는다 | D1과 같은 이유다. `Decoder`가 소유한 필드 이름(`projector`)과 경로(`projector`)는 유지된다 |
| D4 | 시각 RoPE를 **압축 격자**에서 돌리고, 패딩 행 생성 경로(`padding_rows`, `identity_padding`, `build_angles`)를 **삭제**한다 | Nova의 시각 접두부에는 패딩 행이 없다. `m_vision == height_out * width_out`이므로 길이가 자동으로 맞는다. 남겨 두면 조용히 틀린 각도를 적용한다 |
| D5 | `normalize_text`와 `is_cjk`를 `processor.rs`에 두고 **`inference`가 호출**한다 | 참조와 같은 소유권이다. OCR 결과를 소비하는 곳이 `inference` 하나뿐이므로 다른 계층에 노출할 이유가 없다 |
| D6 | `icu_normalizer`를 `koharu-ml`의 의존성에 추가한다 | NFKC 정규화의 유일한 구현 수단이다. 워크스페이스에 이미 고정돼 있다 |
| D7 | `generate` 진입부에 패치 격자 검증(`patch_height > 0 && patch_width > 0 && h*w <= MAX_NUM_PATCHES`)을 **추가**한다 | `DSCProjector`가 홀수 축에서 마지막 행/열을 복제하므로 `patch_height - 1`을 인덱싱한다. 격자가 0이면 조용히 통과하지 않고 실패해야 한다 |
| D8 | CJK 공백 제거를 upstream의 단일 `re.sub` 패스와 **다르게** 구현한다(한 번의 스캔으로 모든 CJK 내부 공백 제거) | upstream의 비중첩 스캔은 `"漢 字 漢"`을 `"漢字 漢"`으로 남긴다. 이는 스캔의 부작용이지 의도가 아니고, 번역 입력으로는 공백이 없는 편이 맞다. 의도적 발산이므로 코드 옆에 근거를 남긴다 |

## 백엔드 구성 — `koharu-ml`

기존 4파일 구조를 유지한다. 새 모듈을 만들지 않는다.

| 파일 | 책임 | 변경 |
|---|---|---|
| `mod.rs` | 공개 수명주기(`load`/`inference`)·체크포인트 고정 | 리포 id·revision 교체, `normalize_text` 호출 1줄, 문서 주석에 Nova 설명 |
| `config.rs` | 상수·설정 검증 | `SPATIAL_DOWNSCALE`, `decoder_vision_grid` 추가, `MAX_NUM_PATCHES` 512 |
| `model.rs` | 네트워크·가중치 로딩·디코딩 | `Projector` 교체, `generate`의 RoPE 경로, 격자 검증, `pixel_unshuffle` 순서 테스트 |
| `processor.rs` | NaFlex 전처리·토크나이저·텍스트 정규화 | `normalize_text`·`is_cjk` 추가와 테스트, `icu_normalizer` import |

`Cargo.toml`에는 `icu_normalizer = { workspace = true }` 한 줄이 늘어난다.

### `config.rs`

```rust
/// Nova `DSCProjector`의 공간 다운샘플링: 2x2 픽셀 언셔플이 패치
/// `SPATIAL_DOWNSCALE^2`개를 디코더 토큰 1개로 합친다.
pub(super) const SPATIAL_DOWNSCALE: i64 = 2;

/// `patch_height x patch_width` 패치 격자가 언셔플 후 갖는 디코더 시각 격자:
/// `ceil(H / 2) x ceil(W / 2)`. 즉 시각 토큰은 `H_out * W_out`개다.
pub(super) fn decoder_vision_grid(patch_height: i64, patch_width: i64) -> (i64, i64) {
    (
        (patch_height + SPATIAL_DOWNSCALE - 1) / SPATIAL_DOWNSCALE,
        (patch_width + SPATIAL_DOWNSCALE - 1) / SPATIAL_DOWNSCALE,
    )
}
```

`MAX_NUM_PATCHES`의 문서 주석은 256/384/512 세 계단과 각 계단의 CER, 상대 지연을
표로 남기고, 512를 고른 이유(D2)를 적는다.

### `model.rs` — `Projector`

```rust
struct Projector {
    norm: nn::LayerNorm,
    mlp1: nn::Linear,
    mlp2: nn::Linear,
    out_norm: RmsNorm,
    d_vision: i64,
}
```

생성 시 `unshuffled_dim = d_vision * SPATIAL_DOWNSCALE * SPATIAL_DOWNSCALE`(= 3072)을 쓴다.
`nn::LinearConfig { bias: false, ..Default::default() }`로 `mlp/0`과 `mlp/2`를 만든다.
`out_norm`은 로컬 `RmsNorm`을 재사용한다.

`forward(&self, visual_features: &Tensor, patch_height: i64, patch_width: i64) ->
(Tensor, i64, i64)`:

1. `decoder_vision_grid`로 `(height_out, width_out)`를 구한다.
2. `visual_features.narrow(1, 0, patch_height * patch_width)`로 유효 패치만 자른다.
3. `reshape([1, patch_height, patch_width, c]).permute([0, 3, 1, 2])`.
4. 홀수 격자면 마지막 행/열을 복제해 모든 출력 셀이 완전한 블록을 보게 한다
   (`patch_height < height_out * scale`일 때).
5. `.contiguous().pixel_unshuffle(scale).permute([0, 2, 3, 1])
   .reshape([1, height_out * width_out, c * scale * scale])`.
6. `mlp1(norm(x)).gelu("none")` → `out_norm(mlp2(…))`.

`pixel_unshuffle`은 `koharu-torch`가 이미 노출한다
(`crates/koharu-torch/src/wrappers/tensor_generated.rs`). 채널 레이아웃은
`c * 4 + dy * 2 + dx`이며, 이는 upstream `gather_patches`의 전치와 같다.
따라서 체크포인트를 다시 배치할 필요가 없다.

### `model.rs` — `generate`

`projector.forward`가 `(vision_embeddings, patch_height_out, patch_width_out)`을 돌려주고,
그 뒤가 이렇게 바뀐다.

```rust
// RoPE는 *압축된* 격자에서 돈다. DSC 투영기가 2x2 패치 블록을 토큰 하나로
// 합쳤으므로 시각 접두부는 정확히 `patch_height_out * patch_width_out`개
// 위치를 갖고, 패딩 행이 없다.
let (visual_cos, visual_sin) = self.visual_rope(patch_height_out, patch_width_out);
let first_text = |angles: &Tensor| angles.narrow(0, 0, 1).unsqueeze(0);
let cos = Tensor::cat(&[visual_cos.unsqueeze(0), first_text(&self.text_rope_cos)], 1);
let sin = Tensor::cat(&[visual_sin.unsqueeze(0), first_text(&self.text_rope_sin)], 1);
```

`m_vision = hidden.size()[1] - 1`은 그대로 시각 접두부 길이이며, 이제
`patch_height_out * patch_width_out`과 같다. 블록 인과 프리필 마스크, KV 캐시 루프,
greedy 디코딩, 반복 패널티는 바뀌지 않는다.
`visual_rope`의 doc 주석은 "이미지의 *압축된* 패치 격자"로 고친다.

### `processor.rs` — `normalize_text`

```rust
pub(crate) fn normalize_text(text: &str) -> String
fn is_cjk(character: char) -> bool
```

`is_cjk`가 인정하는 범위는 CJK 통합 표의(U+4E00–9FFF), 히라가나·가타카나(U+3040–30FF),
CJK 확장 A(U+3400–4DBF), 한글 음절(U+AC00–D7AF)이다.

`normalize_text`는 ① 빈 입력이면 빈 문자열 ② `ComposingNormalizerBorrowed::new_nfkc()`
로 NFKC ③ 모든 공백 연속을 단일 스페이스로 접기 ④ CJK 문자 사이의 스페이스 제거
⑤ 양끝 trim. D8의 발산 근거를 함수 doc 주석에 남긴다.

`mod.rs`의 `inference`는 `tokenizer.decode(&token_ids).map(|text| normalize_text(&text))`가
된다. `ImageInput` 구조와 `preprocess`의 패딩 계약은 바뀌지 않는다.

## 소비자

| 소비자 | 변경 |
|---|---|
| `crates/koharu-pipeline/src/config.rs` (`OcrModel::HayaiOcr`) | 없음. id가 그대로다 |
| `crates/koharu-pipeline/src/stages/ocr.rs`, `bin/run.rs` | 없음 |
| `crates/koharu-ml/src/bin/hayai_ocr.rs`, `benches/hayai_ocr.rs` | 없음 |
| `crates/koharu-ml/src/lib.rs` | 없음 |
| `packages/bridge/src/protocol.ts`, `components/preferences/models.ts`, `PipelinePreferences.tsx` | 없음. id·표시 이름이 그대로다 |
| `packages/docs/{en,ja,zh}/models/vision.mdx` | 없음. 문구가 버전 중립적이다("Hayai OCR") |

체크포인트가 바뀌면 첫 실행에서 새 가중치를 내려받는다. 캐시 키는 리포 id와
revision이므로 이전 v2 파일과 충돌하지 않는다.

## 오류 처리

| 상황 | 처리 |
|---|---|
| 패치 격자가 0이거나 예산 초과 | `generate`가 `ensure!`로 실패한다(D7) |
| 가중치 이름 불일치 | 기존 로더가 그대로 실패한다. `proj.*`가 사라지고 `norm`/`mlp.*`/`out_norm`이 나타나므로 **여기서 먼저 터지는 것이 정상**이다. AGENTS.md에 따라 로더를 고치기 전에 아키텍처·파라미터 이름 불일치를 먼저 의심한다 |
| `vocab_size` 불일치 | 기존 `ensure!(tokenizer.len() == config.vocab_size)` 유지(16004) |
| 빈 이미지 | 기존 `cannot recognize an empty image` 유지 |
| 다운로드 실패 | 기존 `model_repository!` 경로 유지 |

## 검증

| # | 게이트 | 명령 | 통과 조건 |
|---|---|---|---|
| G1 | 단위 테스트(장치 불필요) | `cargo test -p koharu-ml hayai_ocr` | `decoder_vision_grid`, `normalize_text`, 기존 전처리 테스트가 전부 통과 |
| G2 | 컴파일 | `cargo check -p koharu-ml -p koharu-pipeline -p koharu-app --bins` | 성공 |
| G3 | 체크포인트 동등성 | `cargo test -p koharu-ml -- --ignored checkpoint_matches_upstream_character_sequence` | `benches/fixtures/ocr/title.png`에서 `"対策委員会です!"` |
| G4 | 픽셀 언셔플 채널 순서 | `cargo test -p koharu-ml -- --ignored pixel_unshuffle_channel_order` | 통과 |

### 결과 (2026-10-17)

CPU, debug 프로필, 브랜치 `feat/hayai-v25-nova`.

| # | 결과 | 비고 |
|---|---|---|
| G1 | 통과 | `hayai_ocr` 단위 테스트 8개(격자 1, 정규화 5, 기존 전처리 2) |
| G2 | 통과 | 경고 없음 |
| G3 | 통과 | 출력 `"対策委員会です!"`. Nova 체크포인트가 이름 불일치 없이 로드됨 |
| G4 | 통과 | 3×3 격자, 2채널, 패딩 3행 |

G3·G4는 LibTorch 런타임과 체크포인트 다운로드를 요구하므로 `#[ignore]`를 유지한다.
**G3의 기대값은 가정이 아니라 확인 대상이다.** 참조 포크가 같은 픽스처에 같은 문자열을
기대하지만, 그 포크가 실제로 통과하는지는 이 저장소에서 확인되지 않았다. 실패하면
`modeling_hayai.py`와 대조해 아키텍처·파라미터 이름·실행 순서를 먼저 점검한다.

G4가 따로 있는 이유는 순서를 틀려도 출력이 그럴듯하게 나오기 때문이다. 언셔플 순서가
어긋나면 LayerNorm과 MLP 열이 조용히 밀린 채 그럴듯한 텍스트가 나온다.
G3 하나로는 잡히지 않는다.

단일 픽스처는 AGENTS.md가 요구하는 "동일 입력에서의 구조적 비교"에 얇다. 그래서
upstream의 토큰 시퀀스와 기대값을 함께 대조하는 경로를 유지한다(기존 테스트가 그 역할이다).

## 테스트

| 계층 | 내용 | 위치 |
|---|---|---|
| 단위 | `decoder_vision_grid`: 짝수·홀수 격자, 극단 종횡비(`1×512` → `1×256`) | `config.rs` |
| 단위 | `pixel_unshuffle` 채널 인터리빙 `c * 4 + dy * 2 + dx` (참조 테스트 이식) | `model.rs` |
| 단위 | `normalize_text`: NFKC 매핑, 공백 접기, CJK 내부 공백 제거, 빈 입력 | `processor.rs` |
| 체크포인트 | `checkpoint_matches_upstream_character_sequence` — 기대값 `"対策委員会です!"` 유지 | `mod.rs` |

기존 `processor.rs` 테스트가 `get_image_size_for_max_num_patches`에 256을 리터럴로 넘긴다.
이 함수는 예산을 인자로 받는 순수 함수이고 참조도 같은 리터럴을 유지하므로 **고치지 않는다**.

## 범위 밖

| 항목 | 사유 |
|---|---|
| ONNX 경로(`bixii/hayai-ocr-v2.5-nova-onnx`) | LibTorch 경로로 결정했다. 별도 런타임을 들이지 않는다 |
| v2 병존 | 교체로 결정했다 |
| 패치 예산 설정화 | 512 고정으로 결정했다. 노출할 UI가 없다 |
| `koharu-docker`의 RPC·Docker 변경 | OCR 모델 자체와 무관하다. 포크는 비교 대상으로만 쓴다 |
| `max_num_patches`의 배치 단위 가변화 | Koharu는 크롭을 하나씩 처리한다. 배치 정적 버퍼와 마스크 테일은 필요 없다 |

## 참고

- `https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.5-nova` @ `2803d0dc…`
- `https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.5-nova/blob/2803d0dc…/modeling_hayai.py`
- `ternurarl/koharu-docker` — `crates/koharu-ml/src/hayai_ocr/`
- `specs/2026-10-17-sequential-model-residency-design.md` — OCR stage의 VRAM 레지던시.
  Nova는 `MAX_NUM_PATCHES`가 커져 활성화 메모리가 늘어난다. 상주 정책(D11)은 그대로다
