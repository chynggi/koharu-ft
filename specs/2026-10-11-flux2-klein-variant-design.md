# FLUX.2 Klein 변형(9B) 선택기 설계

> **폐기 (2026-10-11)** — 미구현. 이 증분은 채택되지 않았고, 후속 작업은
> stable-diffusion.cpp 최신화만 진행한다. 아래 내용은 평가 기록으로 보존한다.

작성일: 2026-10-11
개정: 2026-10-11 — 12GB 타깃 모델 조사 결과 반영(9B VRAM 근거 정정, 라이선스 고지 추가)

## 배경

현재 `flux2-klein` 인페인팅 모델은 4B 한 가지로 고정되어 있다.
`crates/koharu-ml/src/flux2_klein/mod.rs:29-37`이 세 체크포인트를
`model_repository!` 상수로 핀하고 있다.

| 컴포넌트 | 저장소 | 파일 |
|---|---|---|
| transformer | `unsloth/FLUX.2-klein-4B-GGUF` | `flux-2-klein-4b-Q4_K_M.gguf` |
| text encoder | `unsloth/Qwen3-4B-GGUF` | `Qwen3-4B-Q4_K_M.gguf` |
| VAE | `black-forest-labs/FLUX.2-small-decoder` | `full_encoder_small_decoder.safetensors` |

`fork/PHASE-2-NOTES.md`는 "Klein 9B"를 의도적으로 미출시로 두었다. 근거는
두 가지였다: 고정된 stable-diffusion.cpp 빌드가 9B 변환기를 로드할 수 있는지,
`FLUX.2-small-decoder`의 latent width가 9B 변환기와 맞는지 — 둘 다 Koharu의
Rust 소스를 읽어서는 답할 수 없는 **백엔드 빌드의 속성**이었다.

이번에 그 두 질문을 실제로 검증하기로 하고, 별도의 stable-diffusion.cpp 최신화
작업을 선행시키기로 했다. 이 문서는 검증 이후에 착수할 **변형 선택기**의 목표
상태를 정의한다.

## 확인된 외부 사실

조사 시점 기준으로 다음을 확인했다. 근거와 전체 수치는
`specs/2026-10-11-12gb-inpainting-model-survey.md`에 있다.

- `unsloth/FLUX.2-klein-9B-GGUF`가 Hugging Face에 존재하며
  `flux-2-klein-9b-Q4_K_M.gguf`(및 Q4_0/Q8_0 변형)를 배포한다.
- 9B 모델 카드는 "9B flow model with **8B Qwen3 text embedder**"라고 명시한다.
  즉 9B 변형은 transformer뿐 아니라 **텍스트 인코더도 4B와 다르다**:
  4B는 Qwen3-4B, 9B는 Qwen3-8B(`unsloth/Qwen3-8B-GGUF`).
- **최신 stable-diffusion.cpp는 FLUX.2 klein 9B를 공식 지원한다**
  (`docs/flux2.md`에 9B 항목과 `-r` 편집 예제, `leejet/FLUX.2-klein-9B-GGUF`).
  즉 "백엔드가 9B를 아는가"는 해소됐고, 남은 것은 우리 빌드 최신화와 실측이다.
- **VRAM:** 4B 스택 가중치 합은 5.35GB, 9B 스택은 ~11.2GB다. 9B는 활성화/KV를
  포함하면 12GB를 초과한다.
- **라이선스:** 4B는 Apache 2.0(상용 가능), **9B는 FLUX 비상용 라이선스**다.
- **저비트 기준선은 Q4_K_M이다.** Q4_0/Q3 이하는 마스크 기반 제거 작업에서
  디테일 손실이 보고된다.
- **혼합 포맷 주의:** GGUF 변환기 + 비-GGUF 텍스트 인코더 조합은 아티팩트가
  보고된다. 변환기와 텍스트 인코더를 같은 계열·같은 GGUF로 맞추는 것이 안전하다.

### 12GB 타깃 모델 선정 결론

| 후보 | 판정 |
|---|---|
| FLUX.2 klein 4B (현행) | **채택** — ~7–8GB로 여유 있게 fit, Apache 2.0 |
| Qwen-Image-2.1 | 기각 — 12GB에서 경계형(편집 시 초과), Qwen Research License(연구 전용) |
| FLUX.2 klein 9B | 12GB 부적합(~11.2GB 가중치, 초과) + 비상용 → 16GB+ 전용 선택지로만 유지 |
| Z-Image-Turbo | 기각 — T2I 전용, 텍스트 제거·배경 재구성 역할 대체 불가 |
| Qwen-Image 20B / SDXL 인페인팅 | 기각 — 메모리 초과 / 품질 구세대 |

4B의 기본 quant는 **Q4_K_M을 유지한다.** 이는 조사가 밝힌 최저 합리선이고, 상향
(Q6_K 변환기 + Q5_K_M 텍스트 인코더, ~8.5–9.5GB)은 기존 컴포넌트 오버라이드로
이미 가능하다.

## 결정 사항

| # | 결정 | 근거 |
|---|---|---|
| D1 | 변형 선택기를 **FLUX.2 Klein 설정 내부**에 둔다 (인페인팅 모델 목록에 별도 엔트리를 만들지 않는다) | 모델 목록 엔트리는 "이 체크포인트가 동작한다"는 별도 주장이 되고, 설정·프로세서 미러링이 한 벌 더 늘어난다 |
| D2 | **순차 진행**: sdcpp 최신화 → 9B 검증 → 이 증분 구현·출시 | 경로 스왑만으로 지원을 주장하지 않는다는 기존 원칙 유지 |
| D3 | 변형은 **기본값을 소유하는 열거형**으로 구현하고, 컴포넌트별 오버라이드는 그 위에 적용 | "기본값과 제공자별 동작은 그것을 소유한 컴포넌트가 갖는다" — 저장소 규칙과 일치 |
| D4 | 변형과 다른 컴포넌트 소스를 쓰면 **경고만** 하고 차단하지 않는다 | 사용자가 명시적으로 만든 조합을 막을 근거가 없다. 혼합 조합의 아티팩트 보고가 있어 경고의 실효성이 있다 |
| D5 | 12GB는 **GPU 상주 기준**이며 CPU offload는 성능상 배제한다 | offload 경로는 실사용에서 너무 느리다 |
| D6 | 9B 선택지에 **VRAM 요건과 비상용 라이선스를 UI 라벨로 고지**한다. 4B 기본값(Q4_K_M)은 변경하지 않는다 | 9B는 12GB에서 실행 불가이고 비상용이다. 그 사실이 선택기에 드러나야 한다 |

### `fork/PHASE-2-NOTES.md` D3와의 관계

Phase 2의 D3는 "프리셋 없음, 컴포넌트 오버라이드만"이었다. 이 설계는 그 결정을
검증 게이트(D2)와 함께 갱신한다. 즉 9B 항목은 **검증을 통과한 뒤에만** 출시하고,
검증 전에는 어떤 형태의 9B 프리셋도 존재하지 않는다.

## 설정 스키마와 해석 규칙

`koharu-pipeline/src/stages/inpainting.rs`의 `Flux2KleinSourceConfig`에 `variant`
필드를 추가한다.

```rust
pub enum Flux2KleinVariantConfig {
    #[default]
    Klein4B,
    Klein9B,
}
```

- 직렬화 값은 `klein_4b` / `klein_9b` (기존 `WeightsFormatConfig`의 snake_case 관행).
- `#[serde(default)]`이므로 변형 필드가 없는 기존 `[processor.flux2-klein]` 섹션은
  `klein_4b`로 로드된다. 기존 설정 파일과 프로젝트 파일은 변경되지 않는다.
- 저장되는 것은 변형 이름뿐이며 체크포인트 경로는 저장하지 않는다.

**해석 우선순위**

1. 컴포넌트 소스가 `builtin`이면 → 선택된 변형이 소유한 기본 저장소로 해석
2. `local_file` / `hugging_face` / `url`이면 → 변형과 무관하게 그 값이 우선

변형을 전환하면 `builtin` 컴포넌트는 자동으로 새 변형의 기본값을 따르고, 명시적
오버라이드는 유지된다. VAE는 두 변형이 공유하는 기본값을 갖는다.

**변형별 기본값** (9B 저장소의 커밋 핀은 구현 시점에 40자 해시로 고정한다)

| 변형 | transformer | text encoder | VAE | 라이선스 |
|---|---|---|---|---|
| `klein_4b` (기본) | `unsloth/FLUX.2-klein-4B-GGUF` / `flux-2-klein-4b-Q4_K_M.gguf` | `unsloth/Qwen3-4B-GGUF` / `Qwen3-4B-Q4_K_M.gguf` | `black-forest-labs/FLUX.2-small-decoder` | Apache 2.0 |
| `klein_9b` | `unsloth/FLUX.2-klein-9B-GGUF` / `flux-2-klein-9b-Q4_K_M.gguf` | `unsloth/Qwen3-8B-GGUF` / `Qwen3-8B-Q4_K_M.gguf` | 동일 | FLUX 비상용 |

## 백엔드 구조

### `koharu-ml` — 변형이 기본값을 소유

- `flux2_klein/mod.rs`의 `model_repository!` 호출을 변형별로 분리한다. 4B 변환기 +
  `Qwen3-4B`, 9B 변환기 + `Qwen3-8B`, VAE 공유 1개. 상수 이름이 겹치지 않도록
  변형 접미를 붙인다 (예: `TRANSFORMER_WEIGHTS_4B`, `TRANSFORMER_WEIGHTS_9B`).
- `Flux2KleinVariant { Klein4B, Klein9B }` 열거형을 추가한다. 각 변형이 자기 기본
  파일을 소유하며 `transformer_weights()` / `text_encoder_weights()`로 노출한다.
  VAE는 두 변형 공통 접근자를 쓴다.
- `Flux2KleinSource`에 `variant` 필드를 추가한다. `resolve`는
  `source.transformer.resolve(source.variant.transformer_weights())` 형태가 된다.
- `Flux2KleinSource::default()`는 `Klein4B` + 전부 `Builtin`으로, 이전 동작을 정확히
  재현한다.
- **`Flux2Klein::load` / `Flux2KleinInpaint::load`의 시그니처는 바뀌지 않는다.**
  Phase 2에서 이미 `&Flux2KleinSource`를 받도록 정리되어 있기 때문이다.
- `Model::new`는 여전히 파일 3개와 `VaeFormat::Flux2`만 받는다. 아키텍처는 파일
  메타데이터에서 읽히므로, 백엔드가 9B를 모르면 컨텍스트 생성 단계에서 실패한다.

### `koharu-pipeline` — 미러

- `Flux2KleinVariantConfig`를 추가하고 `ComponentSourceConfig`와 동일한 방식으로
  `koharu_ml::flux2_klein::Flux2KleinVariant`에 1:1 `From` 매핑한다 (로직 없음).
- `Flux2KleinSourceConfig`에 `variant` 필드를 추가하고 `From`에서 전달한다.
- `Flux2KleinConfig::validate`는 변경하지 않는다 (열거형이라 값 검증이 불필요).
- 하위 호환 레이어는 두지 않는다. 인-리포 소비자를 같은 변경에서 갱신한다.

### 디버그 바이너리 / 벤치

- `bin/flux2_klein.rs`에 `--variant` 플래그를 추가한다 (기본 `klein-4b`). 이것이 2단계
  검증의 실제 도구다 — UI 왕복 없이 9B를 로드해 봐야 한다.
- 벤치 호출부도 같은 변경을 따른다.

## UI

### 변형 선택기

`packages/koharu/components/preferences/Flux2KleinOptions.tsx` 상단에 기존 `Select`
컴포넌트로 변형 선택을 추가한다. 라벨은 D6에 따라 VRAM 요건과 라이선스를 고지한다.

```
4B — 12GB급 (기본)
9B — 16GB+ 전용, 비상용
```

### 불일치 경고

- **조건:** `transformer` 또는 `text_encoder`의 소스 `kind`가 `builtin`이 아닐 때.
  VAE는 변형 간 공유이므로 제외한다.
- **문구:** 선택한 변형 라벨을 보간한다. 예: "선택한 변형(9B)과 다른 체크포인트를
  사용 중입니다. 변환기와 텍스트 인코더는 같은 계열이어야 합니다."
- **톤:** 주의(muted/amber) 표시. 차단하지 않고 저장·실행을 허용한다.
- **구현 원칙:** 저장소 이름이나 파일명을 추측해 매칭하지 않는다. 변형×인코더 조합
  규칙이라는 특수 케이스 목록을 만들지 않기 위해서다.
- 백엔드에는 이 경고에 대응하는 검증을 추가하지 않는다.

### i18n

`settings.pipeline.options.variant` 계열 키를 신설한다 (`variant.klein4b`,
`variant.klein9b`, `variantMismatch` 포함). `en-US`·`ko-KR`에 추가하고, 나머지
로케일은 구현 시점에 기존 키 세트 유지 관행을 확인해 맞춘다.

## 테스트

| 계층 | 검증 내용 |
|---|---|
| `koharu-ml` | 변형별 기본 해석이 올바른 파일을 넘기는지, `Flux2KleinSource::default()`가 `Klein4B`인지 |
| `koharu-pipeline` | 변형 필드가 없는 기존 TOML이 `klein_4b`로 로드되는지(기존 backward-compat 테스트 확장), 변형 포함 왕복 직렬화 |
| UI | 변형 선택 시 경고 노출/미노출 (기존 preferences 테스트 확장) |

## 검증 게이트와 출시 순서

1. **stable-diffusion.cpp 최신화** (별도 task)
2. **9B 검증** — 디버그 바이너리로 `--variant klein-9b` 실행: 컨텍스트 생성 성공
   여부, `FLUX.2-small-decoder` latent 정합, 16GB+ GPU에서의 실제 인페인팅 결과.
   아키텍처 지원 자체는 upstream 문서로 확인됐으나, 마스크 고정 인페인팅 경로가
   9B에서도 동일하게 동작하는지는 실측으로 확인한다.
3. **이 증분 구현·출시**

2단계가 실패하면 9B 항목만 빠지고 스키마·경고·UI 구조는 그대로 4B 단일 변형으로
성립한다. 검증을 통과하기 전에는 9B 프리셋, 문서 주장, 기본값 변경을 출시하지 않는다.

## 범위 밖

| 항목 | 사유 |
|---|---|
| Qwen-Image-2.1 도입 | 12GB 경계형(편집 시 초과) + Qwen Research License(연구 전용) → 기각. 근거: 조사 문서 §5 |
| Z-Image-Turbo, Qwen-Image 20B, SDXL 인페인팅 | T2I 전용 / 메모리 초과 / 품질 구세대 |
| 4B 기본 quant 상향 | Q4_K_M이 저비트 기준선이며, 상향은 기존 오버라이드로 가능. 기본값 변경은 기존 사용자 재다운로드를 유발한다 |
| 12GB에서의 파라미터 상주 정책 | `flux2_klein/model.rs:56`이 `memory_free >= 20GiB`일 때만 VRAM 상주로 두고 그 미만은 RAM 상주 경로로 떨어뜨린다(반복 전송으로 느려짐). D5의 "CPU offload 배제" 정책과 충돌하는 지점이나, 이전에 이관된 VRAM 관리 증분이 소유한다 |
| 비-인페인트 FLUX.2 설정의 UI 노출 | `Flux2Klein`(생성)은 현재 디버그 바이너리에서만 도달 가능 |
| Text CFG / scheduler / sample method | diffusers 참조 파이프라인의 아키텍처 선택이며 사용자 취향이 아니다 |

## 참고

- `specs/2026-10-11-12gb-inpainting-model-survey.md` — 12GB 타깃 모델 조사 기록
- `unsloth/FLUX.2-klein-9B-GGUF`, `unsloth/Qwen3-8B-GGUF`
- `fork/PHASE-2-NOTES.md` (D1~D6, Klein 9B 미출시 사유)
- `fork/ARCHITECTURE.md` §5 (FLUX.2 Klein 하드코딩 지점)
