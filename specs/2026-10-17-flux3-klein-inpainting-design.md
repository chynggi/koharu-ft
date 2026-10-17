# FLUX 3 Image Klein 인페인팅 모델 설계

작성일: 2026-10-17

## 배경

Black Forest Labs가 2026-10-13에 FLUX 3 Image Klein을 공개했다. 이 문서는 이 모델을
Koharu의 인페인팅 모델로 지원하는 목표 상태와 출시 게이트를 정의한다.

현재 확산 기반 인페인터는 FLUX.2 Klein 4B(`crates/koharu-ml/src/flux2_klein/`)이며
Apache 2.0이다. FLUX.2 Klein 9B 변형 선택기(`specs/2026-10-11-flux2-klein-variant-design.md`)는
폐기됐다. 그 문서가 "같은 계열 변형"이라는 이유로 별도 엔트리를 기각한 근거는 FLUX 3에
적용되지 않는다. 텍스트 인코더, 스케줄러, 스텝 수, 라이선스가 모두 다르기 때문이다.

## 확인된 외부 사실

조사 시점: 2026-10-17. 근거는 `hf` CLI로 받은 모델 카드와 파일 목록, 그리고 웹 조사(sd.cpp PR,
BFL 블로그, `koharu-rs/diffusion` 릴리스 목록)다.

### 모델

- `black-forest-labs/flux-3-image-klein` @ `337337621b8526e120f2275d250242f4a6bedd75`
  - 16스텝 distilled 체크포인트다. guidance는 없다(CFG 1.0).
  - `-base`는 50스텝·guidance 3.5짜리 미세조정용이며, 이 설계의 범위 밖이다.
  - 이미지 transformer는 5.6B다.
  - 텍스트 인코더는 **Qwen3-VL-4B-Instruct**(Apache 2.0)다.
  - VAE는 **FLUX.2 VAE**(`ae.safetensors`)다.
- 최대 10장의 참조 이미지로 편집한다. 마스크 입력은 없다. 프롬프트 bbox는 "하드 마스크가 아니다"라고
  명시돼 있다.
- 모델 카드는 프롬프트 업샘플러를 **반드시** 쓰라고 한다. 짧은 프롬프트를 그대로 보내면 결과가 평범하다고 한다.
- 공식 업샘플러는 `black-forest-labs/flux-3-image-klein-pup` @ `3f3edb52b2c55de0d49feab28598b68f7f8e4830`다.
  - 아키텍처는 Qwen3.5-9B VLM(`qwen3_5`)이고 Apache 2.0이다. BF16 기준 18.8GB다.
  - `system_prompts.json`의 키는 `t2i`, `i2i`, `t2i_bbox`, `i2i_bbox`다.
- **라이선스는 FLUX Kommunity License v1.0**이다. 비상용 이용, 그리고 연매출(계열사 포함)
  US$5M 미만인 Qualifying User의 상용 출력 생성만 허용한다.
- BFL 블로그 기준 생성기 가중치는 BF16 11.2GB, FP8 5.6GB다.

### 런타임

- stable-diffusion.cpp [PR #2143](https://github.com/leejet/stable-diffusion.cpp/pull/2143)이
  2026-10-15에 머지됐다(`8cd731e`). 이 PR로 FLUX.3 Image Klein과 Klein Base를 지원한다.
  - 지원 범위는 모델 감지, Qwen3-VL 컨디셔너, 생성, 참조 이미지 편집이다.
  - 기본 discrete 스케줄러의 flow shift는 6.93이다.
  - upstream 문서(`docs/flux3.md`)가 검증한 것은 생성과 `-r` 편집뿐이다.
    **마스크 인페인팅 경로는 검증되지 않았다.**
- `koharu-rs/diffusion`에 **`master-968-8cd731e`** 릴리스가 있다(2026-10-16).
  현재 고정 버전(`crates/koharu-runtime/src/runtime/packages/diffusion.rs`의 `RELEASE`,
  `master-956-1b0ba10`)은 FLUX.3를 모른다.
- llama 런타임 고정 버전은 `b11552`다. 이 빌드가 `qwen3_5`를 로드하는지는 확인되지 않았다.

### 가중치 배포본

| 컴포넌트 | 저장소 @ 커밋 | 파일 | 비고 |
|---|---|---|---|
| transformer | `DogukanUrker/FLUX.3-Image-Klein-GGUF` @ `652e3dae6f7f26777c8749c99dcb3bb643a8350c` | `flux3-image-klein-Q4_K.gguf` (3.3GB) | sd.cpp로 변환하고 RTX 3060 12GB에서 테스트했다고 한다. 피크 7.4GB(1024², Q8_0 TE) |
| text encoder | `Qwen/Qwen3-VL-4B-Instruct-GGUF` @ `1cd86afb9a95c410a6038ab3b40d8b578c892266` | `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | sd.cpp PR 예제와 같은 양자화 |
| VAE | `Comfy-Org/flux2-klein` @ `5f526678002e43af5551dadb73ce2e8c91b43afe` | `split_files/vae/flux2-vae.safetensors` | FLUX.2 VAE. BFL 저장소의 `ae.safetensors`도 같은 VAE지만 그 저장소가 gated(`gated=auto`)라 토큰 없이는 받을 수 없다. sd.cpp `docs/flux3.md`도 FLUX.2 VAE를 기본으로 안내한다 |
| upsampler | `prithivMLmods/flux-3-image-klein-prompt-upsampler-GGUF` @ `71713afaf9bce90074c078254d07f5b348f9c372` | `flux-3-image-klein-pup.Q4_K_M.gguf` | 커뮤니티 GGUF 변환본 |
| upsampler mmproj | 위와 같음 | `flux-3-image-klein-pup.mmproj-q8_0.gguf` | i2i 캡션에 필요 |
| system prompts | `black-forest-labs/flux-3-image-klein-pup` @ `3f3edb52b2c55de0d49feab28598b68f7f8e4830` | `system_prompts.json` | `i2i` 키를 사용 |

`ml-resources/flux-3-image-klein-comfy` @ `496984d46f5ceeeb519b4d09a12cc0e58c0d6003`(gated 아님)의
`vae/flux2-vae.safetensors`는 위 VAE와 SHA-256이 같다(`868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3`,
336,211,292바이트). 다만 gated인 BFL 저장소를 제3자가 미러링한 것이라 내장 기본값으로는 쓰지 않는다.
사용자는 오버라이드로 고를 수 있다.

`unsloth/flux-3-image-klein-GGUF`는 diffusers와 Unsloth Desktop용이다. sd.cpp 호환이 명시돼
있지 않아 기본값으로 쓰지 않는다. 사용자가 오버라이드로 선택할 수는 있다.

## 결정 사항

| # | 결정 | 근거 |
|---|---|---|
| D1 | `flux3-klein`을 **별도 인페인팅 모델 엔트리**로 추가한다. FLUX.2 Klein 4B는 기본값으로 유지한다 | 아키텍처·옵션·라이선스가 다르다. 기본값을 바꾸면 라이선스가 Apache에서 Kommunity로 후퇴하고 기존 사용자가 다시 다운로드해야 한다 |
| D2 | **공식 PUP 업샘플러**로 타일마다 캡션을 생성한다 | 모델 카드가 업샘플러 없는 사용을 금한다 |
| D3 | **페이지 단위 2단계 스왑**을 쓴다: 페이지의 모든 타일 캡션(PUP) → 스왑 → 인페인팅(FLUX 3) | 12GB에서 두 모델을 동시에 상주시킬 수 없다. 페이지당 스왑 횟수를 2회로 고정하고, 구조를 인페인팅 stage 안에 가둔다 |
| D4 | 스왑은 **`flux3_klein` 컴포넌트가 직접 소유**한다(접근법 A) | 타일 분할과 캡션의 결합은 이 모델에만 의미가 있다. 다른 stage와 `accelerator.rs`는 바꾸지 않는다 |
| D5 | RAM 캐싱은 런타임마다 이미 있는 수단으로 한다. FLUX 3는 sd.cpp `params_backend: "*=cpu"`, PUP는 mmap 로드 + 해제 후 OS 페이지 캐시 | 새 캐시 계층을 만들지 않는다. 다시 올릴 때 디스크 I/O 없이 RAM→VRAM 복사만 일어난다 |
| D6 | 업샘플러를 끄는 설정과 업샘플러 실패 시 폴백은 **두지 않는다** | 조용한 품질 저하를 숨기지 않는다. 끄고 돌리는 비교는 디버그 바이너리 플래그로만 한다 |
| D7 | 검증 게이트 G1–G4를 통과하기 전에는 엔트리, 문서 주장, 고지 수치를 출시하지 않는다 | 경로만 바꿔 놓고 지원을 주장하지 않는다는 기존 원칙 |
| D8 | 공용 VRAM↔RAM 레지던시 매니저(접근법 B)는 **후속 조사**로 둔다 | 범위가 크고, 런타임마다 "RAM 강등"의 의미가 다르다. `fork/VRAM-ACCOUNTING-DESIGN.md`의 미결 항목과 얽힌다 |

## 백엔드 구성 — `koharu-ml`

새 모듈은 `crates/koharu-ml/src/flux3_klein/`다. upstream 매핑의 기준은 sd.cpp `8cd731e`의 `docs/flux3.md`,
그리고 `black-forest-labs/flux-3-image-klein` @ `3373376…` 모델 카드의 업샘플러 사용 예제다.

| 파일 | 책임 |
|---|---|
| `source.rs` | `Flux3KleinSource { transformer, text_encoder, vae, upsampler, upsampler_mmproj }`. 각 필드는 기존 `ComponentSource`다. `validate`는 다섯 필드를 검증하고 컴포넌트 이름을 컨텍스트로 붙인다 |
| `mod.rs` | 위 표의 기본 저장소를 `model_repository!`로 고정하고 해석한다. `Flux3KleinInpaint`를 둔다 |
| `model.rs` | sd.cpp `Context`를 만든다: `diffusion_model_path`, `llm_path`(Qwen3-VL-4B), `vae_path`, `VaeFormat::Flux2`. 가속기 사용 시 **항상** `params_backend: "*=cpu"`로 둔다. 파라미터는 RAM에 상주하고 실행 중에만 VRAM에 올라간다(D5). 이는 FLUX.2의 20GiB 조건 분기와 다르며, 그 이유를 주석으로 남긴다 |
| `upsampler.rs` | `Flux3PromptUpsampler`. 기존 `llm` 모듈의 `Llm` + `MtmdOptions` 경로로 PUP와 mmproj를 로드한다. `load`는 파일을 해석·다운로드하고 시스템 프롬프트만 읽는다. 모델은 들고 있지 않는다. `caption(images, request) -> Result<Vec<String>>`이 핵심이다 |

### `Flux3PromptUpsampler::caption`

upstream 모델 카드의 `upsample(prompt, images)`를 1:1로 따른다.

- 시스템 메시지는 `system_prompts["i2i"]`다.
- 사용자 메시지는 이미지 1장 다음에 아래 텍스트가 오는 형태다.
  `"Reference images follow in order (the first image, the second image, ...).\nUser request: {request}\nOutput the caption."`
- `add_generation_prompt`, `enable_thinking=False`, greedy 디코딩(`temperature 0`, 반복 패널티 1.0 —
  `GenerationOptions` 기본값 1.2를 명시적으로 덮는다), `max_new_tokens=3584`.
- 참조 이미지는 upstream 추론 코드처럼 최대 1MP로 줄인다(`Flux2ImageProcessor::fit_to_area`).
- 채팅 프롬프트는 이미지와 무관하므로 `caption` 호출마다 한 번만 렌더링한다.
- 출력은 특수 토큰을 빼고 디코딩한 뒤 trim한다.
- 수명주기: `caption`이 한 번의 호출 안에서 ① `Llm`을 로드하고(VRAM 점유) ② 모든 이미지의 캡션을 만든 뒤
  ③ `Llm`을 drop해서 VRAM을 반환한다. 해제를 호출자에게 맡기지 않으므로, 오류가 나도 VRAM이 남지 않는다.
  페이지마다 가중치를 다시 읽는다. OS 파일 캐시에서 올지 디스크에서 올지는 llama.cpp 로드 모드에 달려 있다.
  반환된 future를 drop해도 작업 스레드에 넘어간 캡션 생성은 멈추지 않는다. 모델은 남은 이미지가 끝난 뒤 해제된다.

### `Flux3KleinInpaint::inference(caption, image, mask, options)`

구조는 `Flux2KleinInpaint::inference`와 같다.

- 입력 검증 → `max_pixels`로 축소 → 16배수 정렬 → `padding_mask_crop` 처리 → 마스크 이진화
  → `init_image` + `mask_image` + `reference_images=[init_image]` + `strength` → overlay 합성.
- FLUX 3 고유 값
  - `sample_steps` 기본 16, `text_cfg` 1.0, `SampleMethod::Euler`
  - 스케줄러는 `Scheduler::Discrete`를 명시하고 flow shift는 비워 둔다. sd.cpp가 FLUX.3 기본값 6.93으로 채운다
  - init 이미지를 편집 reference로도 넘기는 것은 FLUX.2 인페인트 포트를 따른 것이다. FLUX 3에는 upstream 인페인트
    파이프라인이 없으므로 효과(또는 지운 글자의 재등장)는 품질 게이트에서 확인한다
- strength는 `Flux2ImageProcessor::sd_strength`(FLUX.2와 공유)로 변환한다. diffusers `get_timesteps`의 시작
  스텝을 정확히 재현하고, sd.cpp의 f32 `t_enc` 절삭이 같은 스텝에 떨어지도록 경계 아래 중간값을 넘긴다. 이 공유
  함수는 기존 FLUX.2 산식의 off-by-one(일부 비기본 스텝 수)도 함께 고친다
- 전처리와 overlay는 `crate::flux2_klein::processor::Flux2ImageProcessor`를 그대로 쓴다. FLUX 3는 FLUX.2
  VAE와 16픽셀 정렬을 공유하고, 마스크 인페인팅 전처리는 모델 아키텍처가 아니라 diffusers FLUX.2 인페인트
  파이프라인에서 옮긴 우리 코드다. 소유권은 그 upstream 대응을 가진 FLUX.2 모듈에 남기고, 가시성만 `pub(crate)`로 넓힌다.

### 런타임 최신화

- `crates/koharu-runtime/src/runtime/packages/diffusion.rs`의 `RELEASE`를 `master-968-8cd731e`로 올린다.
- `koharu-diffusion-sys`의 `include/stable-diffusion.h`는 같은 태그의 upstream 파일을 그대로 받는다
  (`.agents/skills/runtime/scripts/sync.sh`는 llama 헤더까지 최신으로 바꾸므로 diffusion 줄만 고정 태그로 실행).
  바인딩은 `build.rs`의 bindgen이 빌드 시 만든다. 손으로 고치지 않는다.
- 헤더 변경 때문에 `koharu-diffusion`의 안전 API가 바뀌면, 같은 변경 안에서 인-리포 소비자를 모두 갱신한다.
  호환 레이어는 두지 않는다.

## 파이프라인 — `koharu-pipeline`

### 설정 스키마 (`stages/inpainting.rs`)

- `InpaintingModel::Flux3Klein(Flux3KleinConfig)`를 추가한다. 모델 이름은 `"flux3-klein"`이다.
- `Flux3KleinConfig`의 필드(`#[serde(default)]`)는 다음과 같다.
  - `request: String`: 업샘플러 지시문. 기본값은
    `"Remove all text and lettering, and reconstruct the background behind it."`
  - `source: Flux3KleinSourceConfig`: 다섯 필드, 각각 `ComponentSourceConfig`
  - `steps: u32`(16), `strength: f64`, `seed: i64`(`-1`이면 호출마다 새 seed),
    `padding_mask_crop: Option<u32>`, `max_pixels: u32`
- 기본값은 `koharu_ml::flux3_klein::Flux3KleinInpaintOptions::default()`에서 가져온다.
  기본값은 그것을 소유한 컴포넌트에 둔다.
- `From<Flux3KleinSourceConfig> for Flux3KleinSource`는 로직 없는 1:1 변환이다.
- `validate`: 소스 검증, 그리고 `request`의 NUL·빈 문자열 거부.

### 타일 계획 분리

현재 `inpaint_tiled(image, mask, text_mask, flat_fill_regions, inference)`는 타일 계산과
순차 실행이 한 함수에 묶여 있다. 이걸 `TilePlan`으로 나눈다.

- `TilePlan::new(image, mask, text_mask, flat_fill_regions) -> Result<TilePlan>`
  - 차원을 검증하고, 균일 영역을 채우고(`fill_uniform_regions`), 타일과 라벨을 계산한다.
- `TilePlan::crops() -> Vec<DynamicImage>`
  - 균일 영역 채움 뒤, 인페인팅 전 출력에서 자른 타일별 이미지 크롭이다. 순서는 `run`의 호출 순서와 같다.
    캡션에는 마스크가 필요 없으므로 이미지만 돌려준다.
- `TilePlan::run(self, inference: impl FnMut(usize, &DynamicImage, &GrayImage) -> Result<DynamicImage>)`
  - 기존과 같이 진행 중인 출력에서 잘라 순차적으로 합성한다. 첫 인자는 타일 인덱스다.
- 기존 모델(LaMa, MI-GAN, Manga Inpaintor, AOT, FLUX.2, RORem, PowerPaint) 모두
  `TilePlan::new(..)?.run(|_, image, mask| ..)`으로 옮긴다. `inpaint_tiled`는 지운다.

### 페이지 처리 흐름 (`Model::Flux3`)

`Model::Flux3 { inpaint: Arc<Mutex<Flux3KleinInpaint>>, upsampler: Arc<Flux3PromptUpsampler>, config: Box<Flux3KleinConfig> }`.
처리는 accelerator 퍼밋 하나 안에서 진행한다.

1. `TilePlan::new`. 타일이 없으면 바로 반환한다.
2. **Phase 1, 캡션**: `upsampler.caption(plan.crops(), config.request)`. 이 호출이 PUP 로드, 캡션 생성, 해제를 모두 맡는다.
3. **Phase 2, 인페인팅**: `plan.run(|i, crop, mask| inpaint.inference(&captions[i], crop, mask, &options))`.
4. 다음 페이지에서는 1부터 반복한다. FLUX 3 컨텍스트는 stage가 로드돼 있는 동안 유지한다.

**의도적 차이**: 타일이 겹치면 Phase 2의 init_image는 기존처럼 앞 타일의 결과가 반영된 출력에서
자른다. 반면 캡션은 Phase 1에서 인페인팅 전 크롭으로 만든다. 이 차이는 `Model::Flux3` 실행부에 주석으로 남긴다.

### 자원

- 호스트 RAM 추가 사용량은 약 12GB다(FLUX 3 스택 상주 + PUP 페이지 캐시). G4 실측으로 확정한다.
- 최초 다운로드는 고정 파일 6개의 합인 12.36GB다(Hub API로 확인, 실측 불필요).
- PUP 로드에 실패하면 기존 `accelerator.recover` 경로가 다른 stage를 언로드한 뒤 재시도할 수 있다.
  이 경로는 변경하지 않는다.
- 캡션 캐시는 두지 않는다. 같은 페이지를 다시 실행하면 캡션도 다시 생성한다.

## 디버그 바이너리 — `crates/koharu-ml/src/bin/flux3_klein.rs`

- 입력: `--input`, `--mask`, `--output`, `--request`, `--steps`, `--strength`, `--seed`, `--padding-mask-crop`,
  `--max-pixels`, `--cpu`, 컴포넌트별 오버라이드(`--transformer`, `--text-encoder`, `--vae`, `--upsampler`, `--upsampler-mmproj`).
- `--caption-only`: PUP로 캡션만 출력한다(G2). sd.cpp 런타임은 건드리지 않는다.
- `--no-upsampler`: `request`를 그대로 캡션으로 쓴다. 비교 측정 전용이며 파이프라인 설정에는 대응 항목이 없다(D6).
- 순서는 파이프라인과 같다: FLUX 3 로드(`phase=load`) → 그 컨텍스트가 상주한 채로 캡션(`phase=caption`,
  로드+생성+해제 합산) → 워밍업 1회 뒤 측정한 인페인팅(`phase=inpaint`). 시작할 때 장치·입력 크기·옵션을 한 줄로 출력한다.
- VRAM 피크는 바이너리가 아니라 외부 샘플러(`nvidia-smi`)로 잰다(Task 8).

## UI

### 모델 목록 (`packages/koharu/components/preferences/models.ts`)

- `inpainting` 배열에서 `'flux2-klein'` 바로 뒤에 `'flux3-klein'`을 넣는다. 표시 이름은 `'FLUX 3 Image Klein'`이다.
- `defaultModel`의 `'flux3-klein'`은 `{ model, request: <Rust 기본값과 같은 문자열> }`이다.
- 프리셋 매핑 블록에 `'flux3-klein': { request }` 분기를 추가한다.
- `Flux3KleinConfig`·`Flux3KleinSourceConfig` TS 타입, 그리고 `InpaintingModel`·`ProcessorConfig` 확장은
  `packages/bridge/src/protocol.ts`에 손으로 추가한다. 이 파일은 생성물이 아니라 손으로 쓰는 RPC 클라이언트다.
- `replaceStage`의 `'flux3-klein'` 프로필은 모든 필드(`request`, `source`, `steps`, `strength`, `seed`,
  `padding_mask_crop`, `max_pixels`)를 보존한다(LaMa·MI-GAN 분기와 같은 방식).

### 옵션 패널 (`components/preferences/Flux3KleinOptions.tsx`, 신규)

- `Flux2KleinOptions.tsx`와 같은 구조지만 별도 컴포넌트다(필드와 기본값이 다르다).
- 필드: `request`(업샘플러 지시문과 설명), `steps`, `strength`, `seed`, `maxPixels`,
  `paddingMaskCrop`(기존 키 재사용), 컴포넌트 소스 5개(`ComponentSourceField` 재사용).
- `PipelinePreferences.tsx`의 `switch`에 `case 'flux3-klein'`을 추가한다.

### 고지 (패널 상단, 경고 톤 muted/amber, 차단하지 않음)

1. 라이선스: FLUX Kommunity License. 비상용, 그리고 연매출 US$5M 미만 사업자의 상용 출력만 허용된다는 점. `LICENSE.md` 링크를 붙인다.
2. 요구 자원: VRAM, 시스템 RAM 추가 사용량, 최초 다운로드 크기. 수치는 G4 실측값을 쓴다.
3. 속도: 페이지마다 업샘플러와 생성 모델을 번갈아 로드하므로 FLUX.2보다 느리다는 점.

### i18n

- 새 키는 모두 `settings.pipeline.options` 아래에 둔다: `request`, `requestDescription`, `source.upsampler`,
  `source.upsampler_mmproj`(기존 `source.text_encoder`처럼 필드 이름을 그대로 키로 쓴다), `flux3.license`,
  `flux3.licenseLink`, `flux3.resources`, `flux3.speed`.
- `flux3.resources`는 `{{vram}}`, `{{ram}}`, `{{download}}`를 보간한다. 수치는 `Flux3KleinOptions.tsx`의 상수가
  소유한다. G4 실측값이 바뀌어도 9개 로케일 문자열은 고치지 않는다.
- 로케일 9개(en-US, es-ES, ja-JP, ko-KR, pt-BR, ru-RU, tr-TR, zh-CN, zh-TW) 모두에 추가한다.

### 문서 (`packages/docs/{en,ja,zh}`)

- `models/vision.mdx`: FLUX 3 Klein 항목(라이선스, 업샘플러, 요구 자원).
- `guides/cleanup.mdx`: FLUX.2 대신 쓸 때의 지침 한 단락.
- `hardware.mdx`: 가속기 목록 문장(Vulkan 탭)에 FLUX 3 Image Klein을 추가한다. 이 문서에는 VRAM 표가 없다.
- 문서도 G4를 통과한 뒤에만 갱신한다.

## 오류 처리

| 상황 | 동작 |
|---|---|
| PUP 로드 또는 생성 실패(OOM, 아키텍처 미지원) | 페이지 처리를 오류로 끝낸다. 폴백 없음(D6) |
| 빈 캡션, 또는 `max_new_tokens`에 도달 | 오류로 처리하고 타일 인덱스를 컨텍스트에 붙인다 |
| Phase 2 실패 | PUP는 이미 해제된 상태다. FLUX 3 컨텍스트는 유지해서 다음 페이지에 재사용한다 |
| 설정 오류 | 저장할 때와 로드 직전에 같은 `validate`로 거부한다 |

## 검증 게이트와 출시 순서

| # | 게이트 | 통과 기준 |
|---|---|---|
| G1 | sd.cpp 런타임 `master-968-8cd731e`와 헤더 재생성 | `koharu-diffusion-sys` 빌드 성공. `bin/flux2_klein.rs`가 같은 입력과 seed에서 업그레이드 전후 결과가 사실상 같음(최대 차이와 PSNR 기록) |
| G2 | `bin/flux3_klein.rs --caption-only` | llama `b11552`가 `qwen3_5`와 mmproj를 로드하고 i2i 캡션을 생성함. 실패하면 llama 런타임 최신화를 별도 선행 작업으로 분리하고, 이 증분은 그 뒤로 미룬다 |
| G3 | `bin/flux3_klein.rs` 마스크 인페인팅 | 컨텍스트 생성 성공. 마스크 바깥 픽셀은 overlay 뒤 원본과 같음. 대표 만화 페이지 3–5장으로 FLUX.2와 같은 입력에서 비교(`--no-upsampler` 포함)한 결과를 기록 |
| G4 | 대상 12GB GPU 실측 | Phase 1과 Phase 2 각각의 VRAM 피크 ≤ 12GB. 페이지당 시간을 단계별로 기록. 장치·입력 크기·기준선(FLUX.2)·결과·차이를 보고 |

- 측정할 때는 가중치 로드와 워밍업을 측정 구간 밖에 둔다. 비동기 실행은 동기화한 뒤에 시간을 잰다.
- G3에서 품질이 FLUX.2보다 나쁘면 출시하지 않는다. 결과는 이 문서에 기록으로만 남긴다.
- G4 수치가 UI 고지 문구와 문서 수치를 확정한다.

구현 순서: G1 → `TilePlan` 분리 → `koharu-ml::flux3_klein`과 디버그 바이너리 → G2·G3·G4
→ 파이프라인 설정·흐름 → UI·i18n → 문서.

## 테스트

debug 프로필로 관련 테스트만 한 번 실행한다.

| 계층 | 내용 |
|---|---|
| `koharu-ml` | `Flux3KleinSource::default()`가 모두 builtin인지. builtin이 위 표의 저장소·파일로 해석되는지. 캡션 메시지 조립(시스템 프롬프트 키, 사용자 템플릿 문자열) 단위 테스트 |
| `koharu-pipeline` | `TilePlan` 분리 회귀(기존 `inpaint_tiled` 테스트를 옮겨 같은 결과인지). `crops()`와 `run()`의 순서·인덱스 일치. `flux3-klein` 섹션 TOML 왕복. `validate` 거부 사례(NUL, 빈 `request`, 잘못된 소스) |
| UI | `flux3-klein`을 고르면 패널과 고지 3종이 렌더링되는지. 기존 preferences 테스트를 확장 |
| 수동 | 모델 실행 검증(G2–G4)은 디버그 바이너리로만 한다. E2E 테스트는 요청이 있을 때만 |

## 범위 밖

| 항목 | 사유 |
|---|---|
| 공용 VRAM↔RAM 레지던시 매니저 | D8. 후속 조사 |
| 캡션 캐시 | 재실행 빈도에 비해 이득이 불분명하다 |
| 업샘플러 끄기 UI | D6 |
| FLUX 3 Image Klein Base(50스텝) | 미세조정용이다. 인페인팅 실행 경로의 이점이 없다 |
| bbox 프롬프트(`i2i_bbox`), 다중 참조 편집 | 인페인팅 마스크 경로와 무관하다 |
| FLUX.2 Klein 기본값 변경 | D1 |
| 비-인페인트 FLUX 3 생성 기능의 UI 노출 | 현재 사용자 기능 범위 밖 |

## 참고

- `black-forest-labs/flux-3-image-klein` 모델 카드 @ `337337621b8526e120f2275d250242f4a6bedd75`
- `black-forest-labs/flux-3-image-klein-pup` @ `3f3edb52b2c55de0d49feab28598b68f7f8e4830`
- stable-diffusion.cpp PR #2143, `8cd731e`, `docs/flux3.md`
- `koharu-rs/diffusion` 릴리스 `master-968-8cd731e`
- BFL 블로그 "FLUX 3 Image Klein" (2026-10-15)
- `specs/2026-10-11-flux2-klein-variant-design.md` (폐기, 9B 평가 기록)
- `specs/2026-10-11-12gb-inpainting-model-survey.md`
- `fork/VRAM-ACCOUNTING-DESIGN.md`
