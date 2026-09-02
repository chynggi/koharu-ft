# 인페인팅 모델 확장 설계

작성일: 2026-08-22
개정: 2026-08-22 — TorchScript 경로 확인에 따라 접근 전면 수정

## 배경

인페인팅 스테이지는 이미 4개 모델을 스위칭한다. `InpaintingModel`
(`koharu-pipeline/src/config.rs:273`)이 `lama` / `aot-inpainting` /
`flux2-klein` / `rorem-mixed`를 태그 유니온으로 갖고,
`stages/inpainting.rs:271`의 `Model` enum이 로드된 인스턴스를 쥔다. IOPaint의
`HDStrategy`·`InpaintRequest`도 `lama/config.rs`에 그대로 옮겨져 있다. 즉
"여러 인페인팅 모델을 한 툴에서 비교한다"는 구조는 이미 존재한다.

막힌 것은 두 가지다. 체크포인트가 상수로 굳어 있고(`lama/mod.rs:17`),
새 아키텍처를 넣는 비용이 비싸 보였다.

### 초판의 오판

초판은 "CNN/GAN 계열은 아키텍처를 Rust로 손수 포팅해야 한다"를 전제로 파이썬
사이드카를 제안했다. 이 전제가 틀렸다. 확인한 사실은 다음과 같다.

**`koharu-torch`는 tch-rs의 인트리 포크이고, TorchScript 실행 경로가 살아 있다.**

- `koharu_torch::CModule`이 `load_on_device` / `forward_ts` / `set_eval`을
  노출한다 (`koharu-torch/src/wrappers/jit.rs:459,498,590`,
  `lib.rs:12`에서 재수출).
- C++ shim 헤더가 `atm_load_on_device`·`atm_forward`·`atm_eval`을 선언하고
  (`koharu-torch-sys/libtch/torch_api.h:198,201,217`), 빌드 산출물
  `target/debug/build/koharu-torch-sys-*/out/torch_api.rs:1364,1409,1500`에
  동적 심볼 로더가 실제로 생성되어 있다. **shim 작업 없이 오늘 쓸 수 있다.**

**그리고 IOPaint 모델의 절반은 애초에 TorchScript로 배포된다.** IOPaint의
로더를 직접 확인한 결과:

| 모델 | IOPaint 로더 | 배포 형식 |
|---|---|---|
| LaMa (`big-lama.pt`) | `load_jit_model` | TorchScript |
| **AnimeLaMa (`anime-manga-big-lama.pt`)** | `load_jit_model` | TorchScript |
| **MI-GAN** | `load_jit_model` | TorchScript |
| **Manga (`manga_inpaintor.jit` + `erika.jit`)** | `load_jit_model` | TorchScript |
| MAT | `torch.load` + `mat.py` 1000줄 아키텍처 | state_dict |
| FCF, ZITS | 동일 | state_dict |
| PowerPaint, BrushNet | diffusers 파이프라인 | 다중 컴포넌트 |

즉 **MI-GAN·만화 파인튜닝 LaMa·망가 전용 인페인터가 아키텍처 포팅 없이,
파이썬 없이 네이티브로 들어온다.** 초판이 사이드카로 사려던 것의 상당 부분이
이미 무료다.

### 다른 런타임을 쓰지 않는 이유

**전제: libtorch 비용은 이미 지불되어 있다.** tch-rs 경로의 통상적인 단점은
"실행 환경에 LibTorch가 설치되어 있어야 해서 단일 바이너리 배포가 어렵다"이다.
Koharu에는 해당하지 않는다. `koharu-runtime`이 libtorch 2.12.1을 PyTorch 휠에서
백엔드별(CUDA / ROCm / CPU / macOS)로 **자동 내려받아 프로비저닝**하고
(`runtime/packages/torch.rs:22,101-203`), `koharu-torch-sys`가 이를 동적으로
로드한다. 사용자 설치 단계가 없다. 따라서 TorchScript 경로의 한계 비용은 **0**이다.

이 전제 위에서 대안들을 본다.

| 대안 | 얻는 것 | 잃는 것 |
|---|---|---|
| **ONNX (`ort` / `tract`)** | 없음 — 위 표의 4종은 TorchScript로 배포되므로 ONNX로 다시 뽑는 작업이 **추가**된다 | 세 번째 네이티브 런타임(libtorch + sd.cpp + ORT)과 두 번째 CUDA 컨텍스트 |
| **candle** | 없음 — TorchScript 아카이브를 실행하지 못한다. safetensors + Rust 아키텍처만 받으므로 4종이 전부 수작업 대상이 된다 | 두 번째 ML 프레임워크와 할당자 |
| **Burn** | Wasm 실행. Koharu는 데스크톱 앱이라 해당 없음 | 위와 같고, 포팅 분량은 최대 |

공통 비용이 결정적이다. 두 번째 CUDA 할당자가 같은 프로세스에 상주하면
`ResourceMonitor`의 VRAM 회계(`fork/VRAM-ACCOUNTING-DESIGN.md`,
`stages/inpainting.rs:210` `device_for_load`)가 한쪽만 보게 되어 무너진다.
3060 12GB에서 탐지·OCR·인페인팅이 순차로 도는 이 파이프라인에서 이 회계는
장식이 아니다.

**결론: tch-rs(=`koharu-torch`)를 그대로 쓴다. TorchScript 로더를 추가한다.**
ONNX는 상류가 TorchScript를 배포하지 않는 모델이 나타났을 때 다시 검토한다.

## 목표

1. LaMa 계열 체크포인트를 설정만으로 교체한다.
2. TorchScript 인페인팅 모델을 로드하는 공용 경로를 만들고, 그 위에 MI-GAN을
   올린다.
3. 파이썬 런타임 의존을 도입하지 않는다.

### 범위 밖

**파이썬 사이드카, PowerPaint, BrushNet.** MI-GAN·AnimeLaMa·Manga가 네이티브로
들어오면 PowerPaint의 우선순위가 내려간다. 디퓨전 인페인팅은 `flux2-klein`과
`rorem-mixed`가 이미 담당한다. 사이드카는 이 작업이 끝난 뒤 실제 품질 공백이
확인되면 별도 문서로 다룬다.

**MAT, FCF, ZITS.** state_dict 배포라 아키텍처 포팅이 필요하다. MAT는 Swin
트랜스포머 + StyleGAN modulated conv로 `mat.py`가 1000줄을 넘는다. 투자 대비
회수가 나쁘다. 필요하면 상류에서 TorchScript로 트레이싱해 배포하는 쪽이 싸다.

**기본값 변경.** `PipelineConfig::default()`의 인페인팅은 `LaMa {}`로 유지한다
(`config.rs:166`).

---

## 1부: TorchScript 로더

### 접근

`koharu-ml/src/torchscript.rs`를 신설한다. `CModule`을 감싸 로드·디바이스
배치·`set_eval`·`forward_ts`를 한곳에 모으고, 텐서 전처리는 각 모델 모듈이
갖는다.

기존 safetensors 경로(`VarStore::load`)와 **공존한다.** 둘은 다른 파일 형식이고
서로를 대체하지 않는다. 이 구분을 타입으로 드러낸다 — 초판은 이 둘을 섞어
`big-lama.pt`를 `VarStore::load`로 읽을 수 있다고 잘못 적었다.

```rust
pub enum Weights {
    /// nn::VarStore로 읽는 state_dict. 아키텍처가 Rust에 있어야 한다.
    SafeTensors,
    /// CModule로 읽는 TorchScript 아카이브. 아키텍처가 파일 안에 있다.
    TorchScript,
}
```

### 검증

`CModule::load_on_device`는 아카이브가 아닌 파일을 받으면 실패한다.
`forward_ts`는 입력 개수·형상이 트레이싱 시점과 다르면 실패한다. 두 실패 모두
로드/첫 추론 시점에 드러나므로, 별도의 형식 판별은 넣지 않고 에러 문맥만
붙인다.

**성공 기준**: `big-lama.pt`를 CUDA 디바이스에 올려 512×512 입력 한 장의
`forward_ts`가 텐서를 반환한다.

---

## 2부: LaMa 체크포인트 소스

### 접근

`Flux2KleinSource`의 패턴을 재사용한다. `flux2_klein/source.rs`의
`ComponentSource`는 내용상 FLUX와 무관한 `Builtin` / `LocalFile` /
`HuggingFace` 세 갈래이므로, `koharu-ml/src/source.rs`로 끌어올려 공용화하고
`flux2_klein`은 `pub use`로 재수출해 기존 경로를 유지한다.

`LaMaConfig`는 소스와 함께 **형식**을 갖는다. 기본값은 현행 유지 —
`mayocream/lama-manga` + `SafeTensors`. `TorchScript`를 고르면 `CModule` 경로로,
`SafeTensors`면 기존 `FFCResNetGeneratorConfig` 경로로 간다. 전처리
(`lama/processor.rs`의 HD 전략·패딩)는 양쪽이 공유한다.

이 한 번의 작업으로 원본 `big-lama.pt`, **`anime-manga-big-lama.pt`**,
`lama_mpe`가 전부 설정만으로 들어온다.

### 변경 사항

| 파일 | 변경 |
|---|---|
| `koharu-ml/src/source.rs` | 신설. `ComponentSource` 이동 |
| `koharu-ml/src/flux2_klein/source.rs` | `pub use crate::source::ComponentSource` |
| `koharu-ml/src/lama/mod.rs` | `load(device, source, format)`. 내부에 두 백엔드 |
| `koharu-pipeline/src/config.rs` | `LaMa {}` → `LaMa(LaMaConfig)` |
| `koharu-pipeline/src/stages/inpainting.rs` | 소스 전달, `validate()` |
| `packages/ui` | LaMa 소스·형식 선택 UI |

### 하위 호환

`InpaintingModel::LaMa {}`는 빈 구조체 변형이므로 `LaMaConfig`의 모든 필드를
`#[serde(default)]`로 두면 기존 `koharu.toml`이 수정 없이 읽힌다. 테스트로
고정한다.

**성공 기준**

1. 기존 설정 파일이 수정 없이 로드되고 `lama-manga`를 계속 쓴다.
2. `anime-manga-big-lama.pt`를 TorchScript로 지정해 페이지 하나가 인페인팅된다.
3. 형식을 잘못 고르면 로드 시점에 실패하고 메시지에 파일 경로가 들어간다.
4. 상대 경로·부재 파일·잘못된 리비전은 설정 저장 시점에 거부된다
   (`ComponentSource::validate` 기존 테스트가 커버).

---

## 3부: MI-GAN

### 접근

`koharu-ml/src/mi_gan/`을 신설한다. 모델 코드는 없다 — TorchScript 로더 위의
전처리·후처리만이다. IOPaint의 `mi_gan.py`가 요구하는 제약을 그대로 따른다:

- `min_size = 512`, `pad_mod = 512`, `pad_to_square = True`
- 512×512면 그대로, 아니면 마스크 바운딩 박스로 crop 후 512로 리사이즈

`InpaintingModel::MiGan(MiGanConfig)`를 추가한다. `MiGanConfig`는 소스만 갖는다
(프롬프트 없음, LaMa와 같은 성격).

### 성공 기준

1. 말풍선 하나를 지운 결과 PNG가 나온다.
2. 벤치(`koharu-ml/benches/mi_gan.rs`)에서 LaMa 대비 페이지당 지연 시간과
   피크 VRAM이 기록된다. MI-GAN이 더 빠르고 가볍다는 주장이 이 수치로
   확인되거나 반증된다.

---

## 작업 순서

1. `koharu-ml/src/torchscript.rs` + `CModule` 스모크 테스트
   → 검증: `big-lama.pt` 로드·`forward_ts` 성공
2. `ComponentSource`를 `koharu-ml/src/source.rs`로 이동
   → 검증: 기존 FLUX 테스트 통과
3. `LaMa::load`가 소스·형식을 받게 변경, 전처리 공유
   → 검증: `lama` 벤치·바이너리 동작, safetensors 경로 무회귀
4. `InpaintingModel::LaMa(LaMaConfig)` + 하위 호환 테스트
   → 검증: 기존 toml 로드
5. UI 소스·형식 선택 패널
   → 검증: `anime-manga-big-lama.pt`로 페이지 하나 인페인팅
6. `mi_gan` 모듈 + 파이프라인 배선 + `bin`
   → 검증: 말풍선 제거 결과물
7. `benches/mi_gan.rs`
   → 검증: LaMa 대비 지연 시간·VRAM 수치

1~5는 2부, 6~7은 3부다. 1은 둘 다의 선행 조건이다.

## 열린 질문

- ~~**Manga 인페인터(`manga_inpaintor.jit` + `erika.jit`)를 넣을 것인가.**~~
  **결정(2026-08-22): 넣는다.** 구현 완료. 4K CPU 벤치에서 LaMa 11.9s 대비
  65.2s로 세 모델 중 가장 느리지만, 망가 선화 복원 품질이 목적이므로
  속도는 수용한다. 시드 42 고정으로 실행 간 결과가 byte-identical함을
  실측으로 확인했다.
- **체크포인트 재배포.** IOPaint 체크포인트는 GitHub 릴리스에서 오고
  `HuggingFaceFile`은 HF만 받는다. `mayocream/` 아래로 미러링할지, 소스 갈래에
  URL 변형을 추가할지 정해야 한다.
