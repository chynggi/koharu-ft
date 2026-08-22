# HANDOFF — 인페인팅 모델 확장 (`feat/inpainting-torchscript-migan`)

최종 갱신: 2026-08-22 심야 (**계획 Task 0~10 완료 + 설계 문서의 열린 질문이던 Manga 인페인터도 구현 완료.**)

## Manga 인페인터 (`manga_inpaintor.jit` + `erika.jit`) — 추가 세션

설계 문서 "열린 질문"의 마지막 미결정 항목을 사용자 결정으로 넣었다. 네 커밋:

- `c15b92db feat(ml): add the Manga inpainter`
- `c7e75ef6 feat(pipeline): select the Manga inpainter`
- `eb4b9292 feat(ui): expose the Manga inpainter as an inpainting model`
- `f37457d7 bench(ml): compare the Manga inpainter against LaMa and MI-GAN` (이 문서 갱신 포함)

### 구현 노트

- **HD 전략 디스패치 공용화.** 상류 `manga.py`는 LaMa와 같은 base 클래스
  `__call__`을 쓴다(`pad_mod=16`만 다름). LaMa 프로세서의 crop/resize/original
  디스패치를 `inpaint_ops::dispatch_hd_strategy`로 끌어올려 두 모델이 공유한다.
  MI-GAN은 자체 `__call__`이라 그대로다.
- **2-모델 소스.** `MangaSource { inpaintor, line }`은 FLUX의
  `Flux2KleinSource` 패턴을 따른다.
- **시드 재현성.** 상류는 forward마다 `self.seed = 42`로 리셋한다.
  `manual_seed(42)` + (CUDA 시) `Cuda::manual_seed_all(42)`로 맞췄다.
- **그레이 변환.** OpenCV `COLOR_RGB2GRAY`(BT.601 고정소수) 계수를 직접
  구현했다. `image` 크레이트의 luma 변환은 계수가 달라 상류와 byte가 어긋난다.
- **출력은 그레이스케일.** 마스크 영역이 흑백으로 채워진다(만화용 설계).
  `sd_keep_unmasked_area` 블렌드로 마스크 밖은 원색 보존.

### 완료 검증 (전부 실측)

| 항목 | 결과 |
|---|---|
| 체크포인트 MD5 (IOPaint 핀값과 대조) | `manga_inpaintor.jit` `7d8b269c...`, `erika.jit` `0c926d5a...` — 둘 다 일치. BLAKE3로 리모트 저장소 핀 |
| 실제 체크포인트로 5입력 forward, modulo-16 패딩, 시드 재현성 | **1 passed** (ignored 테스트, `KOHARU_MANGA_INPAINTOR_TS`/`KOHARU_MANGA_LINE_TS`로 실행). 두 실행이 byte-identical |
| `cargo test -p koharu-ml --lib -- --test-threads=1` | **85 passed, 0 failed, 20 ignored** (LaMa HD 디스패치 리팩터링 후 무회귀) |
| `cargo test -p koharu-pipeline` | **77 passed, 0 failed** (신규 3: 라운드트립, 기본값 고정, validate 실패 — Task 8 패턴) |
| tsc `--noEmit` | 통과. 이번엔 로케일 신규 키 2개(`source.inpaintor`, `source.line`)를 9개 전 로케일에 추가했다 |
| CLI 스모크 (4K, 로컬 가중치, CPU) | 출력 PNG 정상 생성 (16.2MB) |
| `cargo bench -p koharu-ml --bench manga_inpaintor` | **65.232s (62.4~68.6s), 10 샘플** — 세 모델 중 가장 느림. LaMa 대비 ~5.5×, MI-GAN 대비 ~80×. 구조적 이유: 두 모델이 패딩된 전체 해상도를 통과. README에 기록 |

### 사용자 확인이 필요한 것

- **앱 UI 경로의 Manga 인페인터 엔드투엔드 검증.** MI-GAN UI 경로는
  사용자가 이미 검증했지만 Manga 쪽은 아직이다. 첫 실행 시 245MB
  (72.5+173MB) 다운로드가 발생한다.
- `package.json`의 `release` 스크립트 한 줄은 사용자 로컬 수정으로 보존 중
  (커밋하지 않음).

---

아래는 Task 7~10 완료 시점의 이전 핸드오프 내용이다.

최종 갱신: 2026-08-22 저녁 (**Task 7~10 완료. 계획의 모든 태스크 종료.**)

Task 7~10은 2026-08-22 저녁 세션에서 완료됐다. 다음 네 커밋이다:

- `98b15485 feat(ml): add MI-GAN inpainting` — Task 7
- `36e4e312 feat(pipeline): select MI-GAN for inpainting` — Task 8
- `b6ceccf6 feat(ui): expose MI-GAN as an inpainting model` — Task 9
- `e8b70061 bench(ml): compare MI-GAN against LaMa` — Task 10

## 완료 검증 (전부 이번 세션 실측)

| 항목 | 결과 |
|---|---|
| `pad_to_square` 가정 (계획서 마지막 미해결 위험) | **실측으로 해소.** 실제 `migan_traced.pt`(MD5 `76eb3b...` IOPaint 핀값 일치)로 300×200 비정사각 입력이 512×512 패딩을 통해 모델을 통과하고 출력이 원본 크기로 돌아오며 마스크 밖 픽셀이 보존됨. `pad_forward`에 `ensure!`로 512×512 정사각을 강제해 어긋나면 명시적 실패 |
| `cargo test -p koharu-ml --lib -- --test-threads=1` | **84 passed, 0 failed, 19 ignored** (Task 6 기준 83 + MI-GAN 신규 1) |
| `cargo test -p koharu-ml mi_gan --lib -- --ignored --test-threads=1` (실제 체크포인트) | **1 passed** |
| MI-GAN CLI, 4K 픽스처, 로컬 가중치, CPU | 출력 PNG 정상 생성 (14.8MB) |
| `cargo test -p koharu-pipeline` | **74 passed, 0 failed** (기존 71 + Task 8 신규 3: 라운드트립, 기본값 고정, validate 실패 경로 — Task 4의 두 함정을 처음부터 넣음) |
| `bunx --bun tsc --noEmit -p tsconfig.json` | 통과 |
| `cargo bench -p koharu-ml --bench lama --bench mi_gan` | **LaMa 11.919s vs MI-GAN 821.78ms (약 14.5× 빠름)**, 둘 다 CPU(i5-10400), `benches/fixtures/inpaint/README.md`에 기록 |

## 계획서 스니펫에서 발견·수정한 결함 (Task 7)

계획서 경고대로("스니펫은 제안이지 성경이 아니다") 상류 원문(`mi_gan.py` 핀 커밋 61a759f)과 대조해 두 결함을 수정했다:

1. **`resize_max_size`는 축소 전용이다.** 상류는 `max(h,w) > 512`일 때만 리사이즈한다. 계획 스니펫의 무조건 `resize_dimensions` 호출은 512 이하 crop을 업스케일하는 발산이었다. `call`에서 `max > SIZE`일 때만 리사이즈하도록 고쳤다.
2. **`_pad_forward`의 `sd_keep_unmasked_area` 블렌드가 빠져 있었다.** 상류는 `result*(mask/255) + image*(1-mask/255)`를 적용한다. LaMa의 `pad_forward`와 같은 구조로 넣었다. (crop 경로에서는 외부 paste 복원이 이미 이를 대신하지만, 512×512 직행 경로와 `sd_keep_unmasked_area=false` 갈래에 영향이 있다.)
3. (사소) `Tensor::cat`은 `&[Tensor]`를 받으므로 `erased`에서 `&`를 뗐다.

Task 9의 지시대로 `ComponentSourceField`/`emptySource`를 `Flux2KleinOptions.tsx`에서 `PreferenceFields.tsx`로 옮겼다(세 번째 소비자 MI-GAN 추가 시점).

## 사용자 확인이 필요한 것 (에이전트가 못 하는 일)

- **실제 앱에서의 엔드투엔드 검증.** `bun run dev`로 인페인팅 모델을 MI-GAN으로 바꿔 한 페이지를 처리. 26MB 다운로드와 데스크톱 세션이 필요해 이번 세션에서는 돌리지 못했다. CLI·테스트·벤치 경로는 전부 실측 통과했지만 앱 UI 경로는 미검증이다.
- Task 5와 마찬가지로 이 포크에는 바인딩 자동 생성 바이너리가 없으므로 `protocol.ts`는 Rust 선언과 직접 대조해 수동 갱신했다 (`InpaintingModel`의 `mi-gan` 갈래, `MiGanConfig`, `ProcessorConfig`의 `mi-gan` 필드).
- 로케일 신규 문구는 불필요했다 — MI-GAN UI는 기존 키(`checkpointSource`, `sourceKind.*`, `format.*`)만 재사용한다.

---

아래는 Task 6 종료 시점의 이전 핸드오프 내용이다.

최종 갱신: 2026-08-22 (**Task 0~6 완료. LaMa 기능 완결. 다음은 Task 7 — MI-GAN**)

**중단 사유: 5시간 사용 한도 소진.** 코드 결함이나 막힌 문제 때문이 아니다.
워킹 트리 깨끗, 모든 테스트 통과, 진행 중이던 서브에이전트 없음.

**아래 표의 "확인 방법"은 재검증용이다. 다음 세션은 이 문서를 믿기 전에 그 명령을 다시
돌릴 것.** 특히 §5의 미검증 항목은 아직 아무도 실행해 보지 않은 코드에 대한 주장이다.

브랜치: `feat/inpainting-torchscript-migan`, `main` 위 8커밋. 워킹 트리 깨끗함.

관련 문서:
- 설계: `specs/2026-08-22-inpainting-models-design.md`
- 계획: `plans/2026-08-22-inpainting-models.md` (Task 0~10, 실행 레시피 포함)

---

## 1. 무엇을 하려는 작업인가

인페인팅 모델을 늘리는 작업이다. 두 가지가 목표다.

1. **LaMa 체크포인트를 설정으로 교체** — 지금은 `lama/mod.rs`가 `mayocream/lama-manga`를
   리비전까지 상수로 굽고 있어 원본 big-lama도, 만화 파인튜닝도 바꿔 낄 수 없다.
2. **MI-GAN 추가** — 3060 12GB급에서 빠르고 가벼운 소거 전용 모델.

### 접근이 한 번 크게 바뀌었다 — 이 경위를 모르면 잘못된 방향으로 되돌아가기 쉽다

**초판 설계는 파이썬 사이드카였다.** "CNN/GAN 계열은 아키텍처를 Rust로 손수 포팅해야
한다"를 전제로, IOPaint를 파이썬 프로세스로 띄워 PowerPaint 등을 쓰자는 안이었다.

**그 전제가 틀렸다.** 사용자가 candle/tch-rs 검토를 지시해 확인한 결과:

- `koharu-torch`는 **tch-rs의 인트리 포크**이고 **TorchScript 실행 경로가 살아 있다**
  (`CModule::load_on_device` / `forward_ts` / `set_eval`, `wrappers/jit.rs:459,498,590`).
- libtorch는 `koharu-runtime`이 **자동 프로비저닝**한다
  (`runtime/packages/torch.rs:22,101-203`). 사용자 설치 단계가 없으므로 tch 경로의
  통상적 단점("LibTorch를 깔아야 해서 단일 바이너리 배포가 어렵다")이 여기선 성립하지
  않는다. 한계 비용이 0이다.
- **IOPaint 모델의 절반은 이미 TorchScript로 배포된다.** `load_jit_model`을 쓰는 것들:
  `big-lama.pt`, `anime-manga-big-lama.pt`, `migan_traced.pt`,
  `manga_inpaintor.jit`+`erika.jit`. state_dict라 아키텍처 포팅이 필요한 것은 MAT·FCF·ZITS뿐.

따라서 **파이썬 사이드카도, ONNX(`ort`/`tract`)도, candle도, Burn도 쓰지 않는다.**
ONNX로 가면 이미 존재하는 TorchScript를 버리고 다시 뽑는 작업이 *추가*되고, 어느 쪽이든
libtorch·sd.cpp에 이은 **세 번째 네이티브 런타임과 두 번째 CUDA 할당자**가 생겨
`ResourceMonitor`의 VRAM 회계가 무너진다. 근거는 설계 문서 "다른 런타임을 쓰지 않는 이유"에 있다.

**이 판단은 실증되었다** (§3 참조). 되돌리려면 그 증거부터 반박해야 한다.

---

## 2. 완료된 것 (2026-08-22 확인)

| Task | 항목 | 상태 | 확인 방법 |
|---|---|---|---|
| 0 | `RemoteFile` — 임의 URL을 BLAKE3로 못박아 가져오기 | 완료, 리뷰 2종 통과 | `cargo test -p koharu-runtime` → 21 passed |
| 0 | `PinnedFile` — HF/URL 두 소스를 하나로 받는 enum | 완료 | `koharu-runtime/src/source/mod.rs` |
| 0 | 상류 체크포인트 3종 BLAKE3 실측 | 완료 | §4 표 |
| 1 | `TorchScript` — `CModule` 얇은 래퍼 | 완료, 리뷰 2종 통과 | `cargo test -p koharu-ml --lib torchscript -- --test-threads=1` |
| 1 | **실제 `big-lama.pt`로 동작 실증** | 완료 | §3 |
| 2 | `ComponentSource`를 `flux2_klein` → 크레이트 공용으로 이동 | 완료, 리뷰 2종 통과 | `crates/koharu-ml/src/source.rs` |
| 2 | `ComponentSource::Url { url, digest }` 갈래 추가 | 완료, 리뷰 2종 통과 | `a_url_override_requires_a_digest` |
| 2 | `remote_repository!` 매크로 | 완료 (Task 3이 첫 사용, `#[expect]` 제거됨) | `koharu-ml/src/lib.rs` |
| 3 | `Backend` — safetensors/TorchScript 두 형식 | 완료, 리뷰 2종 + 수정 반영 | `cargo test -p koharu-ml --lib -- --test-threads=1` |
| 3 | `WeightsFormat` 설정 갈래 + CLI `--torchscript`/`--weights` | 완료 | `lama/config.rs`, `bin/lama.rs` |
| 3 | **4K 실사진으로 두 경로 무회귀 실측** | 완료 | 아래 |
| 4 | `InpaintingModel::LaMa(LaMaConfig)` 설정 배선 | 완료, 리뷰 2종 + 수정 | `cargo test -p koharu-pipeline` → 71 passed |
| 4 | 기존 `koharu.toml` 무손상 | 완료 | `an_existing_file_without_a_lama_section_keeps_the_builtin_checkpoint` |
| 5 | LaMa 설정 UI + 로케일 9종 | 완료, 리뷰 2종 + 수정 | `cd packages/koharu && bunx --bun tsc --noEmit -p tsconfig.json` |
| 5 | `protocol.ts` 와이어 타입 정합 | 완료 | Rust 선언과 직접 대조 |
| 6 | 전처리 헬퍼 9종을 `inpaint_ops.rs`로 추출 | 완료, 리뷰 통과 | `cargo test -p koharu-ml --lib -- --test-threads=1` → 83 passed |
| 6+ | 공용 헬퍼 특성화 테스트 18건 | 완료 | 변조로 물림 확인 |

커밋 (오래된 것부터):

```
150b5cda docs: add the inpainting model expansion spec and plan
e710da93 docs(plans): pin the upstream checkpoints by measured BLAKE3 digest
fc5b0de8 feat(runtime): resolve pinned files from arbitrary URLs        ← Task 0
3175d786 docs(plans): settle the digest case policy and flag unverified plan code
f7b6ed6e feat(ml): add a TorchScript archive loader                     ← Task 1
0c626509 docs(plans): record the Windows DLL path recipe and the TorchScript proof
d48ffa05 refactor(ml): lift ComponentSource out of flux2_klein and add a URL source  ← Task 2
84891ac1 docs(plans): koharu-ml tests must run single-threaded
```

---

## 3. 이 작업 전체를 지탱하는 증거

Task 1의 무시 테스트가 **실제 197MB `big-lama.pt`를 로드해 추론을 돌린다.**

```bash
LA=$(cygpath -u "$LOCALAPPDATA")
export PATH="$LA/koharu:$LA/koharu/packages/torch/2.12.1/cpu/libtorch/lib:$PATH"
export KOHARU_BIG_LAMA_TS=/path/to/big-lama.pt
cargo test -p koharu-ml --lib torchscript -- --ignored
```

```
test torchscript::tests::big_lama_accepts_an_image_and_a_mask ... ok
finished in 5.29s
```

`[1,3,512,512]` 이미지 + `[1,1,512,512]` 마스크 → 출력 `[1,3,512,512]`. 코디네이터가
구현자와 독립적으로 재현했다. **아키텍처 포팅 없이 TorchScript 모델이 Rust에서 돈다는
것이 확인된 사실이다.** 여기가 무너졌다면 Task 3·7이 전부 무의미하고 파이썬 사이드카로
돌아가야 했다.

---

## 4. 체크포인트

미러링하지 않고 **상류 원본에서 직접** 받는다(사용자 결정 — 이 저장소는 포크이므로
`mayocream/` 재배포 관행을 따를 이유가 없고, 미러는 조용히 낡는다).

| 파일 | 크기 | BLAKE3 | URL |
|---|---|---|---|
| `migan_traced.pt` | 26 MB | `fde1e5f7c6b6a48082f8eff36b9117e64b8c14ea4d1a76af508e29d357b28cbd` | `https://github.com/Sanster/models/releases/download/migan/migan_traced.pt` |
| `anime-manga-big-lama.pt` | 197 MB | `9213532a6e9990afcd0c9f3f31da82cc4c8c1ec86a13641e3ec37648d5e75f8b` | `https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt` |
| `big-lama.pt` | 197 MB | `1e3e5989dae88d561f1c8e8456c8fe9595739aaa9898862d004abf192c6d9e76` | `https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt` |

`migan_traced.pt`의 MD5가 IOPaint 고정값 `76eb3b1a71c400ee3290524f7a11b89c`
(`iopaint/model/mi_gan.py:21`)와 일치함을 교차 확인했다.

세 URL 모두 2026-08-22 기준 200으로 응답한다. **다음 세션은 다시 받아야 한다** — 이번
세션이 받아둔 사본은 임시 스크래치패드에 있어 job과 함께 사라진다.

---

## 5. 아직 안 된 것

### Task 2 리뷰 결과 (2026-08-22, 스펙·품질 2종 모두 PASS, must-fix 0건)

두 리뷰어 모두 `--test-threads=1`로 직접 재실행해 기준선(60/2/17) 일치를 확인했다.
clippy도 이 diff와 무관한 기존 `large_enum_variant` 경고 외에 깨끗하다.

- 이동 시 유실 없음. derive·`validate()` 로직·에러 문자열 보존, 테스트 3건 온전히 이동.
- `flux2_klein::ComponentSource` 경로가 재수출로 살아 있어 기존 호출처 호환.
- `Url` 갈래는 `resolve`·`validate` 둘 다 `RemoteFile`에 위임 — 다이제스트 로직 중복 없음.

구현자 자진 신고 3건 판정:

1. **테스트 단언** — 실제 메시지는 `remote.rs:50`의 `"digest must be 64 hex characters: {}"`.
   계획서의 느슨한 `contains("64 hex")`보다 나은 수정. **채택.**
2. **`#[expect(unused_macros)]` — 유지 결정.** 워크스페이스에 `#[allow]` 792건 대
   `#[expect]` 1건(이것)으로 선례가 없는 건 사실이나, 여기선 `#[allow]`보다 우월하다.
   Task 3이 매크로를 쓰는 순간 이 속성이 스스로 빌드를 깨뜨려 제거를 강제하기 때문이다.
   **→ Task 3 구현자는 `remote_repository!` 첫 사용과 함께 이 `#[expect]`를 지워야 한다.**
3. **`packages/bridge/src/protocol.ts`** — 수작업 타입이 맞고(파일 헤더로 확인),
   `ComponentSourceConfig`(399행)에 `url` 갈래가 없다. 계획서 Task 2 범위에 `packages/bridge`가
   없으므로 **누락이 아니라 정상 이월**. Task 5·9(UI)에서 처리한다.

품질 리뷰가 제기한 "`remote_repository!`는 호출처 0개인 투기적 추상화" 지적은 **기각**한다.
리뷰어가 커밋 diff만 보고 판단했으나 계획서 812행(Task 3, LaMa 2종)·1341행(Task 7, MI-GAN)에
호출 블록이 있어 사용처는 2곳이다. `model_repository!`와 대칭이므로 존치한다.

### Task 3 리뷰 결과 (스펙 PASS / 품질 CHANGES NEEDED → 수정 완료)

핵심 검증: `processor.rs` diff가 **정확히 3곳**(import, 매개변수 타입 2곳, `?` 하나)뿐이라
**전처리가 비트 단위로 불변**임이 증명됐다. 이 태스크의 위험은 사실상 그것 하나였다.

엔드투엔드 실측(4K 픽셀 diff, 마스크 내부/외부):
safetensors Δ14.1 / Δ0.01, TorchScript Δ14.2 / Δ0.01 — 둘 다 마스크 안쪽만 칠한다.

품질 리뷰 must-fix 2건은 `38784b44`로 반영:

1. **`Box<dyn Backend>` → `enum Backend`.** 변형이 컴파일 타임에 정확히 둘이고 호출당 1회
   디스패치라 동적 디스패치가 불필요했다. 결정적 이유는 따로 있다 — 옛 `impl`의
   `Ok(Model::forward(self, ..))`가 **반드시 UFCS여야 했다.** `self.forward(..)`로 쓰면
   트레이트 메서드로 조용히 무한 재귀하고 컴파일도 경고도 통과한다. enum에선 이 함정이
   존재할 수 없다. (`clippy::large_enum_variant` 때문에 `SafeTensors(Box<Model>)`로 박싱.)
2. **TorchScript 테스트 단언이 거짓 통과했다.** `contains("garbage.pt")`는 경로가 양쪽
   에러 경로에 다 나와서 `resolve`가 실패해 로더가 안 돌아도 초록불이었다.
   `"failed to load TorchScript archive"`로 교체. 구현자가 존재하지 않는 경로로 일회용
   테스트를 만들어 옛 단언이 거짓 통과함을 실증한 뒤 고쳤다.

**보류한 리뷰 제안 1건:** `WeightsFormat`을 `config.rs`에서 `backend.rs`로 옮기자는 제안.
논거는 맞지만(`config.rs`는 이식된 IOPaint YAML용, `WeightsFormat`은 Koharu 로더 노브)
Task 4가 같은 파일에 LaMa 설정을 배선하므로 그 모양을 본 뒤 판단한다.

### Task 4 리뷰 결과 (스펙·품질 모두 PASS, 후속 수정 2건 반영)

`924d6c7c` 구현, `05c2941e`·`9dae6140` 코디네이터 후속. 워크스페이스 빌드 복구됨.
`koharu-pipeline` **71 passed / 0 failed**.

하위 호환이 이 태스크의 핵심이었고 코드로 추적해 확인했다: `Serialize`의 `match`가
`processor.lama`를 채우는 것은 **`LaMa(config)` 팔 안에서뿐**이라, LaMa가 아닌 모델을 쓰는
사용자의 파일에 `[processor.lama]`가 새로 생기지 않는다. `flux2_klein`·`rorem_mixed`와 동일 패턴.

**리뷰가 잡아낸 테스트 구멍 2건 — 둘 다 변이로 실재를 확인한 뒤 고쳤다:**

1. **기본값이 고정돼 있지 않았다.** 구현자가 `assert_eq!(lama.source, ComponentSourceConfig::Builtin)`을
   `LaMaConfig::default().source`와의 비교로 바꿔, 기본값이 `Builtin`에서 벗어나도 초록이었다.
   바꾼 이유 자체는 정당했다 — `mod stages`·`mod inpainting`이 둘 다 비공개라 쓰이지 않는
   `pub use`는 실제로 경고가 난다. 그래서 재수출 대신 **타입이 사는 모듈**에
   `lama_defaults_reproduce_the_previous_checkpoint`를 추가했다(형제 Flux 테스트 옆).
2. **`LaMaConfig::validate()`의 실패 경로가 완전히 무방비였다.** 본문을 `Ok(())`로 바꿔도
   스위트 전체가 초록이었다. `an_invalid_lama_source_is_rejected` 추가.

**포맷:** `resources/bytes.rs`·`vram.rs`·`stages/inpainting.rs`·`stages/mod.rs`는 **main에서 이미
미포맷**이다(worktree로 대조 확인). 이번 태스크가 더럽힌 `config.rs`·`bin/run.rs`만 포맷했다.
`cargo fmt -p koharu-pipeline`을 통째로 돌리지 말 것 — 무관한 4개 파일이 diff에 섞인다.

**`WeightsFormat` 위치 문제 종결:** `lama/config.rs` 유지. 파이프라인 쪽 `WeightsFormatConfig`에서
`Into`로 넘기는 배선에 마찰이 없었다. 다만 그 파일 doc 헤더의 "IOPaint YAML" 설명은 이 enum에
맞지 않으니 주석만 손볼 여지가 있다.

### Task 5 리뷰 결과 (스펙·품질 모두 PASS, 후속 수정 1건)

`eb367c2d` 구현, `1d792eaf` 후속. **계획서 Step 1이 틀렸다** — specta가 `protocol.ts`를
생성하니 바인딩 생성기를 다시 돌리라고 하지만, 그 파일은 **수작업 TypeScript**다
(헤더에 명시). 손으로 편집하는 게 맞다. 계획서 결함 7건째.

**와이어 포맷이 이 태스크의 핵심 위험이었다** — TS와 Rust는 따로 컴파일되므로 serde 문자열이
어긋나도 양쪽 다 빌드를 통과하고 런타임에 터진다. Rust 선언과 직접 대조해 확인:
`#[serde(tag = "kind", rename_all = "snake_case")]`의 4갈래, `"safe_tensors" | "torch_script"`,
`{ model: "lama" } & LaMaConfig`, `ProcessorConfig.lama`. 리뷰어가 UI→`models.ts`→
`PipelineConfig` serde 봉투→`ComponentSourceConfig::Url`→서버측 다이제스트 검증까지 완주 추적.

**기존 와이어 구멍 하나가 드러났다.** `protocol.ts`의 `ProcessorConfig`에 `lama` 필드가
**이 브랜치 이전부터 없었다**(`eb367c2d^`로 확인). Rust에는 있었으므로 프론트엔드가 쓴 LaMa
프로필 설정이 조용히 버려지고 있었다. 이번에 메웠다.

**범위가 FLUX로 번졌고, 정당하다고 판정됐다.** `Url` 갈래 추가로 `Flux2KleinOptions.tsx`의
exhaustive switch가 `TS2366`으로 깨졌다. `default: throw`로 좁게 막으면 FLUX는 3갈래로 남지만
**LaMa 패널이 같은 위젯을 복제**해야 하고, 그건 계획서 Step 3이 금지한 것이다. FLUX가 URL
소스를 얻은 것은 FLUX 전용 코드 0줄로 따라온 부수 효과다.

**돌리지 못한 것:** 계획서 Step 5(`bun run dev`로 TorchScript 전환 후 실제 인페인팅).
197MB 다운로드와 데스크톱 세션이 필요하다. **통과했다고 주장하지 않았다.** 타입과 검증
로직은 정합이 확인됐으나 네트워크 실패·다이제스트 불일치 시의 UX는 미검증이다.

**Task 9(MI-GAN UI)에 미룬 것:** `ComponentSourceField`/`emptySource`가 형제 기능 파일
`Flux2KleinOptions.tsx`에 살고 있어 LaMa가 거기서 import한다. 공용 편집기의 관례적 위치는
`PreferenceFields.tsx`다. **세 번째 소비자(MI-GAN)가 생기는 Task 9이 옮길 시점이다.**

**사용자 확인이 필요한 것:** tr-TR "Biçim", ru-RU "Дайджест"는 기계 번역이다.
별건으로, `sourceKind.builtin`/`localFile`이 **ko-KR을 제외한 전 로케일에서 영어**로 남아
있다 — 이 브랜치와 무관한 기존 상태라 손대지 않았다.

### Task 6 결과 (리뷰 1종 PASS — 순수성을 기계적으로 증명해 2종은 낭비였다)

`ba665623` 이동, `6b458e79` 테스트 추가.

**이동이 순수함을 증명했다.** diff를 정규화해(`pub(crate) ` 접두사 제거 후 제거 텍스트 대
추가 텍스트 대조) 남은 차이는 셋뿐 — 새 모듈 doc 주석, import 조정, 승인된 `post_process`
문맥 문자열(`"...LaMa tensor..."` → `"...output tensor..."`). **아홉 함수 본문 전부 바이트 동일.**
`processor.rs`에 `cfg(test)`가 애초에 0건이라 흘린 테스트도 없다.

계획서 누락 하나: `symmetric_indices`가 부르는 비공개 `symmetric_index`도 같이 옮겨야 한다
(떼어낼 수 없음). 고아가 된 import는 `fast_image_resize`, `imageproc::contours`,
`anyhow::Context`, `anyhow::anyhow`.

**API가 MI-GAN에 그대로 맞는다** — 리뷰어가 계획서 Task 7 코드와 대조해 확인했다.
시그니처 변경 불필요. `pad_img_to_modulo`가 `modulo`를 인자로 받아 LaMa의 `8`과 MI-GAN의
`512`가 같은 함수를 쓴다.

**특성화 테스트 18건을 추가했다 (`6b458e79`).** 이유가 중요하다: 유일한 수치 검증이 4K
전체 이미지의 **평균 픽셀 델타**인데, 반사 경계 off-by-one 같은 국소 오류를 잡기엔 둔감하다.
MI-GAN이 올라가면 같은 버그가 두 모델을 동일하게 오염시켜 모델 간 대조로도 안 드러난다.
실용적으로는 **Task 7에서 텐서 연산이 틀어졌을 때 공용 헬퍼가 옳음을 알아야 버그를 MI-GAN
쪽으로 좁힐 수 있다.**

물림 확인: `symmetric_index`의 `length * 2 - index - 1`에서 `- 1`을 빼자 테스트 4건이 실패했다
(`one_past_the_end_duplicates_the_last_in_range_index`,
`the_far_end_of_the_period_reflects_back_to_the_start`,
`symmetric_indices_pads_the_far_side_by_mirroring_inward`,
`a_misaligned_tensor_is_padded_on_the_far_side_by_reflection`). 복원 후 통과.

**주의 — 이 테스트들은 특성화(characterization)다.** 현재 동작을 고정한 것이지 "옳은 동작"을
규정한 게 아니다. 리팩터 중 동작 변화를 잡는 것이 목적이므로, 실패하면 먼저 **의도한 변경인지**
따질 것.

**남은 문서 흠 1건:** `inpaint_ops.rs`의 doc 주석이 "used by LaMa and MI-GAN"인데 MI-GAN이
아직 없다. Task 7이 사실로 만든다.

### 남은 Task (계획서 참조)

| Task | 내용 | 모델 배정(제안) |
|---|---|---|
| 7 | MI-GAN 모듈 (텐서 연산) | opus |
| 8 | MI-GAN 파이프라인 배선 | sonnet |
| 9 | MI-GAN UI | sonnet |
| 10 | 벤치 — LaMa 대비 지연 시간·VRAM | sonnet |

---

## 5.5 다음 세션이 할 일 (우선순위 순)

1. **Task 7 — MI-GAN 모듈 (opus).** 계획서 1299~1539행. 이 계획에서 가장 무겁고,
   **미검증 추론이 남아 있는 유일한 곳**이다. `pad_to_square` 가정을 반드시 실측으로
   검증할 것: `resize_dimensions`가 긴 변을 512로 맞추고 modulo 512 패딩이 짧은 변을
   512로 올려 결국 512×512가 된다는 추론이다. **출력 텐서 크기를 찍어 확인**하고,
   틀리면 계획서를 고쳐 보고할 것. 체크포인트는 §4의 `migan_traced.pt`(26MB)이며
   다시 받아야 한다(이전 사본은 임시 디렉터리와 함께 사라졌다).
2. **Task 8 — MI-GAN 파이프라인 배선 (sonnet).** 계획서 1540행. Task 4와 같은 모양이다.
   Task 4에서 걸린 두 함정을 그대로 확인할 것: 기본값을 고정하는 테스트가 있는가,
   `validate()` 실패 경로에 테스트가 있는가. Task 4에선 둘 다 없었다.
3. **Task 9 — MI-GAN UI (sonnet), Task 10 — 벤치 (sonnet).** Task 9 착수 시
   `ComponentSourceField`/`emptySource`를 `Flux2KleinOptions.tsx`에서
   `PreferenceFields.tsx`로 옮길 것 — 세 번째 소비자가 생기는 시점이다.

**사용자 확인이 필요한 것 (에이전트가 못 하는 일):**

- **실제 앱에서의 엔드투엔드 검증.** `bun run dev`로 LaMa를 TorchScript로 바꿔 한 페이지를
  인페인팅. 197MB 다운로드와 데스크톱 세션이 필요해 이번 세션에서 **돌리지 못했다.**
  타입·검증 로직 정합은 확인됐지만 네트워크 실패나 다이제스트 불일치 시의 UX는 미검증이다.
- **번역 원어민 확인:** tr-TR "Biçim", ru-RU "Дайджест"는 기계 번역이다.
- **별건:** `sourceKind.builtin`/`localFile`이 ko-KR을 제외한 전 로케일에서 영어로 남아
  있다. 이 브랜치와 무관한 기존 상태라 손대지 않았다.

---

## 6. 환경 함정 두 가지 — 모르면 시간을 크게 버린다

### 6.1 libtorch DLL이 PATH에 없으면 코드 결함처럼 보인다

```
failed to load any dynamic library from [koharu-torch]: koharu-torch.dll: LoadLibraryExW failed
```

**환경 문제다.** 로더는 실행 파일 옆을 먼저 보고 없으면 OS 검색 경로로 넘어가는데,
테스트 바이너리는 `target/debug/deps/`에 있고 거기엔 DLL이 없다. 두 디렉터리가 필요하다.

```bash
LA=$(cygpath -u "$LOCALAPPDATA")
export PATH="$LA/koharu:$LA/koharu/packages/torch/2.12.1/cpu/libtorch/lib:$PATH"
```

`cygpath -u`가 핵심이다. `$LOCALAPPDATA`는 백슬래시를 포함하고, 콜론으로 구분되는
PATH에 `C:\...`를 그대로 넣으면 드라이브 문자에서 잘려 조용히 깨진다. 코디네이터가
이걸로 한 번 오진했다.

### 6.2 `koharu-ml` 테스트는 반드시 단일 스레드로

`cargo test -p koharu-ml`은 **결과 줄을 출력하기 전에 죽는다.**

```
running 80 tests
Key already registered with the same priority: C10
(exit 1)
```

libtorch 정적 레지스트리의 이중 등록 크래시다. 위험한 점은 "test result" 줄이 아예
안 나온다는 것 — **실제로 이 계획의 한 태스크에서 "테스트 통과" 오보가 나왔다.**

```bash
cargo test -p koharu-ml --lib -- --test-threads=1
```

**baseline (2026-08-22 실측):**

| 크레이트 | 결과 |
|---|---|
| `koharu-ml --lib` (단일 스레드) | 65 passed, 0 failed, 18 ignored (Task 3 이후) |
| `koharu-runtime` | 21 passed, 0 failed |
| `koharu-pipeline` | 67 passed, 0 failed |

**정정 (Task 3 시점):** Task 0~2 동안 계속 실패하던 두 테스트

- `baberu_ocr::processor::tests::bicubic_resize_stays_aligned_with_pillow`
- `comic_onomatopoeia::recognizer::processor::tests::bicubic_resize_stays_aligned_with_pillow`

는 **코드 문제가 아니라 환경(Python 경로) 문제였고 지금은 통과한다.** 이 브랜치와 무관하다는
판단 자체는 맞았으나 "영구적 기존 실패"로 본 것은 틀렸다. **현재 기준선은 실패 0건이다 —
어떤 실패든 나오면 회귀로 취급할 것.**

---

## 7. 계획서를 믿지 말 것 — 이미 4건이 틀렸다

`plans/2026-08-22-inpainting-models.md`의 코드 블록은 **한 번도 컴파일된 적이 없다.**
지금까지 드러난 결함:

1. `blake3::Hasher::update_mmap_rayon` — 워크스페이스가 `mmap`/`rayon` 피처를 켜지 않아
   컴파일 불가. `update_reader`로 대체됨.
2. 다이제스트 대소문자 정책이 산문과 코드에서 불일치. **양쪽 다 받되 캐시 경로는 소문자로
   정규화**로 확정(그러지 않으면 같은 파일이 200MB짜리 캐시 엔트리를 둘 차지한다).
3. Task 1 Step 1이 아직 존재하지 않는 `pub mod source;`를 등록하라고 지시 — 그대로 하면
   빌드가 깨진다.
4. Task 2의 테스트 단언 문자열이 실제 에러 메시지와 불일치.

**Task 3의 `Box<dyn Backend>`와 Task 7의 텐서 연산은 아직 검증되지 않은 추론이다.**
구현자에게 "스니펫은 제안이지 성경이 아니다, 고치고 신고하라"고 계속 지시할 것.

계획서 자체 리뷰 절에 남아 있는 미해결 위험 하나: Task 7의 `pad_to_square` 가정
(`resize_dimensions`가 긴 변을 512로 맞추고 modulo 512 패딩이 짧은 변을 512로 올려 결국
512×512가 된다는 추론). Task 7 실행 확인에서 출력 크기로 검증해야 한다.

---

## 8. 진행 방식

사용자 지시로 **subagent-driven development** + 모델 혼합이다.

- 태스크마다 새 구현 서브에이전트 → 스펙 리뷰 → 품질 리뷰 → 다음 태스크
- 간단한 기계적 작업은 sonnet, 설계 판단·다중 파일 파급·수치 정확성은 opus
- 리뷰어도 같은 등급으로 맞춘다

**프로세스 사고 하나가 있었다.** 코디네이터가 구현자의 `--amend` 도중 같은 저장소에
커밋해 코드 수정이 docs 커밋에 섞여 들어갔다. reflog로 분리 복구했다. 이후 규칙:
**구현자는 `--amend` 금지(일반 커밋만), 코디네이터는 서브에이전트 활성 중 커밋 금지.**
