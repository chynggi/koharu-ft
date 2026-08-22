# HANDOFF — 인페인팅 모델 확장 (`feat/inpainting-torchscript-migan`)

최종 갱신: 2026-08-22 (Task 0~2 구현·리뷰 완료. 다음은 Task 3)

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
| 2 | `remote_repository!` 매크로 | 완료(미사용, Task 3에서 첫 사용) | `koharu-ml/src/lib.rs` |

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

### 남은 Task (계획서 참조)

| Task | 내용 | 모델 배정(제안) |
|---|---|---|
| 3 | LaMa `Backend` 트레이트 + safetensors/TorchScript 두 구현 | opus |
| 4 | `InpaintingModel::LaMa(LaMaConfig)` 파이프라인 배선 | sonnet |
| 5 | LaMa 소스·형식 UI + 9개 로케일 | sonnet |
| 6 | 전처리 헬퍼를 `inpaint_ops.rs`로 추출 | sonnet |
| 7 | MI-GAN 모듈 (텐서 연산) | opus |
| 8 | MI-GAN 파이프라인 배선 | sonnet |
| 9 | MI-GAN UI | sonnet |
| 10 | 벤치 — LaMa 대비 지연 시간·VRAM | sonnet |

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
| `koharu-ml --lib` (단일 스레드) | 60 passed, **2 failed**, 17 ignored |
| `koharu-runtime` | 21 passed, 0 failed |
| `koharu-pipeline` | 67 passed, 0 failed |

`2 failed`는 **이 브랜치와 무관한 기존 실패**다:

- `baberu_ocr::processor::tests::bicubic_resize_stays_aligned_with_pillow`
- `comic_onomatopoeia::recognizer::processor::tests::bicubic_resize_stays_aligned_with_pillow`

`git diff --name-only main...HEAD`로 해당 모듈 미변경을 확인했고, 사전 커밋 `f7b6ed6e`를
별도 worktree에 체크아웃해 동일 크래시를 재현했다. **이 둘 외에 실패가 늘면 그건 회귀다.**

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
