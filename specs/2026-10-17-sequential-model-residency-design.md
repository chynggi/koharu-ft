# 순차 모델 레지던시 설계

작성일: 2026-10-17

## 배경

`specs/2026-10-17-flux3-klein-inpainting-design.md`의 D8은 공용 VRAM↔RAM 레지던시
매니저를 후속 조사로 미뤘다. 그 조사의 계기가 된 실측이 나왔으므로 이 문서에서 닫는다.

지금 구조의 문제는 하나다. stage가 끝나도 그 모델은 `ModelCell`에 남는다
(`crates/koharu-pipeline/src/model_cell.rs`). 선제적으로 모델을 내리는 경로는
`AcceleratorGate::recover()` 하나뿐이고, 그건 OOM 재시도에서만 불린다. 그래서 다음
stage가 자기 모델을 올릴 때 앞 stage의 모델이 아직 가속기 위에 있다.

### 실측 (RTX 3060 12GB, 12288 MiB)

`g4-full2`(1페이지), `g4-two`(2페이지, 한 프로세스) 실행의 nvidia-smi 500ms 샘플.

| 구간 | 1페이지 | 2페이지 |
|---|---|---|
| 인페인팅 phase 1(캡션, PUP 상주) | 9,425–9,461 MiB | 11,938–11,968 MiB |
| 인페인팅 phase 2(타일) | 6,534–6,999 MiB | 11,649–12,069 MiB |
| 번역(gemma-4-12B 상주) | 11,944–11,953 MiB | 11,709–12,027 MiB |
| 프로세스 전체 피크 | **11,953 MiB** | **12,069 MiB** |

2페이지는 t=340s부터 t=1700s까지 내내 11.6–12.0 GB를 유지한다. 12,288 MiB 중
여유가 300 MiB 미만이다. OOM은 나지 않았지만 마진이 없다.

구성 요소별 VRAM(CUDA0 model buffer, 실측 로그):

| 모델 | CUDA0 | 페이지 사이 유지 |
|---|---|---|
| PaddleOCR-VL-1.6 | 688.14 MiB | 예 |
| gemma-4-12B-it (번역 LLM) | 6,390.19 MiB | **예** — 2페이지 전체에서 유지 |
| FLUX 3 PUP | 4,812.25 MiB | 아니오 — `caption()`이 페이지마다 로드/해제 |
| FLUX 3 Image Klein | ~0 | `params_backend: "*=cpu"`로 RAM 상주 |

2페이지가 위험한 이유는 하나다: `6,390 + 4,812 + 688 ≈ 11,890 MiB`. 번역 LLM과
PUP의 동시 상주다. FLUX 3 자신은 VRAM을 쓰지 않으므로(phase 2가 6.5–7.0 GB로
떨어지는 것이 그 증거) 범인이 아니다.

원인이 page 단위임을 확인했다. 2페이지 실행에서 PUP 로드는 2회, gemma 로드는 1회다.
PUP는 매 페이지 다시 올라오고, gemma는 페이지 사이에 내려가지 않는다.

### 왜 동시에 실행되지는 않는데 겹치는가

`AcceleratorGate`의 permit은 1개다(`accelerator.rs`). 그래서 두 stage의 *실행*이
겹치지는 않는다. 문제는 permit 범위다. `stage_runner.rs`가 로드와 처리 전체를
한 permit으로 감싸므로, 앞 stage가 permit을 놓아도 그 모델은 셀에 남는다. 다음
stage가 permit을 잡고 자기 모델을 올리면서 겹친다.

## 결정 사항

| # | 결정 | 근거 |
|---|---|---|
| D9 | **가속기 작업 하나가 시작될 때, 다른 모든 stage의 퇴거를 요청한다.** 장치와 무관하게 적용한다 | 2페이지 실측이 지목한 원인이 번역 LLM과 PUP의 동시 상주다. 하나씩만 올리면 12GB에서 여유 4.5GB 이상이 확보된다. CPU도 RAM을 공유하므로 함께 적용한다 |
| D10 | 퇴거 시점은 **`AcceleratorGate::acquire`가 소유**한다. permit을 잡은 **뒤** 다른 stage를 내린다 | 이 타입이 이미 "가속기 위에서 무엇이 언제 도는가"를 소유한다. permit 뒤에 두면 진행 중인 커널을 뽑아내지 않는다 |
| D11 | 요청한 stage 자신은 **내리지 않는다** | 같은 stage의 다음 페이지가 재로드 없이 셀을 재사용한다. FLUX 3는 페이지 사이에 그대로 상주한다 |
| D12 | **예산 기반 판단을 하지 않는다.** Tier M/A/E, 모델 크기 API, scope별 안전 여유는 도입하지 않는다 | 결정된 정책은 "항상 단일 상주"다. 실제로 예산 비교가 필요해지기 전에는 추측성 추상화다. `fork/VRAM-ACCOUNTING-DESIGN.md` §8의 나머지 미결 항목은 열린 채로 둔다 |
| D13 | **D4를 개정한다**: "다른 stage와 `accelerator.rs`는 바꾸지 않는다" → "stage 소유권은 그대로 두고, *언제 내리는가*만 `AcceleratorGate`가 정한다" | D4는 stage 내부 구조를 인페인팅 안에 가두는 결정이었다. 퇴거 시점은 stage 내부가 아니라 가속기 조율의 문제다 |

## 변경

`crates/koharu-pipeline/src/accelerator.rs` 하나만 바뀐다.

```rust
pub(crate) async fn acquire(&self, stage: Stage, stages: &Stages) -> AcceleratorPermit {
    let permit = self.lane_acquire().await;
    // 다른 stage의 모델을 내려서 가속기를 이 stage에 넘긴다. permit 뒤에 두는
    // 이유는 진행 중인 커널을 뽑아내지 않기 위해서다. lane이 있으면 다른
    // stage가 실행 중일 수 없으므로 퇴거가 확실하고, 없으면 best-effort다.
    unload_other_models(stage, stages);
    permit
}

pub(crate) async fn recover(&self, stage: Stage, stages: &Stages) -> AcceleratorPermit {
    let permit = self.acquire(stage, stages).await;
    if self.lane.is_some() {
        let mut changed = self.resources.subscribe();
        let _ = tokio::time::timeout(Duration::from_millis(600), changed.changed()).await;
    }
    permit
}
```

- `lane_acquire`는 기존 `acquire` 본문에서 permit 획득만 남긴 private 함수다.
- `unload_other_models`는 시그니처가 그대로다. `Stage::ALL`을 순회하며 요청 stage를
  제외한 나머지를 내리고, 하나라도 내렸으면 `model_unload` 메트릭을 낸다.
- 호출부는 `stage_runner.rs` 한 곳이다. `acquire()`에 `job.stage`와 `&self.stages`를
  넘긴다. `recover()` 호출부는 시그니처가 그대로라 바뀌지 않는다.

### 불변식의 정확한 서술

`unload()`는 `try_lock` 기반 best-effort다(`model_cell.rs`). 가속기 lane이 있으면
permit이 하나뿐이라 상대 stage가 실행 중일 수 없으므로 퇴거가 확실하다. CPU는
lane이 없어 두 stage가 실제로 동시 실행될 수 있고, 그때는 실행 중인 stage의 셀이
잠겨 있어 퇴거가 실패한다.

그래서 D9의 정확한 문장은 이렇다: **로드 전에 다른 모든 stage의 퇴거를 요청한다.
실행 중인 stage는 자기 모델을 유지한다.** CPU에서 "상주 최대 1개"는 보장이 아니라
통상 결과다.

## 관측

기존 게이트가 깨진다. `specs/2026-10-17-flux3-klein-inpainting-design.md`의 Task 9
Step 8 게이트는 `model_unload` 이벤트가 **0건**임을 "OOM 복구가 없었다"의 증거로
썼다. 이제 정상 경로가 stage 전환마다 그 이벤트를 낸다.

OOM 복구는 이미 별도 신호가 있다: `stage_retry` 메트릭과
`"retrying stage after memory pressure"` WARN이다. 그래서 `model_unload`는
"가속기를 비우려고 모델을 내렸다"는 뜻으로 넓어졌다고 기록하고, 게이트 판별
기준을 `stage_retry`로 바꾼다.

`model_unload`에 이유 필드를 추가하지 않는다. 정상 퇴거와 OOM 퇴거를 가르는 정보는
`stage_retry`가 이미 나르고, 값이 하나뿐인 필드는 추측성이다.

## 오류 처리

| 상황 | 처리 |
|---|---|
| 퇴거 실패(셀이 잠김) | 무시한다. best-effort다. OOM이 나면 `recover()` 재시도 경로가 받는다 |
| `recover()` 재시도 후에도 OOM | 이제 재시도할 카드가 없다. `acquire()`가 이미 다른 모델을 내렸기 때문이다. 실패는 그대로 올라간다 |
| `stage_error()`의 `stages.unload(stage)` | 그대로 둔다. 실패한 stage의 모델을 버려 재시도를 가능하게 하는 별개 책임이다 |

`recover()`의 600ms 샘플 대기는 남긴다. 실제 회수와 다음
`device_for_load()`의 headroom 갱신이 그 시간에 일어난다.

## 비용

재로드 비용(실측):

| 모델 | 로드 시간 |
|---|---|
| PaddleOCR-VL-1.6 | 0.2초 |
| gemma-4-12B-it | 2.3초 |
| FLUX 3 PUP | 3.0초 |
| FLUX 3 Image Klein (sd.cpp 텐서 로드) | 0.20초 |

1페이지 처리 270초 기준으로 페이지당 1–2%다. RAM 쪽도 FLUX 3의 CPU 상주 복사본이
페이지 사이에 해제된다(2페이지 실행의 working set 24.6 GiB가 줄어든다).

## 검증

| 계층 | 내용 |
|---|---|
| 단위 | 퇴거 대상이 `Stage::ALL`에서 요청 stage를 뺀 나머지이고 순서가 `Stage::ALL`과 같은지. 자기 stage를 내리면 같은 stage의 다음 페이지가 매번 재로드되므로 회귀 보호 가치가 있다 |
| 배선 | CPU `Stages`로 `acquire(stage, &stages)`가 permit을 돌려주고 교착이 없는지 |
| 수동 게이트 | 가중치가 필요하므로 단위 테스트로는 퇴거 효과를 관측할 수 없다. `g4_two.sh`를 다시 돌려 2페이지 피크를 확인한다 |

수동 게이트 예측값:

| 구간 | 변경 전 | 예측 |
|---|---|---|
| 인페인팅 phase 1 | 11,968 MiB | 약 5.5–6.0 GB (PUP 4,812 + 오버헤드) |
| 인페인팅 phase 2 | 12,069 MiB | 그 이하 |
| 번역 | 11,944–12,027 MiB | 약 7.1–7.5 GB (gemma 6,390 + 오버헤드) |

12GB 대비 여유가 300 MiB에서 4.5 GB 이상으로 벌어져야 한다. 게이트는 2페이지여야
한다 — 1페이지에는 번역-인페인팅 겹침이 드러나지 않는다.

## 테스트

debug 프로필로 관련 테스트만 한 번 실행한다.

## 범위 밖

| 항목 | 사유 |
|---|---|
| 예산 기반 퇴거와 모델 크기 API | D12 |
| `ModelCell` API 변경 | 퇴거 시점만 바뀐다. 셀 자체는 그대로다 |
| `koharu-diffusion` / `koharu-llama` 변경 | 퇴거는 로더 위 계층의 문제다 |
| FLUX 3 페이지 내 2단계 스왑(D3) | 그대로 유지한다. PUP는 여전히 `caption()` 스코프에서 해제된다 |
| UI·i18n·문서(Phase E) | 별도 작업이다 |

## 기존 문서에 미치는 영향

- `specs/2026-10-17-flux3-klein-inpainting-design.md`
  - D4를 D13에 맞춰 개정한다.
  - D8을 "이 문서에서 닫음"으로 바꾸고 이 문서를 가리킨다.
  - G4 행의 낡은 수치(`7,434MiB`, `5.2–5.5GB`, `RAM 감소 약 6.8GB`)를 위 실측치로
    교체한다. 그 수치는 stage 하나만 돌린 측정이었다.
  - Task 9 Step 8 게이트의 판별 기준을 `stage_retry`로 바꾼다.
- `fork/VRAM-ACCOUNTING-DESIGN.md` §8의 "퇴거 순서" 미결 항목은 이 결정으로 닫혔다.
  나머지(scope별 안전 여유, `params_backend` 20GiB 임계값의 예산 상대화,
  `koharu-diffusion` 크기 API)는 열린 채로 둔다.

## 참고

- `specs/2026-10-17-flux3-klein-inpainting-design.md`
- `fork/VRAM-ACCOUNTING-DESIGN.md`
- 실측: `scratchpad/gate/out/g4-full2`(1페이지), `scratchpad/gate/out/g4-two`(2페이지)
