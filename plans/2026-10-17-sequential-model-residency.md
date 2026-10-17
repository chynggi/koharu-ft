# 순차 모델 레지던시 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 가속기 작업 하나가 시작될 때 다른 모든 stage의 모델을 내려서, 2페이지 처리에서 12GB 카드의 여유를 300MiB에서 4.5GB 이상으로 벌린다.

**Architecture:** `AcceleratorGate::acquire`가 permit을 잡은 직후 `unload_other_models`를 호출한다. 요청한 stage 자신은 내리지 않으므로 같은 stage의 다음 페이지는 재로드 없이 `ModelCell`을 재사용한다. `recover()`는 `acquire()`에 위임하고 샘플 대기만 남긴다. 변경 파일은 `accelerator.rs` 하나이고 호출부는 `stage_runner.rs` 한 줄이다.

**Tech Stack:** Rust, tokio (Semaphore permit, Mutex), tracing 메트릭.

**설계 문서:** `specs/2026-10-17-sequential-model-residency-design.md`

---

## 파일 구조

| 파일 | 책임 | 변경 |
|---|---|---|
| `crates/koharu-pipeline/src/accelerator.rs` | 가속기 접근 직렬화와 퇴거 시점 소유. 퇴거 대상 계산(순수) + permit 획득 + 복구 | 수정 |
| `crates/koharu-pipeline/src/stage_runner.rs` | stage 작업 실행. `acquire`에 stage와 Stages를 넘김 | 1줄 수정 |
| `crates/koharu-pipeline/src/bin/run.rs` | 게이트용 실행 하네스(tracing 및 다중 입력) | 커밋만 (내용 변경 없음) |
| `crates/koharu-pipeline/Cargo.toml` | `tracing-subscriber` 의존성 | 커밋만 |
| `crates/koharu-pipeline/src/stages/inpainting.rs` | FLUX 3 타일 수 로그 | 커밋만 |
| `AGENTS.md` | 지속 제약 기록 | 수정 |
| `plans/2026-10-17-flux3-klein-inpainting.md` | Task 9 Step 8의 판별 기준 | 수정 |
| `specs/2026-10-17-flux3-klein-inpainting-design.md` | G4 행 수치와 D4·D8 문구 | 수정 |

---

## Task 1: Step 8 계측 변경을 커밋한다

FLUX 3 Task 9 Step 8에서 측정을 위해 넣은 변경이 아직 커밋되지 않았다. 뒤의 수동 게이트가 이 하네스를 쓰므로 먼저 커밋한다. 새 코드가 아니므로 테스트 단계는 없다.

**Files:**
- Modify(커밋만): `crates/koharu-pipeline/src/bin/run.rs`, `crates/koharu-pipeline/Cargo.toml`, `crates/koharu-pipeline/src/stages/inpainting.rs`, `Cargo.lock`

- [ ] **Step 1: 커밋한다**

```bash
git add Cargo.lock crates/koharu-pipeline/Cargo.toml crates/koharu-pipeline/src/bin/run.rs crates/koharu-pipeline/src/stages/inpainting.rs
git commit -m "feat(pipeline): instrument the run harness for gate measurement

The gate reads koharu_metrics events out of the binary's stderr, so the
binary needs the same tracing subscriber shape the desktop app installs;
without one it emitted nothing. Repeated --input renders one PNG per page
into an output directory, which is what the two-page residency gate
needs. Log the FLUX 3 tile count so the gate can record it.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

- [ ] **Step 2: 커밋되지 않은 것이 없는지 확인한다**

Run: `git status --short`
Expected: `specs/`와 `packages/` 외에 `crates/koharu-pipeline` 항목이 없다. `packages/koharu/components/editor/usePageImportDrop.ts`는 이 계획과 무관한 미추적 파일이므로 건드리지 않는다.

---

## Task 2: 퇴거 대상을 계산하는 순수 함수를 만든다

퇴거 규칙을 셀 조작에서 분리해 테스트 가능하게 만든다. `Stage::ALL` 순서를 유지해야 퇴거 순서가 예측 가능하다.

**Files:**
- Modify: `crates/koharu-pipeline/src/accelerator.rs`
- Test: `crates/koharu-pipeline/src/accelerator.rs` (파일 끝 `mod tests`)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`crates/koharu-pipeline/src/accelerator.rs` 파일 끝에 추가한다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eviction_targets_exclude_only_the_requested_stage() {
        assert_eq!(
            eviction_targets(Stage::Inpainting).collect::<Vec<_>>(),
            [Stage::Detection, Stage::Ocr, Stage::Translation]
        );
        assert_eq!(
            eviction_targets(Stage::Detection).collect::<Vec<_>>(),
            [Stage::Ocr, Stage::Translation, Stage::Inpainting]
        );
    }

    #[test]
    fn every_stage_evicts_every_other_stage() {
        for requested in Stage::ALL {
            let targets = eviction_targets(requested).collect::<Vec<_>>();
            assert_eq!(targets.len(), Stage::ALL.len() - 1);
            assert!(!targets.contains(&requested));
        }
    }
}
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `cargo test -p koharu-pipeline --lib accelerator::tests`
Expected: 컴파일 실패 — "cannot find function `eviction_targets` in this scope".

- [ ] **Step 3: 최소 구현을 쓴다**

`crates/koharu-pipeline/src/accelerator.rs`에서 기존 `unload_other_models` 함수 **위에** 추가한다.

```rust
/// Stages whose models must be dropped so that only `requested` may hold one.
/// The order follows `Stage::ALL`, which keeps eviction order predictable in
/// logs. The requesting stage is never a target: keeping it resident lets the
/// next page of the same stage reuse its cell without reloading.
fn eviction_targets(requested: Stage) -> impl Iterator<Item = Stage> {
    Stage::ALL.into_iter().filter(move |stage| *stage != requested)
}
```

그리고 기존 `unload_other_models`의 루프를 이 함수를 쓰도록 바꾼다.

변경 전:

```rust
fn unload_other_models(requested: Stage, stages: &Stages) -> bool {
    let mut unloaded = false;
    for stage in Stage::ALL {
        if stage != requested && stages.unload(stage) {
            unloaded = true;
            tracing::info!(target: "koharu_metrics", metric = "model_unload", stage = %stage);
            tracing::debug!(%stage, "unloaded model while recovering from memory pressure");
        }
    }
    unloaded
}
```

변경 후:

```rust
fn unload_other_models(requested: Stage, stages: &Stages) -> bool {
    let mut unloaded = false;
    for stage in eviction_targets(requested) {
        if stages.unload(stage) {
            unloaded = true;
            tracing::info!(target: "koharu_metrics", metric = "model_unload", stage = %stage);
            tracing::debug!(%stage, "unloaded a model to free the accelerator");
        }
    }
    unloaded
}
```

주석 문구가 "while recovering from memory pressure"에서 바뀌는 이유: 이 함수가 이제 정상 경로에서도 불린다. `debug!` 메시지를 실제 호출 맥락에 맞춘다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -p koharu-pipeline --lib accelerator::tests`
Expected: PASS (2 tests).

- [ ] **Step 5: 커밋한다**

```bash
git add crates/koharu-pipeline/src/accelerator.rs
git commit -m "refactor(pipeline): name the stages an accelerator handover evicts

Pull the eviction set out of the unload loop so the rule is testable, and
drop the recovery-specific wording: this runs on the normal path next.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 3: `acquire`가 permit 뒤에 다른 stage를 내리게 한다

**Files:**
- Modify: `crates/koharu-pipeline/src/accelerator.rs`

- [ ] **Step 1: `acquire`와 `recover`를 바꾼다**

`crates/koharu-pipeline/src/accelerator.rs`에서 `impl AcceleratorGate` 안의 `acquire`와 `recover`를 아래로 교체한다.

변경 전:

```rust
    pub(crate) async fn acquire(&self) -> AcceleratorPermit {
        let Some(lane) = self.lane.as_ref() else {
            return AcceleratorPermit::cpu();
        };
        AcceleratorPermit::accelerator(
            lane.clone()
                .acquire_owned()
                .await
                .expect("accelerator lane is never closed"),
        )
    }

    pub(crate) async fn recover(&self, stage: Stage, stages: &Stages) -> AcceleratorPermit {
        let permit = self.acquire().await;
        if self.lane.is_none() {
            return permit;
        }
        if unload_other_models(stage, stages) {
            let mut changed = self.resources.subscribe();
            let _ = tokio::time::timeout(Duration::from_millis(600), changed.changed()).await;
        }
        permit
    }
```

변경 후:

```rust
    /// Takes the accelerator for `stage`, dropping every other stage's model
    /// before the caller loads its own.
    ///
    /// The eviction happens *after* the permit is held. With the single lane
    /// that means no other stage can be mid-run, so nothing that is running
    /// gets its memory pulled out from under it. On CPU there is no lane and
    /// two stages may genuinely overlap; there `ModelCell::unload` is
    /// best-effort and a busy stage keeps its model.
    pub(crate) async fn acquire(&self, stage: Stage, stages: &Stages) -> AcceleratorPermit {
        let permit = self.lane_acquire().await;
        unload_other_models(stage, stages);
        permit
    }

    /// Retries after memory pressure. The eviction already happened in
    /// [`Self::acquire`], so all that is left is to let the driver actually
    /// return the memory and let the monitor publish a fresh reading, which
    /// the next load sees as its headroom.
    pub(crate) async fn recover(&self, stage: Stage, stages: &Stages) -> AcceleratorPermit {
        let permit = self.acquire(stage, stages).await;
        if self.lane.is_some() {
            let mut changed = self.resources.subscribe();
            let _ = tokio::time::timeout(Duration::from_millis(600), changed.changed()).await;
        }
        permit
    }

    async fn lane_acquire(&self) -> AcceleratorPermit {
        let Some(lane) = self.lane.as_ref() else {
            return AcceleratorPermit::cpu();
        };
        AcceleratorPermit::accelerator(
            lane.clone()
                .acquire_owned()
                .await
                .expect("accelerator lane is never closed"),
        )
    }
```

`recover`의 600ms 대기는 lane이 있을 때 항상 기다린다. OOM 재시도에서만 불리는 드문 경로이고, 대기 자체가 다음 로드의 headroom을 갱신하므로 조건을 다시 계산하지 않는다.

- [ ] **Step 2: `Stages::unload`가 `Stage`를 받는지 확인한다**

Run: `grep -n "pub(crate) fn unload" crates/koharu-pipeline/src/stages/mod.rs`
Expected: `pub(crate) fn unload(&self, stage: Stage) -> bool` — 시그니처가 이미 맞으므로 `Stages`를 그대로 넘기면 된다.

- [ ] **Step 3: 커밋하지 않고 컴파일만 확인한다**

Run: `cargo check -p koharu-pipeline --lib`
Expected: 실패 — `stage_runner.rs`의 `acquire` 호출이 인자를 안 넘겨서 E0061. Task 4에서 고친다.

---

## Task 4: 호출부를 고친다

**Files:**
- Modify: `crates/koharu-pipeline/src/stage_runner.rs:76`

- [ ] **Step 1: `acquire` 호출을 바꾼다**

`crates/koharu-pipeline/src/stage_runner.rs`에서 `run_with_recovery` 안의 다음 줄을 찾는다.

변경 전:

```rust
        let permit = self.accelerator.acquire().await;
```

변경 후:

```rust
        let permit = self.accelerator.acquire(job.stage, &self.stages).await;
```

`recover` 호출(`let _permit = self.accelerator.recover(job.stage, &self.stages).await;`)은 시그니처가 그대로라 바꾸지 않는다.

- [ ] **Step 2: 컴파일과 관련 테스트를 확인한다**

Run:
```bash
cargo test -p koharu-pipeline --lib accelerator::tests
cargo test -p koharu-pipeline --lib stages::tests::registry_contains_every_stage
cargo check -p koharu-pipeline --lib
```
Expected: 모두 PASS·성공.

- [ ] **Step 3: 배선 테스트를 추가한다**

`crates/koharu-pipeline/src/accelerator.rs`의 `mod tests`에 추가한다. 모델이 하나도 로드되지 않은 상태이므로 퇴거 효과는 관측되지 않는다. 확인하는 것은 permit이 돌아오고 교착이 없다는 것뿐이다.

```rust
    #[tokio::test]
    async fn acquire_returns_a_permit_with_no_models_loaded() {
        let device = koharu_ml::Device::cpu();
        let translator = koharu_translator::Translator::from_config(
            device.clone(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let resources = crate::resources::ResourceMonitor::new(&device);
        let stages = Stages::new(
            &crate::PipelineConfig::default(),
            translator,
            &device,
            resources.clone(),
        )
        .unwrap();
        let gate = AcceleratorGate::new(&device, resources);

        let _permit = gate.acquire(Stage::Inpainting, &stages).await;
        let _permit = gate.recover(Stage::Inpainting, &stages).await;
    }
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -p koharu-pipeline --lib accelerator::tests`
Expected: PASS (3 tests).

- [ ] **Step 5: 커밋한다**

```bash
git add crates/koharu-pipeline/src/accelerator.rs crates/koharu-pipeline/src/stage_runner.rs
git commit -m "feat(pipeline): evict other stages when a stage takes the accelerator

A stage's model survived in its ModelCell after it released the permit, so
the next stage loaded on top of it. Two-page runs held gemma-4-12B
(6390 MiB) resident while the FLUX 3 upsampler (4812 MiB) captured the
next page's captions, pinning the device at 11.6-12.0 GB of 12288 MiB for
sixteen minutes.

Evict every other stage right after the permit is acquired. The requesting
stage stays resident, so repeat pages of the same stage reload nothing.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 5: 지속 제약을 기록한다

`AGENTS.md`는 리팩터 중에 바뀌는 구현 세부가 아니라 오래 남는 결정 규칙을 담는다. "정확히 한 stage의 모델만 가속기에 둔다"는 이 저장소가 계속 지켜야 할 규칙이므로 기록한다.

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: `## ML Architecture` 절 끝에 추가한다**

`AGENTS.md`의 `## ML Architecture` 목록에서 `- Disable gradient tracking during inference.` **아래에** 추가한다.

```markdown
- Let one stage hold the accelerator at a time. A stage that takes it drops every other stage's model before loading its own, and keeps its own resident so repeat pages reload nothing.
```

- [ ] **Step 2: 확인한다**

Run: `git diff AGENTS.md`
Expected: 위 한 줄만 추가된다. 목록의 기존 항목은 바뀌지 않는다.

- [ ] **Step 3: 커밋한다**

```bash
git add AGENTS.md
git commit -m "docs: record the single-resident-model rule for accelerators

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 6: FLUX 3 계획의 게이트 판별 기준을 고친다

Task 9 Step 8은 `model_unload` 이벤트가 0건임을 "OOM 복구가 없었다"의 증거로 썼다. 이제 정상 경로가 stage 전환마다 그 이벤트를 내므로 기준이 성립하지 않는다. OOM 복구는 `stage_retry` 메트릭과 해당 WARN이 이미 나른다.

**Files:**
- Modify: `plans/2026-10-17-flux3-klein-inpainting.md` (Task 9 Step 8)

- [ ] **Step 1: Step 8의 판별 기준을 바꾼다**

변경 전:

```markdown
G4 머신에서 `koharu-pipeline`의 `run` 바이너리를 `--help`로 확인하고, 인페인팅 선택을 `flux3-klein`으로 해서 대표 페이지 1장을 처리한다. 확인할 것은 세 가지다.
- 로그에 `model_unload` 없이 페이지가 끝나는지
- 결과 PNG가 만들어지는지
- 처리 시간
```

변경 후:

```markdown
G4 머신에서 `koharu-pipeline`의 `run` 바이너리를 `--help`로 확인하고, 인페인팅 선택을 `flux3-klein`으로 해서 대표 페이지 1장을 처리한다. 확인할 것은 네 가지다.
- 로그에 `stage_retry`(`"retrying stage after memory pressure"` WARN)가 없는지. `model_unload`는 정상 경로의 stage 전환에서도 나므로 OOM 복구의 증거가 아니다(`specs/2026-10-17-sequential-model-residency-design.md`).
- `tiles=` 로그의 타일 개수
- 결과 PNG가 만들어지는지
- 처리 시간
```

- [ ] **Step 2: 확인한다**

Run: `git diff plans/2026-10-17-flux3-klein-inpainting.md`
Expected: 위 블록만 바뀐다.

- [ ] **Step 3: 커밋한다**

```bash
git add plans/2026-10-17-flux3-klein-inpainting.md
git commit -m "docs(plans): judge the FLUX 3 gate by stage retries, not model unloads

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 7: 2페이지 게이트를 다시 돌린다 (수동)

실제 퇴거 효과는 가중치가 필요하므로 단위 테스트로 관측할 수 없다. 1페이지에는 번역-인페인팅 겹침이 나타나지 않으므로 반드시 2페이지로 돌린다. 예상 소요는 약 30분이다.

**Files:**
- 없음 (측정)

- [x] **Step 1: 디버그 바이너리를 빌드한다**

Run: `cargo build -p koharu-pipeline --bin run`
Expected: 성공. 게이트 스크립트가 `target/debug/run`을 실행한다.
결과: 성공. `--bin run`만 쓰면 `no bin target named 'run' in default-run packages`로
실패하므로 `-p koharu-pipeline`을 붙여야 한다. 빌드 후 바이너리 mtime이 소스보다
최신인지 확인했다.

- [x] **Step 2: 게이트 스크립트를 복사해 출력 경로를 바꾼다**

기존 `g4-two` 결과를 보존하기 위해 새 디렉터리에 쓴다. 스크립트는 스크래치패드에 있다.

Run:
```bash
G="C:/Users/chyng/AppData/Local/Temp/claude/M--koharu-ft/1f1a87d2-bfa9-4469-8e35-d9020b125795/scratchpad/gate"
sed 's#out/g4-two#out/g4-two-after#' "$G/../gate_tools/g4_two.sh" > "$G/../gate_tools/g4_two_after.sh"
grep -n "OUT=" "$G/../gate_tools/g4_two.sh"
```
Expected: 복사본의 `OUT`이 `out/g4-two-after`를 가리키고, 다른 설정(입력 2장, `--inpainting flux3-klein`, `--llm gemma4-12b-it`)은 `g4-two`와 같다.

- [x] **Step 3: 게이트를 실행한다**

Run: `bash "C:/Users/chyng/AppData/Local/Temp/claude/M--koharu-ft/1f1a87d2-bfa9-4469-8e35-d9020b125795/scratchpad/gate_tools/g4_two_after.sh"`
Expected: `exit=0`, `wall=` 값이 출력된다. 약 30분 걸린다.

결과: `exit=0`, `wall=832s`. 기준선 `g4-two`는 1751s였다.

- [x] **Step 4: 피크를 뽑는다**

Expected: `peak`가 **7500 MiB 이하**다.

결과: **`peak = 10,776 MiB`** (기준선 12,069 MiB). 예측이 틀렸다. 원인은 아래
Step 5에 적었다.

단계별 최대(이벤트 타임스탬프로 구간을 잘라 계산):

| 구간 | 피크 | 평균 |
|---|---|---|
| 1페이지 detection | 1,551 MiB | 1,091 MiB |
| 1페이지 ocr | 3,398 MiB | 2,698 MiB |
| 1페이지 인페인팅 phase 1 (PUP) | 8,010 MiB | 5,541 MiB |
| 1페이지 인페인팅 phase 2 (타일 11개) | 1,784 MiB | 1,670 MiB |
| 1페이지 번역 | **10,478 MiB** | 7,613 MiB |
| 2페이지 detection+ocr | 10,478 MiB | 3,345 MiB |
| 2페이지 인페인팅 (캡션+타일 22개) | 8,119 MiB | 5,770 MiB |
| 2페이지 번역 | **10,776 MiB** | 7,514 MiB |
| 전체 | **10,776 MiB** | 5,635 MiB |

p50은 11,819 MiB에서 **5,482 MiB**로 내려갔다.

- [x] **Step 5: 회귀 신호를 확인한다**

Expected:
- `retrying stage after memory pressure`가 0건이다.
- `CUDA0 model buffer size`가 6줄 이상이다.
- `rendered/`에 `page1.png`, `page2.png`가 있다.

결과:
- `retrying stage after memory pressure` **0건**. OOM 재시도는 없었다.
- `model_unload` **7건**. stage 전환 7회와 정확히 일치한다.
- `CUDA0 model buffer size` **6줄**: 688.14(OCR), 4,812.25(PUP), 6,390.19(gemma)가
  각각 2회. 페이지마다 재로드되었으므로 퇴거가 실제로 일어났다. 기준선은 4줄이었다.
- `rendered/page1.png`, `rendered/page2.png` 둘 다 있다.

**피크 10,776의 정체.** 스파이크는 번역 구간에만 있다. 퇴거 후 gemma 단독 상주
상태의 실측 footprint가 예상보다 크다:

| 항목 | MiB |
|---|---|
| `CUDA0 model buffer` (gemma) | 6,390 |
| `CUDA0 KV buffer` (non-SWA 44 + SWA 880) | 924 |
| `CUDA0 compute buffer` | 1,168 |
| CLIP vision + compute | ≈ 72 |
| 합계 (CUDA 컨텍스트 오버헤드 제외) | ≈ 8,554 |

계획의 "약 7.1–7.5 GB (gemma 6,390 + 오버헤드)"는 KV·compute 버퍼를 빠뜨린
추정이었다. 다른 stage가 전부 퇴거된 상태이므로 10,776은 이 정책에서 나올 수 있는
값에 가깝다. 12,069 → 10,776 감소 자체가 겹침 제거의 증거다.

**주의: 스크립트 추출기 버그.** `vram.log`의 첫 100줄(0–99)이 누락되어
`vram.log` 기반 피크는 10,794로 나온다(`stderr.log`·타임스탬프 교차검증 결과 실제
피크는 10,776). Task 8에는 타임스탬프 기준값을 쓴다.

- [x] **Step 6: 수치를 파일로 남긴다**

결과: `exit.txt` = `exit=0`, `wall=832s`. `pipeline finished in 801.22s`.

페이지당 단계 시간(타임스탬프 교차검증):

| 페이지 | 인페인팅 | 번역 |
|---|---|---|
| 1페이지 | 265.20s (타일 11개) | — |
| 2페이지 | 504.73s (타일 22개) | 503.58s |

---

## Task 8: FLUX 3 스펙의 게이트 수치를 실측으로 교체한다

현재 G4 행은 7,434MiB·5.2–5.5GB·RAM −6.8GB인데, 이는 stage 하나만 돌린 측정이다. 전체 파이프라인 실측과 다르다. G4 행을 실측치로 바꾸고 D4·D8을 개정된 결정에 맞춘다.

**Files:**
- Modify: `specs/2026-10-17-flux3-klein-inpainting-design.md`

- [ ] **Step 1: G4 행을 바꾼다**

Task 7에서 읽은 값으로 채운다. `{peak}`·`{단계별}`·`{픽셀}`은 Task 7 Step 4·Step 6의 출력이다.

변경 전:

```markdown
| G4 | 통과 | 프로세스 전체 VRAM 피크 7,434MiB(기준 856MiB 포함)는 캡션 단계에서 나왔다(FLUX 3 컨텍스트 상주 + 업샘플러). 인페인팅 단계는 5.2–5.5GB. 시스템 RAM 감소 약 6.8GB. 크롭 1개 기준 FLUX 3 로드 0.7–0.8초, 인페인팅 32–35초(16스텝), 캡션 약 11초. 최초 다운로드 12.36GB(Hub API) |
```

변경 후 (숫자를 Task 7의 실측값으로 바꿔 쓴다):

```markdown
| G4 | 통과 | 전체 파이프라인 2페이지 실행(순차 레지던시 적용 후)의 VRAM 피크는 `{peak}`MiB/12,288MiB이고, 단계별 최대는 번역 `{번역}`MiB, 인페인팅 phase 1 `{phase1}`MiB, phase 2 `{phase2}`MiB다. 페이지당 인페인팅 `{인페인팅}`초(타일 `{타일개수}`개), 번역 `{번역초}`초, 전체 `{전체}`초. 크롭 1개 기준 FLUX 3 로드 0.7–0.8초, 인페인팅 32–35초(16스텝), 캡션 약 11초. 최초 다운로드 12.36GB(Hub API) |
```

- [ ] **Step 2: D4와 D8을 개정된 결정에 맞춘다**

변경 전:

```markdown
| D4 | 스왑은 **`flux3_klein` 컴포넌트가 직접 소유**한다(접근법 A) | 타일 분할과 캡션의 결합은 이 모델에만 의미가 있다. 다른 stage와 `accelerator.rs`는 바꾸지 않는다 |
```

변경 후:

```markdown
| D4 | 스왑은 **`flux3_klein` 컴포넌트가 직접 소유**한다(접근법 A) | 타일 분할과 캡션의 결합은 이 모델에만 의미가 있다. **개정(2026-10-17)**: "다른 stage와 `accelerator.rs`는 바꾸지 않는다"는 이 결정은 무효다. stage 소유권은 그대로 두되, *언제 내리는가*는 `AcceleratorGate`가 정한다. `specs/2026-10-17-sequential-model-residency-design.md` |
```

변경 전:

```markdown
| D8 | 공용 VRAM↔RAM 레지던시 매니저(접근법 B)는 **후속 조사**로 둔다 | 범위가 크고, 런타임마다 "RAM 강등"의 의미가 다르다. `fork/VRAM-ACCOUNTING-DESIGN.md`의 미결 항목과 얽힌다 |
```

변경 후:

```markdown
| D8 | 공용 VRAM↔RAM 레지던시 매니저(접근법 B)는 **후속 조사**로 둔다 | 범위가 크고, 런타임마다 "RAM 강등"의 의미가 다르다. `fork/VRAM-ACCOUNTING-DESIGN.md`의 미결 항목과 얽힌다. **닫힘(2026-10-17)**: 2페이지 실측이 순차 레지던시를 필요로 함을 보였다. `specs/2026-10-17-sequential-model-residency-design.md`가 이 항목을 다룬다 |
```

- [ ] **Step 3: 확인한다**

Run: `git diff specs/2026-10-17-flux3-klein-inpainting-design.md`
Expected: G4 행, D4 행, D8 행만 바뀐다. `{...}` 자리 표시가 하나도 남지 않았는지 눈으로 확인한다.

- [ ] **Step 4: 커밋한다**

```bash
git add specs/2026-10-17-flux3-klein-inpainting-design.md
git commit -m "docs(specs): replace the stale G4 row with full-pipeline measurements

The old row measured one stage at a time (7434 MiB peak, 5.2-5.5 GB
inpainting). The full pipeline keeps detection, OCR and translation
resident, so the real figures come from the two-page run. Also mark D4 as
amended and D8 as closed by the residency design.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Task 9: 최종 확인

- [ ] **Step 1: 관련 테스트를 한 번 돌린다**

Run:
```bash
cargo test -p koharu-pipeline --lib accelerator::tests
cargo test -p koharu-pipeline --lib stages::inpainting::tests
cargo check -p koharu-pipeline -p koharu-app --bins
```
Expected: 모두 PASS·성공.

- [ ] **Step 2: 변경 범위를 확인한다**

Run: `git diff --stat main...HEAD`
Expected: `accelerator.rs`와 `stage_runner.rs`가 이 계획의 코드 변경 전부이고, 나머지는 문서·계측이다. 이 계획 밖의 파일이 보이면 되돌린다.

---

## 범위 밖 (하지 않는다)

| 항목 | 사유 |
|---|---|
| 예산 기반 퇴거, 모델 크기 API, Tier M/A/E | 설계 D12 |
| `ModelCell` API 변경 | 퇴거 시점만 바뀐다 |
| `koharu-diffusion`·`koharu-llama` 변경 | 퇴거는 로더 위 계층의 문제다 |
| FLUX 3 페이지 내 2단계 스왑 | 그대로 유지한다 |
| UI·i18n·문서(Phase E) | 별도 계획이다 |
| `packages/koharu/components/editor/usePageImportDrop.ts` | 이 계획과 무관한 미추적 파일이다 |
