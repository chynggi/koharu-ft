# HANDOFF — 일괄 내보내기 (`feat/batch-export`)

최종 갱신: 2026-08-22 (정리 세션 — §3 로케일 누락 해결로 테스트 98/98, 문서 정정 완료.
남은 것은 §4 실제 앱 확인뿐)

**아래 표의 "확인 방법"은 재검증용이다. 다음 세션은 이 문서를 믿기 전에 그 명령을 다시
돌릴 것.** 특히 §4는 아직 아무도 실행해 보지 않은 코드에 대한 주장이다.

브랜치: `feat/batch-export`, `main` 위 15커밋. `main`에는 별개로 커밋된 SSE/라우팅 버그
수정(`809b8361`)이 있고 이 브랜치는 그 위에 있다.

관련 문서:
- 설계: `specs/2026-08-22-batch-export-design.md`
- 계획: `plans/2026-08-22-batch-export.md`

---

## 1. 무엇을 만들었나

프로젝트의 페이지를 PNG·PSD로 한 번에 내보내는 기능. 네 가지가 목표였다.

1. 선택 작업 없이 전체 페이지를 한 번에
2. 진행률 표시와 중단
3. 형식·대상·파일명·폴더 구조를 한 화면에서
4. PNG와 PSD를 한 번의 실행으로 함께

내보내기를 **기존 Job 인프라 위의 작업으로** 만들었다. `Processing`(단일 작업 제약),
`JobChannel`(SSE 브로드캐스트), `/process/stop`(취소), `ActivityCenter`(진행률 UI)가 이미
있었으므로 새로 만든 개념은 파일명 템플릿 모듈과 브라우저 ZIP용 스테이징 슬롯 둘뿐이다.

따라서 **번역 파이프라인이 도는 중에는 내보내기가 거부된다.** 둘 다 GPU를 많이 쓰므로
의도된 제약이다.

---

## 2. 완료된 것 (2026-08-22 확인)

| 항목 | 상태 | 확인 방법 |
|---|---|---|
| 파일명 템플릿 (`{index}`, `{index:04}`, `{label}`) | 완료 | `cargo test -p koharu-app naming` → 11 passed |
| 다중 형식·하위 폴더·협조적 취소 | 완료 | `crates/koharu-app/src/commands/output.rs`의 `export_pages_to` 시그니처 |
| 내보내기 Job 래퍼 | 완료 | 같은 파일의 `start_export`; `Job.kind == JobKind::Export` |
| ZIP 재귀 (하위 폴더 보존) | 완료 | `cargo test -p koharu-rpc archive` → 3 passed |
| 라우트 4개 + 2단계 다운로드 | 완료 | `crates/koharu-rpc/src/routes/pages.rs`의 `export_download_archive` |
| `protocol.ts` 갱신 | 완료 | `bun run --filter @koharu/bridge typecheck` → exit 0 |
| `runExport`/`finishExport` 분리 | 완료 | `packages/koharu/tests/lib/transfer.test.ts` → 5 passed |
| 내보내기 대화상자 | 완료 | `packages/koharu/tests/components/export-dialog.test.tsx` → 4 passed |
| `ActivityCenter` 라벨 | 완료 | `job.kind === 'export'` 분기 (`ActivityCenter.tsx`) |
| 로케일 7개 | 완료 | `git show af24401c --stat` |
| 로케일 9개 | 완료 | `bun run test` → 98 passed |

### 검증 명령 한 줄 요약

```bash
cargo test -p koharu-app -p koharu-rpc        # 19 + 10 passed
cd packages/koharu && bunx tsc --noEmit       # 출력 없음
cd packages/koharu && bun run test            # 98 passed
cd packages/koharu && bun run lint            # clean
```

**`bun run test`만으로는 부족하다.** 이 패키지의 테스트는 bridge 명령을 목으로 대체하므로
타입이 어긋난 호출이 런타임을 그냥 통과한다. 이 브랜치의 타입 깨짐 두 건(내보내기
시그니처 변경, `Job.kind` 추가)은 전부 vitest에 안 잡히고 `tsc`에만 잡혔다. 반드시 함께
돌릴 것.

`bun run check`(oxfmt)는 실패하지만 **이 브랜치와 무관하다.** 베이스 브랜치에서도 같은
~160개 파일(대부분 `packages/ui/**`)이 걸린다. `git stash` 후 재실행으로 확인했다.

---

## 3. [해결됨] 깨져 있던 것 — 로케일은 아홉 개였다

**해결됨 (정리 세션):** `zh-CN`·`zh-TW`에 아래 14개 키를 채우고 `menu.exportPng`·
`menu.exportPsd`를 지워 `bun run test` → 98/98, `bunx tsc --noEmit` 클린. `specs/`·`plans/`의
"7개 로케일" 표현도 9개로 정정했다. 아래는 당시 원인 기록이다.

`bun run test` → `tests/lib/localization.test.ts > defines the same flattened translation
schema in every locale` **실패 (97/98).**

`packages/koharu/public/locales`에는 로케일이 **아홉** 개 있고 `lib/i18n.ts:8-21`이 전부
등록한다. `zh-CN`과 `zh-TW`가 빠진 채 설계·계획·구현 지시가 전부 "7개"로 쓰였다.

원인은 이 세션에서 목록을 확인할 때 `ls ... | head`로 출력이 잘린 것을 알아채지 못한
것이다. 잘못된 숫자가 문서 세 곳과 태스크 지시문에 그대로 전파됐다.

**할 일:** `zh-CN`, `zh-TW`에 아래 14개 키를 중국어로 채운다. 나머지 일곱 로케일의
`export` 블록(`af24401c`)을 그대로 본뜨면 된다.

```
menu.export
activity.exporting, activity.exportFailed
export.{title, description, scope, scopeAll, scopeSelected, formats,
        pattern, subfolders, start, cancel}
export.patternError.{unclosed, token, width, separator}
```

두 파일에서 `menu.exportPng`·`menu.exportPsd`도 지워야 한다 — 참조하는 코드가 없다.

주의: i18next는 `{{...}}`만 보간한다. 오류 문구 안의 `{index}`·`{label}`은 literal이고
이스케이프가 필요 없다. `{{count}}`는 보간이므로 이중 중괄호를 유지할 것.

문서도 함께 고칠 것: `specs/`와 `plans/`의 "7개 로케일" 표현.

---

## 4. 아직 아무도 실행해 보지 않았다

**이 기능은 단 한 번도 실제로 동작하는 것이 확인되지 않았다.** 테스트는 통과하지만
`export_pages_to`의 새 경로, Job 배선, 2단계 다운로드는 전부 정적 검증만 거쳤다.

koharu.exe는 **관리자 권한 터미널에서 CEF 런타임을 찾지 못한다**(HANDOFF.md의 경로 B
참조). 에이전트 셸이 그 환경이라 이 세션에서는 서버를 띄울 수 없었다. 사람이 직접
띄워야 한다.

```bash
cargo build --release
bun run ui:build          # packages/koharu/out 재생성 — protocol.ts 변경이 여기 있다
# koharu.exe는 사람이 일반 권한 터미널에서 실행
```

확인할 것:

| 확인 항목 | 기대 |
|---|---|
| 전체 페이지를 PNG+PSD로, 하위 폴더 켜고 내보내기 | `png/`, `psd/`가 생기고 각 페이지가 같은 번호를 공유 (`0001_x.png`, `0001_x.psd`) |
| 내보내는 동안 `ActivityCenter` | "내보내는 중" + 진행률 + 중단 버튼 |
| 중단 버튼 | job이 `stopped`가 되고, **이미 쓴 파일은 남는다** (의도된 동작) |
| 브라우저(`http://127.0.0.1:47823`)에서 내보내기 | ZIP이 하위 폴더 구조를 보존 |
| 파이프라인 실행 중 내보내기 시도 | 폴더 선택창이 **뜨지 않고** 즉시 "another process is already running" |
| 잘못된 패턴 입력 | 대화상자에서 즉시 오류, 시작 버튼 비활성 |
| `curl -s -o /dev/null -w '%{http_code}' .../api/v1/pages/export/download/<없는-job>` | `404` |

마지막 항목이 특히 중요하다. `ApiError`는 모든 오류를 500으로 매핑하므로, 이 라우트만
명시적으로 404를 만든다(`d966a2eb`). 회귀하면 취소된 내보내기와 서버 고장을 클라이언트가
구분하지 못한다.

---

## 5. 설계상 알고 있는 절충

기록해 두는 것이지 버그가 아니다.

- **취소해도 이미 쓴 파일은 지우지 않는다.** 되돌리려면 어느 파일이 이번 실행 소산인지
  추적해야 하는데, 사용자가 기존 파일을 덮어썼을 수 있어 안전하게 되돌릴 수 없다.
- **실패·취소된 브라우저 내보내기의 스테이징은 남는다.** 다음 내보내기가 시작될 때
  교체되며 지워진다. 작업 종료 즉시 지우려면 `start_export`가 스테이징 슬롯을 알아야
  하는데, 그 결합을 피하려고 슬롯을 라우트 계층에 뒀다. 남는 것은 언제나 최대 한 개다.
- **`ExportFormat::extension()`은 `subfolder()`만 쓴다.** 파일명은 여전히 `"{stem}.png"`
  처럼 하드코딩돼 있다. 정리 대상이지만 이번 범위 밖으로 뒀다.
- **PNG 투명 배경은 넣지 않았다.** 검토 중 전제가 틀린 것으로 확인됐다 — 래스터라이저는
  이미 RGBA를 내고, "투명 배경"은 사실상 원본 에셋 레이어를 제외한 오버레이 렌더링을
  뜻하게 된다. 그런데 PNG 출력물 자체가 번역·조판 결과물이므로 현재 기본값이 정확히
  의도한 산출물이다. 자세한 근거는 설계 문서 "범위 밖" 절에 있다.

---

## 6. 다음 세션이 할 일

1. [x] `zh-CN`·`zh-TW` 번역 추가 → `bun run test` 98/98 (§3)
2. [x] `specs/`·`plans/`의 "7개 로케일" → 9개로 정정
3. [ ] 실제 앱에서 §4 표를 위에서 아래로 확인
4. [ ] §3 해결분과 문서 정정을 커밋하고 `main`에 병합
