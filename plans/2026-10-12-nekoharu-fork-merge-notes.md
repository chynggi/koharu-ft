# 메모 — NeKoharu 포크 병합 (fork `f9b22ba0`)

최종 갱신: 2026-10-12 (병합 커밋 `2bbafb25`, 브랜치 `merge/nekoharu`)

디스코드에서 발견한 별도 포크 [nekojin116/NeKoharu](https://github.com/nekojin116/NeKoharu)가
`bcac8a89`(upstream #1168) 위에 얹은 32개 커밋을 `main`에 병합했다. 포크는
Tauri invoke 브리지와 자체 브랜딩을 전제하므로 히스토리는 병합하되 **기능만 우리
구조로 옮겼다.** 아래는 무엇을 어디로 옮겼고, 어디서 의도적으로 갈라졌고, 무엇을
검증하지 않았는지에 대한 기록이다.

**아래 "확인 방법"은 재검증용이다. 다음 세션은 이 문서를 믿기 전에 명령을 다시 돌릴 것.**
특히 §5의 "실기 확인 안 함"은 이번 병합의 가장 큰 구멍이다.

---

## 1. 기능 → 착지점

| 기능 | 포크 커밋 | 우리 쪽 착지점 |
|---|---|---|
| 좌우 화살표 페이지 이동 | `15363dea` | `components/editor/PageRail.tsx` |
| 활성 페이지 자동 스크롤 | `fa0f6467` | 같음 (`scrollToIndex`) |
| 폰트 키보드 탐색 + 라이브 프리뷰 | `43a1cf95` `2843f84e` `2a5707a5` | `components/controls/FontPicker.tsx`, `components/editor/Inspector.tsx` |
| 페인트 도구 선택 시 cleanup 레이어 자동 선택 | `2a698a66` | `components/editor/ToolBar.tsx`, `CanvasWorkspace.tsx`, `lib/store.ts` |
| 선택 페이지만 내보내기 | `f166adba` | `ExportDialog.tsx`의 scope(전체/선택) — **이미 있었다** |
| CBZ 아카이브 내보내기 | `f166adba` | `commands/output.rs` `ExportFormat::Cbz` + `/pages/export*` + `ExportDialog` (§3.2) |
| 여러 페이지를 한 요청으로 번역 | `47e2d351` | `koharu-pipeline` `config.rs`/`scheduler.rs`/`execution.rs`/`stage_runner.rs`/`stages/` + `TranslationPreferences.tsx` |
| 텍스트 레이어 복사/붙여넣기 | `98bb6f0e` | `lib/layerClipboard.ts`, `commands/project.rs::paste_text_layers`, `/layers/paste` |
| 페이지 드래그 재정렬(CEF 드롭 보정) | `66c7fcc8` `f80018ef` `901d2b1f` `62a59c70` | `PageRail.tsx` — 기존 단순 `onDrop`을 대체 |
| 챕터 시작 표시 | `66c7fcc8` | `koharu-scene` `Page` revision 2, `commands/project.rs::set_page_chapter_start`, `/page/chapter-start` |
| 인접 페이지 스프레드 병합 | `beb45417` | `commands/project.rs::merge_pages`, `/pages/merge` |
| 사각형 드래그 레이어 다중 선택 | `148e0497` | `CanvasWorkspace.tsx`, `lib/geometry.ts` |
| 최근 색상 8개 스와치 | `90321acf` | `lib/colorHistory.ts`, `components/controls/ColorWell.tsx`, `CanvasWorkspace.tsx`(스포이드) |

확인 방법:

```sh
git log --oneline main..nekoharu/main      # 32개
git show --stat 15363dea 47e2d351 90321acf # 파일 단위 확인
grep -n 'setPageChapterStart\|mergePages\|pasteTextLayers' packages/bridge/src/protocol.ts
```

## 2. 받지 않은 것 (포크 소유)

- **브랜딩.** `README.md`(NeKoharu 정체성), `AboutDialog.tsx`(`NeKoharu` / fork author),
  `cliff.toml`(owner `nekojin116`), `.github/workflows/release.yml`(포크 릴리스 워크플로),
  `Cargo.toml`/`Cargo.lock`의 `0.84.0` 범프, `nekoharu.md`(포크 기능 목록)는 전부 되돌리거나 지웠다.
  `nekoharu.md`의 내용은 이 문서 §1 표가 대신한다.
- **업데이터 엔드포인트.** `crates/koharu/tauri.conf.json`의 `pubkey`/`endpoints`가
  `nekojin116/NeKoharu`로 바뀌는 변경은 **충돌 없이 자동 병합됐다.** 그대로 두면 우리
  업데이터가 포크 릴리스를 받는다. `git checkout HEAD --` 로 되돌렸으니 다음 병합에서도
  이 파일을 먼저 확인할 것.
- **Tauri invoke 브리지.** 포크의 `protocol.ts`는 specta 생성 invoke 클라이언트다. 우리는
  `fcdba40a`부터 손으로 쓴 fetch/SSE 클라이언트를 쓰므로 포크 쪽은 전부 버리고, 포크가
  추가한 **타입 필드만** 자동 병합분으로 남겼다(`combine_pages`, `max_pages_per_request`,
  `chapter_start` 등).
- **메뉴바 내보내기 하위 메뉴.** 포크의 `Export Project` → 형식별 항목 대신 우리
  `ExportDialog`를 확장했다(CBZ 체크박스).
- **포크의 프런트 테스트.** `editor-components.test.tsx`의 포크 쪽 테스트는 그들의 메뉴바와
  `commands.export(format, pages)` 시그니처를 전제하므로 받지 않았다. 대신
  `tests/components/export-dialog.test.tsx`에 CBZ 케이스 3개를 우리 UI 기준으로 추가했다.

확인 방법:

```sh
git diff 66b45279 2bbafb25 -- crates/koharu/tauri.conf.json Cargo.toml README.md   # 비어 있어야 한다
git remote -v | Select-String nekoharu                                             # 저장소만 남겨 둠
```

## 3. 의도적으로 갈라진 지점

### 3.1 커맨드 등록은 라우트로

포크는 `commands/bindings()`(tauri-specta `collect_commands!`)에 새 커맨드를 등록한다.
우리 트리에서 그 테이블은 `fcdba40a` 이후 죽은 코드라 병합 시 삭제했다(포크 쪽 수정분을
받지 않음). 대신 `koharu-rpc`에 라우트 3개를 새로 만들었다.

| 라우트 | 커맨드 |
|---|---|
| `POST /page/chapter-start` | `editing::set_page_chapter_start` |
| `POST /pages/merge` | `editing::merge_pages` |
| `POST /layers/paste` | `editing::paste_text_layers` |

`editing.rs`의 새 커맨드는 포크에서 `pub(crate)`이고 `canvas_channel.channel.publish`를
쓴다. 라우트가 다른 크레이트에서 부르므로 `pub`로, 발행은 우리 `CanvasChannel` 방식인
`canvas_channel.publish`로 고쳤다.

### 3.2 CBZ는 upstream #1095의 복구다

CBZ 내보내기는 **upstream `69e67f96`(PR #1095)에 이미 있던 기능**이고, 포크의 `f166adba`는
그것을 물려받아 선택 페이지 인자만 더한 커밋이다. 우리 트리에서는 `fcdba40a`
("use shared HTTP RPC host")가 내보내기 명령을 다시 쓰면서 그 경로를 통째로 지웠다.
즉 이번 병합은 upstream 기능을 되살리는 셈이다.

구현은 upstream과 다음 지점에서 갈라진다(우리 Job 파이프라인에 맞추기 위해서다).

| | upstream #1095 / 포크 | 우리 |
|---|---|---|
| 대상 지정 | 저장 파일 대화상자로 `.cbz` 경로를 직접 고른다 | 폴더를 고르면 그 안에 `<프로젝트 이름>.cbz` |
| 항목 이름 | 페이지 라벨 기반, 자릿수 `max(4, 자릿수)` | 사용자 패턴(`{index:04}_{label}`) 줄기 + `.png`. 기본값은 upstream과 같은 이름이 된다 |
| 쓰기 | 임시 파일에 쓰고 `persist` (원자적) | 대상 파일에 직접 쓰고 중단되어도 중앙 디렉터리를 써서 읽을 수 있게 남긴다 |
| PNG와 CBZ 동시 선택 | 프레임을 한 번만 렌더링해 바이트를 공유 | 아카이브가 별도 순차 패스라 **페이지를 두 번 렌더링한다** |
| 진행률 | 없음(동기) | 페이지×형식 + 아카이브 항목 수로 분모를 잡고 Job으로 발행 |

쓰기 방식의 차이는 우리 취소 규칙(`specs/2026-08-22-batch-export-design.md`: "취소 시 이미
쓴 파일은 지우지 않고 남긴다")을 따른 결과다. 두 번 렌더링은 아카이브가 페이지 순서를
지켜야 해서 병렬 스트림에 섞지 않았기 때문이며, 코드에도 같은 이유를 적어 두었다.

### 3.3 번역 배치는 포크 것 + 우리 소유권

포크는 `Translator::supports_vision(...)` 정적 호출을 쓴다. 우리는 `d48ffa05` 이후
`self.translator.supports_vision(...)` 인스턴스 메서드이므로 그 두 곳만 우리 것을 유지하고
배치 로직(`process_batch`, `batch_size`, 페이지 경계 지시문)은 포크 것을 받았다. 여러 페이지를
묶은 요청에는 이미지를 붙이지 않는다는 규칙도 그대로다.

## 4. 검증한 것

| 검사 | 결과 |
|---|---|
| `cargo check -p koharu-rpc -p koharu-pipeline` | 통과 |
| `cargo check -p koharu-scene -p koharu-app -p koharu-pipeline --all-targets` | 통과 (테스트 코드 포함) |
| `cargo test -p koharu-app --lib output::` | 4/4 (선택 페이지 해석, 아카이브 이름) |
| `cargo test -p koharu-app --lib project::` | 9/9 (스프레드 병합, 챕터 표시, 붙여넣기, 이동 순서) |
| `cargo test -p koharu-scene` | 35/35 |
| `cargo test -p koharu-pipeline` | 81/82 — 실패 1건은 §6, 이번 병합과 무관 |
| `bun x tsc --noEmit` (`packages/bridge`, `packages/koharu`) | 통과 |
| `bun x vitest run` (`packages/koharu`) | 15 files / 118 tests 통과 |
| `cargo clippy -p koharu-app -p koharu-rpc -p koharu-pipeline -p koharu-scene --all-targets` | 경고 4건 — 전부 병합 이전부터 있던 것(§6) |

## 5. 검증하지 않은 것

- **실기 동작.** 데스크톱 창에서 드래그 재정렬, CEF 드롭, 스프레드 병합, 챕터 표시,
  사각형 선택, 색상 스와치, CBZ 파일 열기를 **해보지 않았다.** vitest는 jsdom이라
  포인터/드래그 이벤트와 CEF 동작을 덮지 않는다.
- **CBZ 파일 자체.** 아카이브를 만들어 압축 해제해 보거나 뷰어로 열어보지 않았다.
  단위 테스트는 `resolve_export_pages`와 파일 이름만 덮는다. `write_cbz`는 렌더러·래스터라이저가
  필요한 경로라 단위 테스트가 없다.
- **번역 배치의 실제 LLM 요청.** `process_batch`가 만드는 지시문과 페이지 경계 표기가 실제
  모델에서 이름·줄거리 일관성을 올리는지는 확인하지 않았다. 스케줄러 단위 테스트(그룹이 다
  준비된 뒤에만 묶는지)까지가 검증 범위다.

## 6. 이번 병합과 무관한 기존 문제

**`koharu-pipeline` 테스트 1건 실패.**
`stages::inpainting::tests::a_hugging_face_override_round_trips_through_toml` 이
`Hugging Face revision is required: unsloth/FLUX.2-klein-4B-GGUF` 로 실패한다. 테스트는
`81e6ac07`(포크의 로컬 LLM·FLUX.2 Klein 작업)에서 들어왔고, `d48ffa05`(ComponentSource를
URL 소스까지 확장)가 리비전을 필수로 만든 뒤로 깨져 있다. 병합은 `stages/inpainting.rs`도
`koharu-ml`도 건드리지 않는다.

**clippy 경고 4건(CI는 `cargo clippy -- -D warnings`).** 병합 이전부터 있던 것으로,
이번 병합이 추가한 파일에서는 경고가 나오지 않는다.

| 위치 | 경고 | 들어온 커밋 |
|---|---|---|
| `koharu-ml/src/powerpaint/processor.rs:56` | `is_multiple_of` | 병합 이전 |
| `koharu-pipeline/src/config.rs:86` | `match` → `if let` | `933a466a` |
| `koharu-pipeline/src/stages/inpainting.rs:431` | variant 크기 차 | 병합 이전 |
| `koharu-app/src/lib.rs:17` | 복잡한 타입 | 병합 이전 |

확인 방법:

```sh
cargo test -p koharu-pipeline --lib stages::inpainting
cargo clippy -p koharu-app -p koharu-pipeline --all-targets 2>&1 | Select-String '^warning'
git log -S 'a_hugging_face_override_round_trips_through_toml' --oneline   # 81e6ac07
```

## 7. 다음 세션 메모

- 포크는 계속 갱신될 수 있다. `git fetch nekoharu && git merge nekoharu/main` — 이번 병합으로
  분기점이 포크 tip까지 올라갔으므로 다음 충돌은 §2의 브랜딩 파일에만 생겨야 한다.
  특히 `crates/koharu/tauri.conf.json`(업데이터)과 `README.md`, `AboutDialog.tsx`를 먼저 본다.
- 원격을 정리하려면 `git remote remove nekoharu`. 임시 워크트리(`M:/neko-fork`, `M:/neko-head`)는
  병합 후 제거했다.
- §5의 실기 확인이 남은 일의 전부다. 확인 순서는 (1) 페이지 레일 드래그/화살표,
  (2) 챕터 표시와 스프레드 병합, (3) CBZ 내보내기 파일 열기, (4) 색상 스와치와 사각형 선택.
