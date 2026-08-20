# HANDOFF — koharu-rpc 포팅

최종 갱신: 2026-08-20 (5차 세션 — 상태 재검증 + Tauri IPC 잔존 정리)

이 문서는 2·3·4차 세션 기록이 층층이 쌓이면서 서로 모순된 상태였다(상단은 "Phase 3c
완료", 58행은 "Phase 3c 아직 시작 안 함"). 5차 세션에서 각 항목을 트리에 대고 실제로
확인한 뒤 다시 썼다. **아래 표의 "확인 방법"은 재검증용이므로, 다음 세션은 문서를
믿기 전에 그 명령을 다시 돌릴 것.**

브랜치: 포팅 작업은 전부 `main`에 있다. `restore/koharu-rpc`는 main에 완전히 병합되어
고유 커밋이 0이므로(`git rev-list --left-right --count main...restore/koharu-rpc` → `2 0`)
삭제 후보다.

---

## 1. 완료된 것 (2026-08-20 확인)

| 항목 | 상태 | 확인 방법 |
|---|---|---|
| Phase 3a — 백엔드 HTTP 라우트 | 완료 | `ls crates/koharu-rpc/src/routes/` → 12개 모듈 |
| Phase 3b — 프론트엔드 HTTP 전환 | 완료 | `protocol.ts`에 Tauri invoke 호출 0건 |
| Phase 3c — 정적 서빙 + CEF 통합 | 완료 | `tauri.conf.json`의 `devUrl: http://127.0.0.1:47823` |
| CEF 실행 경로 결정 | 해결 (경로 B) | `vendor/tauri-runtime-cef/Cargo.toml`의 `default = []` |
| 토큰 인증 | 있음 | `crates/koharu/src/main.rs`의 `KOHARU_API_TOKEN` |
| 임시 파일 정리 | 완료 | `rpc-pid.txt`, `target/debug/*-out.log` 전부 없음 |

### CEF 실행 경로 — 어떻게 끝났나

3차 세션의 `RunDeElevated` 진단(관리자 권한 셸에서 CEF가 뜨지 않음)은 유효했고,
최종 해법은 **경로 B(no-sandbox 단일 exe)** 였다. `vendor/tauri-runtime-cef`의
`default` feature를 `[]`로 패치해 `sandbox`를 끈다. 이 패치는 진단용 실험이 아니라
**현재 유지되어야 하는 구성**이다 — vast.ai 같은 마켓플레이스 템플릿은 docker
capability(`--cap-add=SYS_ADMIN`)를 표현할 수 없으므로, 샌드박스를 끄는 쪽이 배포
가능한 유일한 경로다. (3차 세션 기록의 "가설 확정 시 vendor 패치를 되돌릴 것"은
따라서 **무효**다.)

### 토큰 인증

- `KOHARU_RPC_HOST`가 비-루프백이면 `KOHARU_API_TOKEN`이 필수. 없으면 경고.
- 토큰은 이미지에 굽지 않는다. `koharu-rpc`가 기동 시 서빙하는 `index.html`의
  `<head>`에 `window.__KOHARU_API_TOKEN__`을 주입하고, `protocol.ts`가 그 전역을 읽는다
  (`crates/koharu-rpc/src/api.rs`의 `index_with_token`).
- API base URL은 상대경로 `/api/v1`이라 CEF 창과 원격 브라우저가 같은 번들을 쓴다.
- **config 라우트가 provider 시크릿(키링)을 다룬다.** 비-루프백 바인드 시 토큰 없이
  노출하는 것은 금지.

### Tauri IPC 잔존 정리 (5차 세션)

4차 세션이 "별도 후속 정리 대상"으로 남긴 항목. 실제로는 단순 정리가 아니라 **지뢰**였다:
`crates/koharu-app/src/bin/generate.rs`가 tauri-specta로 `packages/bridge/src/protocol.ts`를
**덮어쓰는** 바이너리였는데, 그 파일은 Phase 3b에서 손으로 쓴 HTTP 클라이언트로 바뀌어
있었다. `cargo run --bin generate` 한 번이면 Phase 3b 전체가 날아간다.

제거한 것 (호출자 0건 확인 후):
- `crates/koharu-app/src/bin/generate.rs`
- `commands::bindings()` 및 `app.rs`의 `.invoke_handler(bindings().invoke_handler())`
- `lib.rs`의 `pub use commands::bindings`
- 위 제거로 고아가 된 `tauri-specta`, `specta-typescript` 의존성 (workspace 포함)

**남긴 것과 그 이유:**
- `#[tauri::command]` 51개 — 제거 불가. `koharu-rpc`의 `error.rs::ipc_bytes()`가
  `tauri::ipc::IpcResponse`로 바이너리 응답(`CanvasBytes`/`ThumbnailBytes`/
  `FontPreviewBytes`)을 언랩하므로 **실사용 중**이다.
- `#[specta::specta]` 51개 — `bindings()` 제거로 소비자가 사라져 이제 죽은 속성이다.
  51개소 기계적 제거는 별도 패스로 남긴다.
- `@tauri-apps/plugin-{opener,process,updater}`, `@tauri-apps/api/window` — 플러그인
  API이지 우리 커맨드 IPC가 아니다. `WindowChrome`/`Updater`가 `__TAURI_INTERNALS__`
  런타임 감지로 조건부 사용하는 정상 데스크톱 기능.

---

## 2. 남은 작업

### A. importPages / exportPages 브라우저 갭 (우선순위 높음)

원격 브라우저에는 네이티브 파일 다이얼로그가 없다. 현재 `protocol.ts`에서:

```ts
importPages: (_source: PageImportSource) => post<null>("/pages/import", { files: [] }),
exportPages: (pages, format, directory = "") => post<null>("/pages/export", { ... }),
```

빈 경로를 보내 서버에서 검증 실패한다. **참고할 선례가 생겼다** — `pickGgufFile`이
같은 문제를 만나 이렇게 풀었다(`crates/koharu-rpc/src/routes/llm.rs`):

- 루프백 호출자에게만 네이티브 다이얼로그를 열고, 원격에는 `null` 반환
- UI가 `null`을 받으면 서버 측 경로를 직접 입력받는 필드로 전환

import/export도 같은 모양이 맞는지, 아니면 실제 업로드/다운로드 플로우가 필요한지가
설계 결정 지점이다. **주의:** 컨테이너에서 CEF 창은 headless Xvfb 위에 있으므로,
원격 호출자에게 다이얼로그를 열면 아무도 볼 수 없는 화면에 떠서 요청이 영영 반환되지
않는다. 루프백 판정은 `ConnectInfo<SocketAddr>`로 하며, 이를 위해 `lib.rs`가
`into_make_service_with_connect_info::<SocketAddr>()`를 쓴다.

### B. 제거된 upstream 기능 재도입 (별도 이슈, 착수 전 필요 여부 판단)

Phase 2에서 fork에 대응 구현이 없어 삭제한 것들:

| 기능 | 원래 위치 | 비고 |
|---|---|---|
| MCP 서버 | `mcp/` (rmcp) | 수요 확인 필요 |
| PSD export | `psd_export.rs` | fork에는 `koharu-psd` 크레이트가 따로 있음 |
| downloads / history / pipelines 라우트 | `routes/*.rs` | fork 대응 구현 여부부터 확인 |
| OpenAPI 문서 바이너리 | `bin/openapi.rs` | utoipa 의존성 재도입 필요 |

`routes/llm.rs`는 5차 세션에서 다시 생겼다(포크 작업의 로컬 LLM 설정 UI가 필요로 함).
나머지는 아직 없다.

### C. `#[specta::specta]` 51개 정리 (낮음)

`bindings()` 제거로 고아가 된 속성. 기계적이지만 51개소 × 11파일이라 별도 패스.

### D. `restore/koharu-rpc` 브랜치 삭제 (낮음)

main에 완전히 병합됨.

---

## 3. 검증 명령 (이 저장소 기준)

`-j 6`은 필수다 — 무제한 병렬은 이 머신(12코어)의 데스크톱을 얼린다.

```
cargo check -j 6 --workspace
cargo test -j 6 -p koharu-pipeline --lib      # 52/52
cargo test -j 6 -p koharu-ml --lib            # 44 passed, 16 ignored
bunx tsc --noEmit -p packages/koharu/tsconfig.json
bunx oxlint packages/koharu
bun run --filter '@koharu/app' test           # 83/83 across 12 files
```

위 수치는 **`main` 기준**이다. `fork/vram-accounting` 브랜치는 자체 테스트를 더해
52→65, 44→61, 83→84가 되므로 두 브랜치의 숫자를 섞지 말 것.

트리의 유일한 기존 경고는 `koharu-torch-sys`의 빌드 스크립트 `linker_messages`다.
우리 것이 아니다.

**한 번도 검증되지 않은 것:** 실제 원격 브라우저로 vast.ai 배포본에 붙어 프로젝트를
처리하는 end-to-end 실행. 4차 세션의 Edge 검증은 로컬 `bun run dev` 기준이었다.

---

## 4. 참고 — 구조

### koharu-rpc 파일 구성

```
crates/koharu-rpc/
├── Cargo.toml
└── src/
    ├── lib.rs      # serve(AppHandle<Cef>, host, port, static_dir, api_token)
    ├── api.rs      # router 조립 + 토큰 미들웨어 + index_with_token 주입
    ├── error.rs    # ApiError/ApiResult, ipc_bytes()
    └── routes/
        ├── mod.rs
        ├── meta.rs        # 서버 메타데이터
        ├── config.rs      # 설정 + 번역 모델 목록
        ├── llm.rs         # LLM capabilities + GGUF 등록
        ├── projects.rs    # 프로젝트 CRUD
        ├── pages.rs       # 페이지 조회/편집
        ├── layers.rs      # 레이어 편집
        ├── canvas.rs      # 캔버스 매니페스트/리소스/커밋
        ├── operations.rs  # 파이프라인 실행
        ├── fonts.rs       # 폰트 목록/미리보기
        ├── agent.rs       # 에이전트 로그인/실행 (SSE)
        └── events.rs      # GET /events — 5종 SSE 멀티플렉싱
```

### 임베딩 방식

`koharu-app`은 독립 `App` 상태 객체가 없고 모든 것이 `AppHandle<Cef>`에 `manage()`된
상태 + `#[tauri::command]` 함수다. 따라서 koharu-rpc는 별도 프로세스가 아니라
**데스크톱 프로세스 내 임베디드 HTTP 서버**이며, 라우트가 `AppHandle<Cef>`를 axum
state로 받아 같은 상태/명령에 접근한다.

- 진입점: `koharu_app::extend_setup(hook)` — Tauri setup 단계에 사이드 서비스를 부착
- `crates/koharu/src/main.rs`가 `koharu_rpc::serve(...)`를 등록. 기본 포트 `47823`
  (`DEFAULT_RPC_PORT`), `KOHARU_RPC_PORT`로 override
- 정적 프론트엔드는 `tower_http::ServeDir`로 서빙 (`KOHARU_STATIC_DIR`로 override).
  디렉터리가 없으면 경고만 남기고 `/api/v1`은 계속 동작

라우트에서 명령 함수를 호출할 때는 시그니처가 Tauri 그대로이므로
`handle.state::<T>()`로 상태를 얻어 넘긴다 (상태 타입은 `pub`).

### 알려진 한계

`KOHARU_RPC_PORT`를 override하면 `tauri.conf.json`의 `devUrl`(고정 47823)과 어긋난다.
`.navigate()` best-effort 보정이 있으나, `tauri-runtime-cef`의 `navigate_first_webview`
(`vendor/tauri-runtime-cef/src/runtime.rs:834`)가 **CEF 브라우저 생성 전이면 조용히
no-op**이라 레이스가 재발할 수 있다.

---

## 5. 이전 세션 원본 기록

2·3차 세션의 상세 조사 기록(bootstrap two-binary 가설, `RunDeElevated` 진단 과정)은
git 히스토리에 있다: `git log --oneline -- HANDOFF.md`. 결론은 §1에 반영했고, 폐기된
가설을 본문에 남겨두면 다음 세션이 잘못된 방향으로 다시 파고들 위험이 있어 옮겼다.
