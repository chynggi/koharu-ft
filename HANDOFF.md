# HANDOFF — koharu-rpc 포팅 (restore/koharu-rpc 브랜치)

최종 갱신: 2026-08-18 (4차 세션 — Phase 3a/3b/3c: 원격 브라우저 UI 접근 + CEF HTTP 통합 완료)

## Phase 3c 완료 (CEF 창도 HTTP UI로 통합)

- `koharu-rpc`가 이제 항상 기동됨 (더 이상 `KOHARU_RPC_PORT` 옵트인 전용 아님).
  기본 포트 `47823` (`crates/koharu/src/main.rs`의 `DEFAULT_RPC_PORT`), `KOHARU_RPC_PORT`로 override 가능.
- 정적 프론트엔드(`packages/koharu/out/`)를 `tower_http::ServeDir`로 `koharu-rpc`가 직접 서빙
  (`KOHARU_STATIC_DIR`로 override 가능, 기본값은 `CARGO_MANIFEST_DIR` 기준 상대경로).
  디렉터리가 없으면 경고 로그만 남기고 `/api/v1`은 계속 동작 (non-fatal).
- **중요 버그 수정**: 최초 구현은 `WebviewWindowBuilder::build()` 직후 `.navigate()`로 CEF 창을
  HTTP URL로 이동시키려 했으나, `tauri-runtime-cef`의 `navigate_first_webview`
  (`vendor/tauri-runtime-cef/src/runtime.rs:834`)가 **CEF 브라우저가 아직 생성 안 됐으면 조용히
  no-op** 하는 구조라 레이스 컨디션으로 항상 실패(구 devUrl `localhost:3000`이 그대로 로드되어
  "connection refused" 표시). **해결**: `crates/koharu/tauri.conf.json`의 `devUrl`을 아예
  `http://127.0.0.1:47823`(DEFAULT_RPC_PORT와 동일)로 바꿔서, 레이스 없이 창 생성 시점에 바로
  올바른 URL을 로드하도록 함. `.navigate()` 호출은 `KOHARU_RPC_PORT` override 시의 best-effort
  보정으로 남겨둠 (override 시에는 동일한 레이스가 재발할 수 있음 — 알려진 한계).
- **실증 검증**: 관리자 권한 아닌 셸에서 `koharu.exe`를 env var 없이 실행 → CEF 창이
  `http://127.0.0.1:47823/`을 로드해 실제 "Preparing Koharu" 초기화 화면이 정상 표시됨을
  스크린샷으로 확인 (이전엔 `ERR_CONNECTION_REFUSED` 표시).
- 남은 것: `commands/mod.rs::bindings()`의 tauri-specta IPC 등록, `@tauri-apps/api` 등
  Tauri IPC 관련 코드는 의도적으로 미제거 (별도 후속 정리 대상, 이번 세션 스코프 아님).

## 4차 세션 요약

3차 세션에서 관리자 권한 가설이 확정된 후, 사용자의 실제 요구사항이 "원격 브라우저로
koharu UI 접근"임이 확인됨. 계획(`C:\Users\<user>\.claude\plans\lucky-sniffing-hopper.md`)에
따라 Phase 3a(백엔드 SSE+라우트)와 Phase 3b(프론트엔드 HTTP 트랜스포트 전환)를 완료.

**Phase 3a (백엔드) — 완료, curl로 검증됨**
- `koharu-app`의 33개 `#[tauri::command]` 전부에 `koharu-rpc` HTTP 라우트 추가.
- `CanvasChannel`/`JobChannel`/`ProjectChannel`에 `tokio::sync::broadcast::Sender<T>`를
  병행 추가 (기존 Tauri `Channel` 유지, 파괴적 변경 없음).
- `GET /api/v1/events` — canvas/job/download/resource/project 5종 SSE 멀티플렉싱, 실동작 확인.
- `POST /api/v1/agent/login`, `/agent/run` — `tauri::ipc::Channel::new()`를 콜백으로 감싸
  webview 없이도 스트리밍 가능하게 브릿지 (`routes/agent.rs`).
- `tower-http`(fs/cors/trace), `tokio-stream` 워크스페이스 의존성 추가.

**Phase 3b (프론트엔드) — 완료, 실브라우저(Edge)로 검증됨**
- `packages/bridge/src/protocol.ts`를 tauri-specta 생성 코드 → 손으로 작성한 fetch 기반
  클라이언트로 전환 (`commands.*` export 형태 유지, 호출부 무변경).
- `openEventStream()` 신설(EventSource 기반), `providers.tsx`가 `subscribe()` +
  `openEventStream()`으로 초기 스냅샷/실시간 갱신 분리.
- `WindowChrome`/`Updater`는 `__TAURI_INTERNALS__` 런타임 감지로 조건부 렌더링,
  `TitleBar`는 `window.open`, `AboutDialog`는 `/api/v1/meta` 재사용.
- 테스트 83/83 통과, `next build` 정상.
- **실증 검증**: `bun run dev`(NEXT_PUBLIC_API_BASE_URL로 koharu-rpc 지정) + 순정
  Edge 브라우저(CEF/Tauri 아님)로 `http://127.0.0.1:3000` 접속 → 실제 프로젝트 목록
  (KeiLove/아오바/AVTranslate)이 HTTP만으로 정상 렌더링됨을 스크린샷으로 확인.

**알려진 갭 (후속 필요)**
- `importPages`/`exportPages`: 브라우저에는 네이티브 파일 다이얼로그가 없어 현재
  플레이스홀더 상태(빈 경로 전송 → 서버에서 검증 실패). 브라우저 파일/폴더 선택 UI
  또는 업로드 플로우 설계가 별도로 필요 (Phase 3b 스코프 밖으로 명시적으로 flag됨).

**Phase 3c (정적 서빙 + CEF도 HTTP로 통합) — 아직 시작 안 함**
- 계획서의 보안 결정 지점(바인드 주소 확장 여부 + 인증)을 진입 전 반드시 재확인할 것.
- `next.config.ts`에 `trailingSlash` 미설정 — 정적 export 결과물의 실제 디렉터리
  구조(`route.html` vs `route/index.html`)를 빌드 후 직접 확인하고 `ServeDir` 폴백
  필요 여부 판단할 것 (가정 금지).

---

## (3차 세션 기록 — 아래는 관리자 권한 가설 규명 원본 기록)

## ⚠️ 3차 세션 요약 (최우선으로 읽을 것)

**이전 세션들의 "CEF sandbox / bootstrap.exe two-binary" 가설은 근본 원인이 아니었던 것으로 보인다.**

검증 과정:
1. `vendor/tauri-runtime-cef/`에 `tauri-runtime-cef`(git `feat/cef`#8c204b8)를 로컬 vendoring하고,
   루트 `Cargo.toml`에 `[patch."https://github.com/tauri-apps/tauri"]`로 연결해
   `cef` crate의 `sandbox` 기본 feature를 실제로 껐다 (`no_sandbox=1`이 되도록).
   → **빌드는 성공했지만 실행 실패는 그대로 재현됨.** sandbox on/off는 원인이 아니었다.
2. `cef::Settings`에 `log_severity = VERBOSE`, `log_file = <cache>/koharu-cef-debug.log`를
   추가해 CEF 내부 로그를 직접 받아본 결과 (`vendor/tauri-runtime-cef/src/runtime.rs`의
   `CefRuntime::init` 내 `settings` 구성 참조):
   ```
   RunDeElevated: Started process, PID: 31572
   RunDeElevated: ::AllowSetForegroundWindow failed: 액세스가 거부되었습니다. (0x5)
   ```
   **`RunDeElevated`는 Chromium/CEF가 "브라우저 프로세스가 관리자 권한(elevated)으로 떠 있음"을
   감지했을 때 비-elevated 자식 프로세스로 위임(de-elevate)하려는 내장 보안 메커니즘이다.**
   이 위임이 `AllowSetForegroundWindow` 거부로 실패 → 원 프로세스의 `cef::initialize`가
   실패를 반환 → "Could not find the webview runtime" 패닉으로 이어짐.
3. **결정적 증거**: 현재 머신에 이미 설치되어 있던 정식 배포 0.73.0 release 빌드
   (`C:\Users\<user>\AppData\Local\koharu\koharu.exe`, restore/koharu-rpc 브랜치와 무관,
   CEF 프로필 디렉터리에 실사용 이력이 있는 known-working 빌드)를 **같은(관리자 권한) 셸에서
   실행하면 동일하게 실패**한다. 즉 koharu-rpc 포팅이나 sandbox feature와 무관하게,
   **"명령을 실행하는 셸/세션이 관리자 권한(High Mandatory Level)이면 CEF 기반 koharu가
   구조적으로 뜨지 않는다"**는 것이 원인일 가능성이 매우 높다.
   (`whoami /groups`로 `Mandatory Label\High Mandatory Level` 확인됨)

**미검증 항목 (다음 세션에서 최우선으로 확인)**
- [ ] koharu.exe를 **비-elevated 일반 권한 셸**에서 실행했을 때 정상 기동하는지 (사용자 직접 확인 필요 —
  이 조사를 수행한 Claude Code 세션 자체가 elevated라서 셸 안에서 자체 검증 불가).
  성공하면 가설이 100% 확정되고, sandbox 패치(vendor/tauri-runtime-cef)는 불필요하므로 되돌려야 함.
- [ ] 만약 비-elevated에서도 실패한다면 sandbox 패치 방향(경로 A/B)으로 재복귀해서 조사 계속.

**정리 필요 (가설 확정 시)**
- `vendor/tauri-runtime-cef/` 및 루트 `Cargo.toml`의 `[patch."https://github.com/tauri-apps/tauri"]`
  블록은 실험적 진단용으로 추가한 것 — sandbox가 원인이 아니었다면 되돌리는 것을 고려.
  단, `log_severity`/`log_file` 진단 코드는 유용하니 유지 여부는 별도 판단.
- `KOHARU_RPC_PORT` 자체 기능(코드 변경)은 이번 세션에서 건드리지 않음 — 여전히 2차 세션 기준 유효.

---

## (2차 세션 기록 — 아래는 sandbox 가설 기준 원본 기록, 위 3차 세션 결과로 대체됨)

## 1. 목표

upstream `mayocream/koharu` 커밋 `092fee3`의 `koharu-rpc`(HTTP API 서버)를
본 fork `chynggi/koharu-ft`에 포팅한다. fork는 upstream 0.44.6과 달리
Tauri(CEF) 데스크톱 아키텍처(v0.73.0)로 재작성되어 있어 직접 복붙이 불가능하므로,
**A안: 어댑터 계층** 방식으로 재작성하기로 함 (사용자 승인).

## 2. 브랜치 / 커밋 상태

- 브랜치: `restore/koharu-rpc` (origin에 푸시됨)
- 커밋 `299582c0`: Phase 1 — upstream에서 `koharu-rpc`, `koharu-core` 그대로 vendoring + workspace 등록 (빌드 실패 상태로 커밋됨. 빌드 가능 상태를 만든 뒤 squash/정리 권장)
- **이후 Phase 2 변경사항은 아직 미커밋** (`git status` 참조)
- 제거 예정 잔여물: 루트의 `rpc-pid.txt` (untracked, 삭제 필요)

## 3. Phase 2에서 내린 설계 결정 (커밋되지 않은 working tree)

### 접근 방식: Tauri 임베디드 서버

fork의 `koharu-app`은 독립 `App` 상태 객체가 없고, 모든 기능이
Tauri `AppHandle<Cef>`에 `manage()`된 상태 + `#[tauri::command]` 함수로 존재:

- 상태: `CurrentProject`, `ProjectLibrary`, `Processing`, `Desktop`, `AgentState`, 각종 `*Channel`, `Initialization`
- 명령 모듈: `commands/{lifecycle,project,editing,processing,preferences,fonts,canvas,import,output,agent}.rs`

따라서 koharu-rpc를 **별도 프로세스가 아닌 데스크톱 프로세스 내 임베디드 HTTP 서버**로 만들고,
라우트가 `AppHandle<Cef>`를 axum state로 받아 동일 상태/명령에 접근하게 함.

### 변경 내역

**`crates/koharu-app` (fork 측 최소 변경)**
- `commands/*.rs`: `pub(crate)` → `pub` 전체 치환 (라우트에서 직접 호출하기 위함)
- `src/lib.rs`:
  - `pub mod commands` 공개 (기존 `pub use commands::bindings` 제거 — `app.rs:89`만 내부 사용)
  - `SETUP_HOOKS` 정적 + `pub fn extend_setup(hook)` / `pub(crate) fn take_setup_hooks()`
    추가 — 임베더가 Tauri setup 단계에 사이드 서비스를 부착하는 공식 진입점
- `src/app.rs`: setup 내 `Desktop`/`AgentState` manage 직후 `take_setup_hooks()` 루프 삽입

**`crates/koharu-rpc` (upstream 코드에서 어댑터로 재작성)**
- 삭제: `mcp/`(rmcp), `psd_export.rs`, `binary.rs`, `events.rs`, `server.rs`,
  `bin/openapi.rs`, `routes/{downloads,history,llm,pipelines}.rs`
  — fork에 대응 구현이 없는 것들. 필요 시 후속 작업.
- 유지·재작성: `lib.rs`, `api.rs`, `error.rs`, `routes/{mod,config,meta,projects,pages,operations,fonts}.rs`
- `lib.rs` 핵심 구조:
  - `pub type AppState = AppHandle<Cef>;`
  - `pub fn serve(app: AppHandle<Cef>, port: u16)` — `tokio::spawn`으로 `127.0.0.1:port` 바인드 후 `axum::serve`
- `crates/koharu-core`는 **삭제함** (Phase 1에서 vendoring했으나 어댑터 방식에서는 불필요. workspace 등록도 제거)

**루트 `Cargo.toml`**
- `koharu-core` path 의존성 제거
- Phase 1에서 추가했던 의존성 대부분 제거 (async-stream, camino, chrono, dashmap, indexmap, natord, proptest, rmcp, tokio-stream, tower-http, utoipa, utoipa-axum)
- `axum = "0.8"` (feature 없음)만 유지 — koharu-rpc가 이제 axum 기본 기능만 사용
- ※ `koharu-rpc/Cargo.toml`의 의존성 목록과 루트 workspace 항목 일치 여부 재확인 요망

**`crates/koharu` (메인 바이너리 연동)**
- `Cargo.toml`: `koharu-rpc = { workspace = true }` 추가
- `src/main.rs`: `KOHARU_RPC_PORT` 환경변수가 유효한 포트면
  `koharu_app::extend_setup(move |handle| koharu_rpc::serve(handle, port))` 등록 (옵트인)

## 4. 현재 상태

### 빌드: ✅ 성공
- `target\debug\koharu.exe`가 2026-08-18 13:36 에 재빌드됨 (RPC 변경사항 포함).
  즉 `koharu-app` 확장 + `koharu-rpc` 재작성 + 메인 바이너리 연동까지 **컴파일 통과**.

### 실행: ❌ 차단 (RPC와 무관한 데스크톱 런타임 문제)
`KOHARU_RPC_PORT=8787`로 `koharu.exe` 실행 시:

```
Failed to run the desktop application : could not find the webview runtime
  (panic 위치: crates/koharu/src/main.rs:47 → app::run → .expect)
```

**원인 추적 결과 (2026-08-18 2차 세션에서 재검증, 결론 갱신):**

> ⚠️ **이전 기록의 "bootstrap.exe stale / dll-bootstrap 버전 불일치" 가설은 폐기.**
> 실제 조사 결과 CEF 파일은 모두 **동일 배포본**이다:
> - `libcef.dll`(275MB), `bootstrap.exe`, `bootstrapc.exe`, `.pak`, `locales/` 전부
>   2026-07-09 산출, 버전 `150.0.10+g8042e43+chromium-150.0.7871.101`로 동일
> - `target/debug`의 CEF와 `target/debug/build/cef-dll-sys-*/out/cef_windows_x86_64`가
>   파일 해시까지 일치 (배포: `cef_binary_150.0.10+g8042e43..._windows64_minimal.tar.bz2`).

- 에러 문자열은 fork 코드가 아니라 tauri 쪽:
  `tauri-runtime-cef/src/runtime.rs:1483`(8c204b8) — `cef::initialize(...) != 1`이면
  `Err(Error::WebviewRuntimeNotInstalled)` 반환 → "Could not find the webview runtime".
  즉 **CEF 네이티브 `cef::initialize`가 실패**하는 것. Rust 로직 문제가 아님.
- 재현 확인: `KOHARU_RPC_PORT=8787`로 `koharu.exe` 실행 시 동일 panic (main.rs:47).
- **실패 근본 원인으로 밝혀진 의미 구조 (CEF 150 "bootstrap / two-binary" 모델):**
  - 이 CEF 배포의 `bootstrap.exe`는 **클라이언트 DLL 방식 전용 진입점**이다.
    bootstrap.exe 단독 실행 시
    `[0818/...:FATAL:cef\libcef_dll\bootstrap\bootstrap_win.cc:387] Missing module name` FATAL
    (bootstrap_util_win.cc: exe 이름이 bootstrap/bootstrapc면 그대로 종료).
  - 정상 사용법: bootstrap.exe를 `<앱>.exe`로 복사하거나 `--module=<이름>` 스위치로
    실행하면 `<앱>.dll`을 로드해 거기서 `RunWinMain`(GUI) / `RunConsoleMain`(콘솔) export를 호출.
    즉 **브라우저 프로세스 = bootstrap.exe(+클라이언트 DLL)** 구조.
  - cef-rs(`tauri-apps/cef-rs` dev, `examples/cefsimple`)가 이를 확정:
    `[lib] crate-type = ["cdylib"]` + `src/win.rs`의
    `#[unsafe(no_mangle)] unsafe extern "C" fn RunWinMain(...)` export.
    그리고 `main.rs`는 **sandbox + windows에서
    `Err("Running in sandbox mode on Windows requires bootstrap.exe or bootstrapc.exe.")`**
    → **Windows sandbox 모드 = two-binary 필수. no-sandbox(비 sandbox 빌드)에서만 단일 exe가 유효**.
  - fork의 `koharu`는 **단일 exe 구조**(순수 bin 크레이트, `koharu.dll` 없음)이며,
    tauri-runtime-cef가 `cef::initialize`를 직접 호출. sandbox(feature 기본값)와 결합돼
    exe 직접 실행이 실패하는 것으로 진단됨.
- **tauri/runtime 쪽 재확인 (실사용 커밋 = 8c204b862, 2026-08-16):**
  - `crates/koharu/Cargo.toml` → `tauri`는 `feat/cef#8c204b862` (Cargo.lock 갱신됨, 로컬 checkout 존재:
    `C:\Users\chyng\.cargo\git\checkouts\tauri-69fbbe4d0942e697\8c204b8`)
  - tauri-runtime-cef: `default = ["sandbox"]`, runtime.rs `no_sandbox: !cfg!(feature="sandbox")`.
    `execute_process` → `cef::initialize` 직접 호출, bootstrap 처리 코드 없음,
    `browser_subprocess_path`도 미설정. ⇒ **이 단일 exe 경로는 CEF 150 sandbox 배포에서 성립하지 않음**.
  - tauri-cli/tauri-build: Windows CEF 관련 bootstrap 배포/`RunWinMain` 처리는 없음
    (tauri-build는 `windows-cef-app-manifest.xml`, Linux rpath만).

## 5. 다음 단계 (권장 순서)

1. **CEF 실행 경로 결정 (둘 중 하나, 아직 미검증)**
   - **경로 A — two-binary(bootstrap) 구조로 전환 (권장 후보, cef-rs 정식 방식):**
     - `crates/koharu`를 클라이언트 DLL로 빌드 (`crate-type=["cdylib"]`)하고
       Windows에서 `RunWinMain` export 제공 → bootstrap.exe를 `koharu.exe`로 복사.
     - 단, tauri가 `RunWinMain`을 대신 만들어주는 코드가 **없으므로**
       koharu 측(또는 tauri-runtime-cef 측)에 export가 필요함을 실험으로 확인 필요.
   - **경로 B — no-sandbox 단일 exe (cefsimple의 main() 경로가 유효하다는 점에 기반):**
     - tauri-runtime-cef의 `sandbox` feature를 끄고(`tauri`의 `cef` feature가
       `dep:tauri-runtime-cef`를 default-features 포함으로 가져오므로 fork 쪽에서
       feature 조정이 가능한지 확인) exe 직접 실행 실험.
   - 사전 실험 포인트: 이 libcef.dll 배포가 no-sandbox 단일 exe를 실제로 허용하는지.
     `cef-rs` dev 브랜치의 `cefsimple`이 두 경로를 모두 보여주는 기준 코드.
   - 참고: `bun dev`(= `cargo tauri dev --features default`)가 어떤 실행 구성을
     만드는지는 아직 확인 안 됨 — dev 플로우로 먼저 시도해보는 것도 옵션.
2. **동작 검증** (기동 후)
   - `curl http://127.0.0.1:8787/...` (실제 라우트 prefix는 `crates/koharu-rpc/src/api.rs` 확인)
   - projects/pages/operations/config/fonts 라우트별 스모크 테스트
   - 데스크톱 UI와 HTTP API가 동시에 같은 프로젝트 상태를 조작하는지 확인
3. **커밋 정리**
   - `rpc-pid.txt` 삭제 (내용: `9924` — 유효 정보 아님)
   - 조사 중 생성된 임시 파일 삭제: `target/debug/rpc-test-out.log`,
     `rpc-test-err.log`, `bootstrap-out.log`, `bootstrap-err.log`
   - Phase 1 커밋(`299582c0`)과 Phase 2 변경을 하나의 coherent 커밋으로 squash 권장
     (AGENTS.md 정책: 호환 레이어 없는 일관된 소유권 설계)
   - 커밋 후 `git push origin restore/koharu-rpc`
4. **후보 후속 작업 (별도 이슈)**
   - 제거한 기능 재도입 검토: MCP 서버, PSD export, 다운로드/이력/LLM/파이프라인 라우트,
     SSE 이벤트(`events.rs`), OpenAPI 문서 바이너리
   - 인증: 현재 127.0.0.1 바인드 전용. LAN 노출 전 토큰 인증 필수

## 6. 보안 주의사항

- 서버는 **루프백(127.0.0.1) 전용**이며 인증 없음. `KOHARU_RPC_PORT`는 신뢰 환경에서만 설정할 것.
- config 라우트가 provider 시크릿(키링)을 다루므로 원격 노출 절대 금지.

## 7. 참고

### koharu-rpc 최종 파일 구성

```
crates/koharu-rpc/
├── Cargo.toml
└── src/
    ├── lib.rs      # serve(AppHandle<Cef>, port), AppState 정의
    ├── api.rs      # router 조립
    ├── error.rs    # ApiError/ApiResult
    └── routes/
        ├── mod.rs
        ├── meta.rs        # 서버 메타데이터
        ├── config.rs      # 설정
        ├── projects.rs    # 프로젝트 CRUD (ProjectLibrary 연동)
        ├── pages.rs       # 페이지 조회/편집 (editing 명령 연동)
        ├── operations.rs  # 파이프라인 실행 (process/stop_job 연동)
        └── fonts.rs       # 폰트 목록/미리보기 (Desktop renderer 연동)
```

### 핵심 koharu-app API (라우트가 호출하는 것들)

| 기능 | 위치 |
|---|---|
| `CurrentProject`, `ProjectLibrary`, `Project`, `Page`, `Layer` | `koharu_app::commands::project` |
| `get/list/create/open/close/delete_project`, `import_pages`, `select_page`, `get_pages`, `get_page` | `koharu_app::commands::lifecycle` |
| `rename/move/delete_page`, `set_source_text/translation/typography/geometry/visibility`, `delete_layers`, `move_layer`, `undo`, `redo` | `koharu_app::commands::editing` |
| `process(scope, operation) -> JobId`, `stop_job(job)` | `koharu_app::commands::processing` |
| `Preferences::load`, `save_preferences`, `get_translation_models` | `koharu_app::commands::preferences` |
| `get_fonts`, `get_font_preview` | `koharu_app::commands::fonts` |
| `koharu_pipeline::{Scope, Operation, Stage}` — 모두 serde 직렬화 지원 | `koharu-pipeline` |

주의: 명령 함수는 `State<'_, T>` 파라미터를 받는 Tauri 시그니처 그대로이므로,
라우트에서는 `handle.state::<T>()`로 상태를 얻어 호출해야 함
(상태 타입은 `pub`로 공개됨).
