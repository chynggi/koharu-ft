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
최종 해법은 **경로 B(no-sandbox 단일 exe)** 였다. 이 구성은 진단용 실험이 아니라
**현재 유지되어야 하는 구성**이다 — vast.ai 같은 마켓플레이스 템플릿은 docker
capability(`--cap-add=SYS_ADMIN`)를 표현할 수 없고 호스트가 비특권 user namespace를
막아둘 수도 있으므로, 샌드박스를 끄는 쪽이 배포 가능한 유일한 경로다. (3차 세션
기록의 "가설 확정 시 vendor 패치를 되돌릴 것"은 따라서 **무효**다.)

**정정 (2026-08-20, 5차 세션):** 이전 판은 "`default` feature를 `[]`로 패치해
`sandbox`를 끈다"까지만 적었고, 그것으로 충분한 것처럼 읽혔다. **충분하지 않았다.**
vast.ai 실배포에서 컨테이너가 다음으로 죽었다:

```
Failed to move to new namespace: PID namespaces supported, Network namespace
supported, but failed: errno = Operation not permitted
FATAL:content/browser/zygote_host/zygote_host_impl_linux.cc:207]
Check failed: . : Operation not permitted (1)
```

샌드박스를 끄는 데는 **두 조각**이 필요하다:

1. `vendor/tauri-runtime-cef/Cargo.toml`의 `default = []` → `CefSettings::no_sandbox`.
2. 같은 크레이트가 `command_line_args`에 `--no-sandbox`를 넣는 것.

(1)만으로 부족한 이유는 순서다. `runtime.rs`에서 `cef::execute_process`가 **먼저**
호출되고 `Settings`는 그 뒤에 만들어져 `cef::initialize`에만 전달된다. 그런데
서브프로세스(zygote 포함)는 같은 바이너리로 재진입해 `execute_process`에서 처리되므로
`no_sandbox`를 볼 기회가 없다. 모든 프로세스에서 실행되는 훅은
`CefApp::OnBeforeCommandLineProcessing` 하나뿐이고, `command_line_args`가 바로 그
경로다. Linux에서 `sandbox` cargo feature의 다른 사용처는 전부
`#[cfg(target_os = "macos")]`이므로, 그 feature는 사실상 `no_sandbox` 플래그 하나만
바꿨을 뿐이었다.

둘은 같은 `#[cfg(not(feature = "sandbox"))]`로 묶어 두었다 — 어긋나면 조용히 다시
이 증상으로 돌아온다.

**아직 실기 검증 안 됨:** 이 수정으로 컴파일은 통과하지만, 고쳐진 이미지를 vast.ai에서
실제로 띄워본 적은 없다. 확인 방법은 §3 아래 "배포 확인" 참조.

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

### import / export 브라우저 갭 (5차 세션)

Phase 3b가 "브라우저에 네이티브 다이얼로그가 없다"며 플레이스홀더로 남긴 항목.
해결됐다. 갈래를 나눈 기준은 취향이 아니라 **파일이 어느 머신에 있어야 하는가**다.

| 라우트 | 대상 | 방식 |
|---|---|---|
| `/pages/import`, `/pages/export` | 스크립트·API | 서버 경로 (기존) |
| `/pages/import/dialog`, `/pages/export/dialog` | 데스크톱 | 네이티브 다이얼로그, **루프백 전용** |
| `/pages/import/upload`, `/pages/export/download` | 원격 브라우저 | multipart / zip |

클라이언트는 `packages/koharu/lib/transfer.ts`의 `runImport`/`runExport`에서
`__TAURI_INTERNALS__`로 갈라진다. 서버측 루프백 판정만으로는 부족하다 — 컨테이너
안의 CEF 창도 루프백이지만 headless Xvfb 위에 있어서, 다이얼로그를 열면 아무도 볼 수
없는 화면에 떠서 요청이 반환되지 않는다. 라우트의 루프백 가드는 이중 방어로 남겼다.

**두 번 방향을 고친 지점 (같은 함정을 다시 밟지 말 것):**

1. 업로드 바이트용 디코드 경로를 따로 만들 뻔했다. `zip::extract` / `rar::extract` /
   `pdf::render`가 **셋 다 `&Path`를 받는다**(`import/mod.rs:97-99`). 바이트
   디스패치는 그 네 갈래 분기의 두 번째 사본이 되어 갈라진다. 그래서 업로드는 임시
   파일로 흘려보내고 기존 `import()`를 그대로 부른다. export도 같은 이유로 임시
   디렉터리에 렌더한 뒤 zip으로 묶는다 — `export_pages_to`의 렌더·이름·순서 규칙을
   복제하지 않기 위해서다.
2. `importPages(files: string[])`로 서명을 바꾼 뒤에야 **CEF 창의 브라우저
   컨텍스트도 실제 경로를 얻을 수 없다**는 것을 알아챘다. 그래서 데스크톱 경로는
   클라이언트가 경로를 넘기는 방식이 아니라 서버측 다이얼로그 라우트여야 한다.
   기존 Tauri 커맨드(`lifecycle::import_pages`, `output::export_pages`)가 다이얼로그
   부터 커밋까지 이미 전부 하므로 그대로 호출한다.

**주의해서 다룬 두 지점:**

- 업로드 파일명은 caller가 정하는 값이 곧 경로가 된다. `Path::file_name()`으로 마지막
  성분만 취해 staging 디렉터리 밖으로 나가는 경로를 잘라낸다.
- 필드는 `field.bytes()`가 아니라 `chunk()`로 흘려 쓴다. CBZ 하나가 메모리에 담기
  어려운 크기일 수 있고, staging의 존재 이유가 그것을 담지 않는 것이다. 그래서
  업로드 라우트만 `DefaultBodyLimit::disable()`이다 — 메모리는 여전히 유계고, 쓰는
  자원은 임시 디렉터리 공간이며, 비-루프백 바인드에서는 토큰이 있어야 도달한다.

**아직 검증되지 않음:** 실제 원격 브라우저로 업로드/다운로드를 돌려본 적이 없다.
단위 테스트는 분기만 덮는다(`tests/lib/transfer.test.ts`) — 브라우저 파일 피커는
jsdom에서 열리지 않으므로 업로드 경로의 실제 왕복은 실기 확인이 필요하다.

---

## 2. 남은 작업

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

### C. specta 전체 제거 (낮음, 단 범위가 큼)

**정정 (2026-08-20):** 이전 판은 "`#[specta::specta]` 51개 정리"라고 적었다. 범위를
4배 축소한 서술이었다. 확인해 보니 **specta는 워크스페이스 전체에서 죽었다.**

```
grep -rn "TypeCollection\|specta_typescript\|\.export(\|Typescript::" crates/ --include=*.rs
→ 결과 없음
```

`Type` impl을 읽는 쪽이 어디에도 없다. 모든 사용처가 `use specta::Type` +
`derive(..., Type)`, 즉 **읽는 쪽 없는 쓰기**다. 유일한 소비자였던 `bindings()` /
`bin/generate.rs`가 사라졌기 때문이다.

실제 범위: `#[specta::specta]` 51개 + `derive(Type)` 약 140개 + 20여 파일의 import +
8개 크레이트의 `specta` 의존성 + 워크스페이스 항목. `#[specta(type = f64)]`(specta가
BigInt를 금지해 넣은 것)도 함께 사라진다.

제거를 권하는 이유는 정리 그 자체가 아니다. **죽은 타입 내보내기 체계가 살아 있는 척
하는 것이 더 나쁘기 때문**이다 — 지금 `derive(Type)`은 `protocol.ts`가 생성되거나
검증되는 듯한 인상을 주지만, 그 파일은 손으로 유지되며 Rust 쪽과의 일치를 아무도
강제하지 않는다. 게다가 `specta`는 `=2.0.0-rc.25`로 핀되어 있어 업그레이드를 묶는다.

되돌릴 근거: 데스크톱 IPC/타입 생성 경로를 언젠가 복원한다면 전부 다시 필요하다.

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
bun run --filter '@koharu/app' test           # 86/86 across 13 files
```

위 수치는 **`main` 기준**이다. `fork/vram-accounting` 브랜치는 자체 테스트를 더해
rust 쪽이 52→65, 44→61이 되므로 두 브랜치의 숫자를 섞지 말 것. (그 브랜치는 아직
import/export 작업 이전의 main 위에 있다 — 리베이스하면 프런트 수치도 바뀐다.)

트리의 유일한 기존 경고는 `koharu-torch-sys`의 빌드 스크립트 `linker_messages`다.
우리 것이 아니다.

### 배포 확인 (vast.ai)

컴파일과 단위 테스트로는 컨테이너 기동을 검증할 수 없다. 이미지를 새로 빌드한 뒤
로그에서 확인할 것:

- `zygote_host_impl_linux.cc ... Check failed` 가 **없어야** 한다. 남아 있다면
  `--no-sandbox`가 실제로 붙지 않은 것이다. 실행 중인 프로세스의 커맨드라인으로
  직접 확인할 수 있다: `tr '\0' ' ' < /proc/<pid>/cmdline`.
- `XDG_RUNTIME_DIR ... is owned by uid` dbus 경고가 없어야 한다.
- `Owner of /tmp/.X11-unix should be set to root` 경고가 없어야 한다.
- `curl -H "Authorization: Bearer $KOHARU_API_TOKEN" http://<host>:<port>/api/v1/meta`
  가 이름과 버전을 돌려주어야 한다.

**이미지가 최신 소스에서 빌드된 것인지 먼저 확인할 것.** 위 증상은 `--no-sandbox`
누락으로도, 그냥 오래된 이미지로도 똑같이 나타난다. GHCR 워크플로는 수동 트리거다.

**한 번도 검증되지 않은 것:** 실제 원격 브라우저로 vast.ai 배포본에 붙어 프로젝트를
처리하는 end-to-end 실행. 4차 세션의 Edge 검증은 로컬 `bun run dev` 기준이었다.
import/export 업로드·다운로드도 실기로는 아직 한 번도 돌지 않았다.

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
