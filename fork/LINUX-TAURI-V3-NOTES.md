# Linux 데스크톱 — Tauri v3(GTK4 + CEF 152) 이후 회귀와 대응

최종 갱신: 2026-09-30

upstream의 Tauri v3 전환(`4a3c7b03 chore(deps): upgrade Tauri to v3 alpha`, 이후
`d05973d4` rev 고정)을 병합한 뒤 Linux 데스크톱에서 나온 문제와, 이 fork가 한 대응을
정리한다. 측정은 한 대의 머신(RTX 3060 + Intel UHD 630 하이브리드, NVIDIA 615.71,
KDE Plasma Wayland 세션의 XWayland)에서 했다. **다른 GPU·드라이버·세션에서 같은
결론이 나온다고 가정하지 말 것.** 아래 "확인 방법"은 그대로 다시 돌릴 수 있게 적었다.

관련 upstream 이슈: koharu-rs/koharu#1147(키보드 입력 불가), #1109(WebGPU 캔버스 불가).

---

## 1. 요약

| 증상 | 원인 | 대응 | 커밋 |
|---|---|---|---|
| 키보드 입력이 페이지에 전혀 안 들어감 | GDK가 X 입력 포커스를 자기 포커스 창에 둠. 그 창은 CEF 창의 조상이 아님 | 창이 포커스를 얻으면 CEF 브라우저 창에 `XSetInputFocus` | `8c4ef86a`, `0d4cbaf4` |
| "WebGPU 캔버스를 사용할 수 없습니다 … webgpu found no adapters" | `VulkanFromANGLE`에서 GPU 프로세스가 `vkDestroySurfaceKHR` 바인딩 실패 → Vulkan·EGL 초기화 실패 | `VulkanFromANGLE` 제거 | `6789fbe5` |
| 제목 표시줄 끌기·가장자리 크기 조절 불가 | winit-gtk4의 `drag_window`는 GDK가 받은 버튼 누름이 필요한데, 클릭은 전부 CEF 창으로 감 | koharu-rpc 경유로 EWMH `_NET_WM_MOVERESIZE` 직접 요청 | `0d4cbaf4` |
| 첫 페이지 로딩 때 창 전체가 5~12초 멈춤, 페이지 전환 수백 ms | WebGPU가 SwiftShader(CPU) 폴백 어댑터로 돌고 있었음 | GL 합성 + Dawn OpenGLES 백엔드, 캔버스가 호환 모드 어댑터 요청 | 이 문서와 같은 커밋 |

---

## 2. 창 구조 (원인 이해용)

`xwininfo -root -tree`로 본 Koharu 창:

```
GTK4 최상위 창 ("Koharu")
├── CefX11Host (tauri-runtime-cef가 만든 24비트 X11 창)
│   └── CEF 브라우저 창
└── 1x1 GDK 포커스 창 (-1,-1)
```

GTK4는 이 트리 안의 CEF 창에 입력 이벤트를 전달하지 않는다. X 서버는 포커스 창(또는
그 하위)에만 키 이벤트를, 포인터 아래 창에만 버튼 이벤트를 보낸다. 그래서 키 입력은
GDK 포커스 창으로 가서 버려지고, 클릭은 CEF 창으로만 가서 GDK는 모른다. 1번과 3번
문제가 모두 여기서 나온다. GTK4 이전(winit_x11)에는 CEF 창이 최상위 창의 직접 하위였고
최상위 창이 포커스를 받았기 때문에 문제가 없었다.

---

## 3. 문제별 상세

### 3.1 키보드 입력

- 재현: 앱 실행 → 입력칸 클릭 → 입력해도 아무것도 안 들어감. 한글(fcitx5)도 동일.
- 확인: X 포커스가 `0x…0009`(GDK 포커스 창)이면 입력 불가, CEF 창으로 옮기면 입력됨.
  XTest로 두 경우를 비교해 확인했다.
- `BrowserHost::set_focus(1)`만으로는 부족했다. 이벤트는 제때 오지만 X 포커스를 옮기지
  않는다(로그로 확인).
- 대응: `crates/koharu-app/src/linux_window.rs`의 `focus_cef_window`. 창이 보이는
  상태(`IsViewable`)일 때만 `XSetInputFocus`한다(보이지 않는 창에 요청하면 BadMatch).
- 부수 효과: 앱이 뜬 직후 입력칸 자동 포커스 표시가 사라져, 입력칸을 한 번 클릭해야 한다.

### 3.2 WebGPU 어댑터 없음

- CEF 로그(`~/.cache/Koharu/cef/chrome_debug.log`):
  `Failed to bind vulkan entrypoint: vkDestroySurfaceKHR` →
  `Failed to create and initialize Vulkan implementation` →
  `No suitable EGL configs found for initialization`.
- 0.80.0(CEF 151.1.0)에서는 이 경고가 없었다. 151.8.1, 152.3.0에서는 계속 난다.
- `VulkanFromANGLE`을 켜면 Chromium이 ANGLE이 만든 VkInstance를 빌려 쓰는데, ANGLE이
  surface 확장 없이 인스턴스를 만들어 바인딩이 실패하는 것으로 보인다.
  `--disable-vulkan-surface`, `DefaultANGLEVulkan` 등으로도 해결되지 않았다.

### 3.3 창 이동·크기 조절

- `crates/koharu-app/src/linux_window.rs`의 `begin_move_resize`가 포인터 위치를
  `XQueryPointer`로 읽고 `_NET_WM_MOVERESIZE`를 루트 창에 보낸다. 요청이 도착했을 때
  버튼이 이미 떼어져 있으면 아무것도 하지 않는다.
- 경로: `POST /api/v1/window/move-resize` (`crates/koharu-rpc/src/routes/window.rs`).
  Linux가 아니면 501.
- 프론트엔드(`packages/koharu/components/app/WindowChrome.tsx`)의 `startWindowDrag`는
  **캡처 단계**(`onMouseDownCapture`)에 걸어야 한다. Next App Router에서 React 루트가
  `document`이고, Tauri 드래그 스크립트(`drag.js`)가 먼저 등록된 `document` 리스너에서
  `stopImmediatePropagation()`을 부르기 때문에 버블 단계 핸들러는 호출되지 않는다.
- 더블클릭 최대화는 Tauri 경로 그대로 쓴다(누름 기록이 필요 없음).

### 3.4 첫 로딩 멈춤 (SwiftShader 폴백)

측정 결과:

| 설정 | 페이지가 받은 어댑터 | 첫 로딩 중 GPU 메인 스레드 바쁜 시간 |
|---|---|---|
| `Vulkan` + `use-angle=vulkan` (3.2 대응 직후) | SwiftShader, `isFallbackAdapter: true` | 8.5초 |
| GL 합성 + `--use-webgpu-adapter=opengles` + 호환 모드 요청 | NVIDIA(ANGLE 위 OpenGLES), 폴백 아님 | 0.5초 |

원인 추적:

- 멈춤 동안 GPU 프로세스 메인 스레드 스택이 전부 `libvk_swiftshader.so` 안이었다.
- `chrome://gpu`의 Dawn Info에는 `SwiftShader (Vulkan)`과
  `ANGLE on NVIDIA (OpenGLES, Compatibility Mode)` 두 어댑터만 있었다. Chromium 자신은
  NVIDIA로 하드웨어 가속 중이었다(`GaneshVulkan`, Vulkan: Enabled).
- Chromium `gpu/command_buffer/service/webgpu_decoder_impl.cc`: Linux 기본값에서는
  Vulkan 백엔드만 찾고, `SupportsExternalImages()`이거나 SwiftShader인 어댑터만 받으며,
  없으면 폴백 어댑터로 넘어간다. `--use-webgpu-adapter=opengles`이면 OpenGLES 백엔드를 찾는다.
- OpenGLES 어댑터는 호환 모드라 기본(core) 요청에는 안 잡힌다. wgpu 29의 WebGPU 백엔드는
  `featureLevel`을 넘기지 않으므로, `packages/bridge/src/canvas.ts`의
  `preferHardwareAdapter`가 `GPU.prototype.requestAdapter`를 감싸서 기본 요청이 폴백을
  돌려줄 때만 `featureLevel: "compatibility"`로 다시 요청한다. 하드웨어 어댑터가 바로
  잡히는 플랫폼(Windows, macOS)에는 영향이 없다.
- 합성이 Vulkan(`GaneshVulkan`)이면 호환 모드 장치가 캔버스 텍스처를 만들지 못한다
  (`Device does not support ANGLE GL texture sharing`). 그래서 `Vulkan` 기능과
  `use-angle=vulkan`을 빼서 합성을 GL(`GaneshGL`)로 돌린다.

효과가 없었던 것(모두 SwiftShader): `--ignore-gpu-blocklist`, `--use-vulkan=native`,
`--use-webgpu-power-preference=default-high-performance`, `--disable-gpu-sandbox`,
`VK_DRIVER_FILES=nvidia_icd.json`, `__NV_PRIME_RENDER_OFFLOAD=1` +
`__GLX_VENDOR_LIBRARY_NAME=nvidia`, `EGL_PLATFORM=x11`, `--use-angle=gl|default|gl-egl`.
`--disable-gpu-blocklist`는 CEF 152에 없는 스위치다. 인자로 `--disable-features=…`를
덧붙이면 tauri-runtime-cef 자신의 `--disable-features`와 겹쳐 앱이 segfault한다.

남은 위험: 호환 모드에는 WGSL·텍스처 제약이 있다. 첫 페이지 표시는 확인했지만 모든
편집 도구의 셰이더를 검증하지는 않았다.

---

## 4. 확인 방법

- **X 포커스**: `xwininfo -root -tree`로 창 ID를 보고, Xlib `XGetInputFocus`로 포커스 창을
  읽는다. CEF 브라우저 창이어야 한다.
- **WebGPU 어댑터 / GPU 부하**: `GpuWatchdog` 스레드를 가진 `koharu` 프로세스가 GPU
  프로세스다. 프로젝트를 연 뒤 그 프로세스 메인 스레드의 CPU 시간(`/proc/<pid>/task/<pid>/stat`)을
  본다. 수 초 이상 바쁘면 SwiftShader를 의심한다. `libvk_swiftshader.so`가 매핑됐다는
  사실만으로는 판단할 수 없다(기본 요청이 폴백을 한 번 거치기 때문).
- **chrome://gpu**: release 빌드는 원격 디버깅을 거부한다. tauri-runtime-cef는 원격
  디버깅이 꺼져 있을 때 `~/.cache/Koharu/cef/Local State`의
  `devtools.remote_debugging.allowed`를 false로 고정하고, 켤 때 되돌리지 않는다. 확인이
  필요하면 앱을 끈 상태에서 이 값을 잠시 true로 바꾸고 `RemoteDebugging::Port`로 띄운 뒤,
  끝나면 되돌린다.
- CEF 로그 상세도 스위치(`--v`, `--vmodule`)는 `log_severity=warning` 설정 때문에 무시된다.
  Dawn 경고는 CEF 로그가 아니라 앱의 표준 에러로 나온다.

---

## 5. 아직 남은 문제

- **메인 스레드 CPU 100%**: 아무 조작이 없어도 Tauri 이벤트 루프(CEF UI 스레드)가 한 코어를
  계속 쓴다. 3초에 `ppoll`(타임아웃 0) 약 3만 번, X 연결 읽기 약 16만 번. CEF 메시지 펌프
  (초당 약 60회), 포커스 핑퐁, 레이아웃 루프는 원인이 아님을 확인했다. winit-gtk4 루프의
  어느 GSource가 대기를 막는지는 아직 모른다.
- upstream 이슈 #1147이 닫히면 3.1·3.3의 우회 코드와 이 문서를 다시 검토한다.

---

## 6. 이 머신에서의 빌드 메모

- 저장소가 ntfs3 드라이브에 있으면 wasm-pack 0.15.0이 `wasm-opt`를 끝없이 반복한다(디렉터리
  순회 중 rename한 파일이 다시 순회에 잡힘). canvas wasm은 `--out-dir`를 ntfs 밖으로 지정해
  빌드한 뒤 `packages/bridge/src/wasm/`에 복사하고, deb는
  `cargo tauri build --bundles deb --config '{"build":{"beforeBuildCommand":"bun run --filter @koharu/app build"}}'`로 만든다.
- Tauri v3 이후 Linux 빌드에는 `libgtk-4-dev`, `koharu-torch-sys`의 bindgen에는
  `libclang-common-21-dev`가 필요하다.
