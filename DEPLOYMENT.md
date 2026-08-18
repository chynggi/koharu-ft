# DEPLOYMENT — koharu 원격/Docker 배포 가이드 (restore/koharu-rpc 브랜치)

이 문서는 `restore/koharu-rpc` 브랜치의 Phase 3 작업(원격 브라우저 UI 접근 + Docker 배포)을
실제로 사용하기 위한 가이드다. 배경/설계 결정 과정은 `HANDOFF.md` 참고.

## 1. 아키텍처 요약

- `koharu-rpc`가 HTTP API(`/api/v1/*`)와 정적 프론트엔드(`/`)를 **같은 포트**로 서빙한다.
- 데스크톱 CEF 창도 이제 이 HTTP 서버를 로드한다 (Tauri IPC 아님) — 즉 데스크톱 앱과 원격
  브라우저가 완전히 동일한 UI/API를 공유한다.
- 서버는 기본적으로 `127.0.0.1`(루프백)에 바인드되고 인증이 없다. 원격 접근을 위해
  `KOHARU_RPC_HOST=0.0.0.0`으로 바꾸면 **`KOHARU_API_TOKEN` 설정이 강제**된다 (없으면 앱이
  시작을 거부함).

## 2. 로컬(같은 머신) 사용 — 별도 설정 불필요

```
./koharu.exe   # 또는 리눅스 바이너리
```

기본 포트 `47823`으로 API+UI가 뜨고, CEF 창이 자동으로 이걸 로드한다. 인증 없음(루프백
전용이므로 안전).

## 3. 원격(다른 머신/네트워크) 접근 — 토큰 필수

### 3.1 프론트엔드 빌드 시 토큰 주입

`NEXT_PUBLIC_API_TOKEN`은 **빌드 타임에 정적 파일에 박히는 값**이다 (런타임에 바꿀 수 없음).
실행 시 사용할 `KOHARU_API_TOKEN`과 **반드시 동일한 값**으로 미리 설정해야 한다.

```bash
export NEXT_PUBLIC_API_TOKEN="<임의의 긴 랜덤 문자열>"
bun run ui:build   # 또는 cargo tauri build 안에서 beforeBuildCommand로 자동 실행됨
```

### 3.2 서버 실행

```bash
KOHARU_RPC_HOST=0.0.0.0 \
KOHARU_API_TOKEN="<위와 동일한 토큰>" \
./koharu
```

토큰 없이 `KOHARU_RPC_HOST`를 루프백이 아닌 값으로 설정하면 앱이 panic으로 즉시 종료된다
(`crates/koharu/src/main.rs`).

### 3.3 인증 방식

`/api/v1/*` 모든 요청에 둘 중 하나가 필요하다:
- `Authorization: Bearer <token>` 헤더
- `?token=<token>` 쿼리 파라미터 (브라우저 `EventSource`가 커스텀 헤더를 못 보내서 SSE
  `/api/v1/events` 연결에 사용됨)

정적 UI(`/`)는 토큰 없이 로드된다 — 민감 정보가 없고, 실제 API 호출 시점에 토큰이 필요하다.

## 4. Docker 배포 (vast.ai 등)

`Dockerfile`이 리포지토리 루트에 있다. `vastai/linux-desktop:cuda-13.2-ubuntu24.04-2026-06-16`
베이스로 실제 로컬에서 빌드·실행까지 검증됨.

### 4.1 빌드

```bash
docker build \
  --build-arg KOHARU_API_TOKEN=<임의의 긴 랜덤 문자열> \
  -t koharu:latest .
```

`KOHARU_API_TOKEN` build-arg가 프론트엔드에 `NEXT_PUBLIC_API_TOKEN`으로 그대로 박힌다.
**다음 실행 단계의 `KOHARU_API_TOKEN`과 반드시 같은 값을 써야 한다.**

### 4.2 실행

```bash
docker run -d \
  --cap-add=SYS_ADMIN \
  -p 47823:47823 \
  -e KOHARU_RPC_HOST=0.0.0.0 \
  -e KOHARU_API_TOKEN=<빌드 때와 동일한 토큰> \
  koharu:latest
```

**`--cap-add=SYS_ADMIN`은 필수다.** 이게 없으면 CEF의 sandbox가 컨테이너 안에서 네임스페이스를
만들지 못해 앱이 시작하지도 못하고 죽는다 (`Failed to move to new namespace: ... Operation not
permitted`). 이 capability를 컨테이너에 부여하고 싶지 않다면, 대신
`vendor/tauri-runtime-cef/Cargo.toml`의 `default = ["sandbox"]`를 `default = []`로 바꿔
CEF sandbox 자체를 끄고 재빌드하는 방법도 있다 (검증 안 됨 — 3차 세션에서 이 패치를 했다가
되돌린 이력 있음, `HANDOFF.md` 참고).

### 4.3 vast.ai에서 열 포트

**포트 하나만 열면 된다: `47823`** (API+UI 동시 서빙).

vast.ai 인스턴스 설정에서 이 포트를 외부로 매핑하면, 발급받은 주소로 브라우저에서 바로
`http://<vast.ai 주소>:<매핑된 포트>/`에 접속해 koharu UI를 쓸 수 있다.

### 4.4 검증

```bash
# 인증 없이 UI는 뜨는지
curl -s -o /dev/null -w "%{http_code}\n" http://<host>:47823/

# 토큰으로 API 확인
curl -s -H "Authorization: Bearer <token>" http://<host>:47823/api/v1/meta

# 토큰 없이는 거부되는지
curl -s -o /dev/null -w "%{http_code}\n" http://<host>:47823/api/v1/meta   # 401이어야 정상
```

### 4.5 주의: vast.ai 자체 SSH/포탈 기능 상실

이 `Dockerfile`은 베이스 이미지의 자체 `ENTRYPOINT`(vast.ai instance-portal, SSH 프로비저닝
등)를 완전히 대체한다 — 그 entrypoint가 `exec "$@"`를 하지 않고 자체 인자 파싱만 하기 때문에
우리 앱을 그 안에 끼워 넣을 방법이 없었다. 즉 **이 이미지를 그대로 쓰면 vast.ai 대시보드의
SSH 접속/인스턴스 포탈 기능을 못 쓴다.** 그게 필요하면 entrypoint를 병행 실행하도록 별도
작업이 필요하다 (이번 세션 범위 밖).

## 5. 알려진 제약

- **파일 import/export**: 브라우저에는 네이티브 파일 다이얼로그가 없어서 `importPages`/
  `exportPages`가 아직 플레이스홀더 상태다 (빈 경로 전송 → 서버 검증 실패). 브라우저 파일
  업로드/폴더 선택 UI가 별도로 필요함 — 후속 작업.
- **CEF 창은 여전히 필요**: 이 배포에서도 앱은 실제로 CEF 브라우저(Xvfb 위에서 헤드리스)를
  띄운다. 렌더링 결과를 아무도 보지 않지만, CEF 초기화 자체가 앱 구동에 필요하다.
- **updater 서명**: 이 Docker 빌드는 `createUpdaterArtifacts`를 꺼서 만든다 — 즉 이 배포
  경로로 만든 바이너리는 자동 업데이트 기능이 빠져 있다 (원래 데스크톱 배포용 서명 키가
  없기 때문. 의도된 것).

## 6. 트러블슈팅 (Docker 빌드 중 실제로 겪은 문제들)

빌드가 실패하면 우선 아래 항목들을 의심할 것 (전부 이 브랜치에서 실제로 마주친 것):

| 증상 | 원인 | 해결 |
|---|---|---|
| `npm: command not found`, wasm-pack이 `/crates/koharu-canvas` 같은 이상한 경로 에러 | 이미지에 npm 없음 (`npm prefix`가 빈 문자열로 치환됨) | Node.js(npm 포함) 설치 확인 |
| `You are using Node.js 18.x.x. For Next.js, Node.js version ">=20.9.0" is required` | apt 기본 nodejs가 구버전 | NodeSource 등으로 Node 20+ 설치 |
| `curl \| bash` 방식 설치가 조용히 실패 | 네트워크 일시 오류를 `-fsSL`이 삼킴 | 다운로드와 실행을 분리하고, 설치 후 `--version`으로 검증 |
| `A public key has been found, but no private key. TAURI_SIGNING_PRIVATE_KEY` | `tauri.conf.json`의 `createUpdaterArtifacts: true` | 빌드 시 `--config '{"bundle":{"createUpdaterArtifacts":false}}'` |
| 컨테이너가 뜨자마자 종료, 로그에 별 내용 없음, CMD가 실행 안 된 것처럼 보임 | 베이스 이미지 자체 ENTRYPOINT가 우리 CMD를 무시 | `ENTRYPOINT`를 명시적으로 재정의 |
| `Failed to move to new namespace: ... Operation not permitted` | CEF sandbox가 컨테이너 기본 권한으로 네임스페이스 생성 불가 | `docker run --cap-add=SYS_ADMIN` |
| `Failed to setup app: ... the Documents directory is unavailable` | `dirs::document_dir()`가 XDG user-dirs 설정 없이 실패 | `xdg-user-dirs` 설치 + `xdg-user-dirs-update` 실행 |
