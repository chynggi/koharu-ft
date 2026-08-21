# Koharu 원격/Docker 배포 가이드

Koharu는 `koharu-rpc`의 HTTP API(`/api/v1/*`)와 정적 UI(`/`)를 같은 포트에서 제공합니다. 기본 desktop 실행은 `127.0.0.1:47823`을 사용하며, 원격 접근을 위해 `0.0.0.0`에 bind하면 `KOHARU_API_TOKEN`이 필수입니다.

## Docker image 빌드

```bash
docker build -t koharu:latest .
```

API token과 provider key를 `ARG` 또는 Dockerfile `ENV`로 넣지 마세요. Build layer와 image metadata에 secret이 남을 수 있습니다. 모든 secret은 container를 시작할 때 주입합니다.

## Runtime 환경변수

`.env.example`을 복사한 뒤 필요한 값만 채웁니다.

```bash
cp .env.example .env
openssl rand -hex 32
# 출력값을 .env의 KOHARU_API_TOKEN에 입력
```

Linux image에서는 provider credential을 환경변수로만 읽습니다. Settings는 설정 여부와 변수명을 표시하지만 값을 변경하거나 삭제하지 않습니다. 값을 바꾸면 container를 재시작하거나 Vast.ai instance를 재배포해야 합니다.

| Provider | Environment variable |
|---|---|
| Atlas Cloud | `ATLASCLOUD_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Claude | `ANTHROPIC_API_KEY` |
| Grok | `XAI_API_KEY` |
| MiniMax | `MINIMAX_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| OpenAI-compatible | `OPENAI_COMPATIBLE_API_KEY` (optional) |
| OpenRouter | `OPENROUTER_API_KEY` |
| LM Studio | `LM_STUDIO_API_TOKEN` (optional) |
| DeepL | `DEEPL_API_KEY` |
| Google Cloud Translation | `GOOGLE_CLOUD_API_KEY` |
| Caiyun | `CAIYUN_API_KEY` |

Linux에서는 writable credential store가 없으므로 Codex OAuth login/logout을 지원하지 않습니다. Translation provider와 local model serving에는 영향이 없습니다.

## 로컬 Docker 실행

```bash
docker run -d \
  --name koharu \
  --env-file .env \
  -p 47823:47823 \
  koharu:latest
```

CEF sandbox는 image에서 비활성화되어 있으므로 `--cap-add=SYS_ADMIN`, `--privileged`, `--security-opt seccomp=unconfined`이 필요하지 않습니다.

브라우저에서 다음 URL을 엽니다.

```text
http://127.0.0.1:47823/?token=<KOHARU_API_TOKEN>
```

Token이 설정된 상태에서 bare `/`는 401이 정상입니다. Query token은 초기 HTML에 runtime으로 주입되며 image나 frontend bundle에 포함되지 않습니다.

## Vast.ai custom image

1. Docker Hub/GHCR에 push한 Koharu image tag를 template의 image로 지정합니다.
2. Launch mode는 image의 `ENTRYPOINT`를 보존하는 **Entrypoint/args** 방식을 사용합니다.
3. Template environment에 `KOHARU_API_TOKEN`과 실제로 사용할 provider 변수만 추가합니다.
4. Internal TCP port `47823`을 엽니다.
5. Instance의 **IP Port Info**에서 `47823/tcp`에 배정된 외부 host/port를 확인합니다.
6. 다음 주소로 접속합니다.

```text
http://<PUBLIC_IP>:<MAPPED_PORT>/?token=<KOHARU_API_TOKEN>
```

Vast.ai template의 Docker Options가 임의의 security option을 허용하지 않아도 이 image는 기본 container 제한으로 동작합니다. 현재 Dockerfile은 Vast.ai base image의 entrypoint를 대체하므로 base image가 제공하던 SSH/Jupyter/instance portal은 자동으로 시작되지 않습니다.

## 검증

```bash
# 인증 없이 API와 root가 거부되어야 함
curl -sS -o /dev/null -w '%{http_code}\n' http://<host>:<port>/api/v1/meta
curl -sS -o /dev/null -w '%{http_code}\n' http://<host>:<port>/

# 인증된 API와 UI가 성공해야 함
curl -fsS -H "Authorization: Bearer <token>" http://<host>:<port>/api/v1/meta
curl -fsS "http://<host>:<port>/?token=<token>" > /dev/null
curl -fsS -H "Authorization: Bearer <token>" http://<host>:<port>/api/v1/config
```

Expected status는 unauthenticated 요청 401, authenticated 요청 200입니다. `/api/v1/config`에는 credential 값이 아니라 `configured`, `editable`, `environment_variable` 상태만 반환됩니다.

## 알려진 제약

- App은 headless container에서도 CEF를 Xvfb 위에서 초기화합니다.
- `KOHARU_RPC_PORT`를 변경하면 desktop CEF의 initial navigation과 맞도록 `crates/koharu/tauri.conf.json`의 `devUrl`도 build 전에 변경해야 합니다.
- Provider model discovery는 외부 API에 연결하므로 key가 유효하지 않거나 provider가 unreachable이면 해당 model 목록이 비어 있을 수 있습니다.
- `.env`는 Git에서 ignore됩니다. `.env.example`에 실제 token이나 provider key를 넣지 마세요.
