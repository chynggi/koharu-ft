# 메모 — `koharu-metrics` 병합 후 남은 문제 (upstream `e4c3d069`)

최종 갱신: 2026-08-22 (upstream 병합 커밋 `09875684` 직후 작성)

upstream이 `5f87445e refactor: refine metrics` + `299e8835 fix(metrics): exclude
telemetry from logs`로 계측 계층을 갈아엎었다. 병합은 컴파일이 통과하지만
(`cargo check --workspace` exit 0), **포크가 자체로 추가했던 계측 일부가 조용히
죽거나 이중으로 발행되는 상태**로 남아 있다. 아래는 그 목록과 근거다.

**아래 "확인 방법"은 재검증용이다. 다음 세션은 이 문서를 믿기 전에 명령을 다시 돌릴 것.**
런타임 전송을 실제로 관찰한 적은 없다 — 전부 코드 독해에 근거한 주장이다.

---

## 1. upstream이 바꾼 것

세 가지가 동시에 바뀌었다.

**(1) 필드 이름 화이트리스트.** `crates/koharu-metrics/src/fields.rs`에 `EventField`
enum이 새로 생겼고, `strum::EnumString`으로 파싱한다. `lib.rs`의 `FieldVisitor::insert`가
이렇게 걸러낸다:

```rust
if field.name() == "metric" {
    self.metric = value.as_str().map(str::to_owned);
} else if let Ok(field) = field.name().parse() {
    self.values.insert(field, value);
}
```

`else if let Ok(...)` — **파싱에 실패한 필드는 경고 없이 버려진다.** enum에 없는 이름을
쓰면 컴파일도 통과하고 로그도 남지 않고 그냥 사라진다.

**(2) 스팬 기반 계측.** `#[tracing::instrument(target = "koharu_metrics", name = "...")]`을
쓰면 `on_close`에서 `duration_ms`를 붙여 스팬 이름 그대로 발행한다. 즉 스팬 이름이
이벤트 이름이 된다 — `metric = "..."` 이벤트와 같은 네임스페이스를 공유한다.

**(3) 로그에서 텔레메트리 분리.** `crates/koharu/src/main.rs`의 `filter`가
`EnvFilter`에서 `filter_fn(|m| m.target() != "koharu_metrics")`로 바뀌었다. Sentry
레이어와 `TimingLayer`가 이 필터를 받으므로, 계측 이벤트는 이제 **일반 로그에도 Sentry에도
찍히지 않는다.** 병합 과정에서 콘솔로 계측을 눈으로 확인하던 방법이 막혔다는 뜻이다.

확인 방법:

```sh
sed -n '/pub enum EventField/,/^}/p' crates/koharu-metrics/src/fields.rs
sed -n '/impl FieldVisitor/,/^}/p' crates/koharu-metrics/src/lib.rs
grep -n 'filter_fn' crates/koharu/src/main.rs
```

---

## 2. 조용히 버려지는 포크 전용 필드

화이트리스트에 없어서 전송되지 않는 필드는 두 개다.

| 필드 | 위치 | 상태 |
|---|---|---|
| `import_source` | `crates/koharu-app/src/commands/lifecycle.rs` — `metric = "import"` 이벤트 | 버려짐 |
| `export_formats` | `crates/koharu-app/src/commands/output.rs` — `export` 스팬 `fields(...)`와 `metric = "export"` 이벤트 양쪽 | 버려짐 |

같은 정보가 upstream 스키마에는 다른 이름으로 이미 있다:

- 가져오기 소스 → upstream은 `import` 스팬에 `fields(origin = "user", method = ?source)`로
  `method`에 담는다. `Method`는 화이트리스트에 있다. `import_source`는 중복이자 사문(死文).
- 내보내기 형식 → upstream은 페이지마다 `metric = "page_exported", format = ?format`을
  발행한다. `Format`은 화이트리스트에 있다. 형식별 집계는 이쪽에서 나온다.

즉 **두 필드 모두 지워도 잃는 정보가 없다.** 남겨두면 "보내고 있다고 착각하는 필드"가 된다.

확인 방법 — 화이트리스트에 없는 이름 찾기:

```sh
python - <<'PY'
import re, subprocess
src = open('crates/koharu-metrics/src/fields.rs', encoding='utf-8').read()
enum = re.search(r'pub enum EventField \{(.*?)\n\}', src, re.S).group(1)
snake = lambda n: re.sub(r'(?<!^)(?=[A-Z])', '_', n).lower()
allowed = {snake(x.strip().rstrip(',')) for x in enum.strip().split('\n') if x.strip()}
out = subprocess.run(['grep', '-rn', 'koharu_metrics', 'crates', '--include=*.rs', '-A8'],
                     capture_output=True, text=True).stdout
used = set()
for line in out.split('\n'):
    body = re.sub(r'^.*?\.rs[:-]\d+[:-]', '', line)
    used |= {m.group(1) for m in re.finditer(r'(?:^|[,(\s])([a-z][a-z0-9_]*)\s*=\s*[^=]', body)}
print(sorted(used - allowed))
PY
```

출력에는 평범한 `let` 바인딩(`filter`, `pipeline`, `progress`, `publish_control`,
`result`, `str`)이 섞여 나온다. 실제 계측 필드는 `export_formats`, `import_source` 둘뿐이다.

---

## 3. 이중 발행: `import`

`crates/koharu-app/src/commands/lifecycle.rs`의 `import_pages` 하나에 둘이 겹쳐 있다.

- 함수 어트리뷰트 `#[tracing::instrument(target = "koharu_metrics", name = "import",
  fields(origin = "user", method = ?source))]` → 스팬이 닫힐 때 `import` 발행
  (`origin`, `method`, `duration_ms`)
- 함수 본문 끝의 `tracing::info!(metric = "import", import_source = ?source, page_count)`
  → **같은 이름으로 한 번 더** 발행 (`import_source`는 버려지므로 실질 `page_count`뿐)

병합 시 upstream의 새 이벤트 `page_imported`도 함께 남겨서, 지금은 한 번의 가져오기가
`import` ×2 + `page_imported` ×1을 낸다. `import` 카운트가 2배로 부풀고, 두 발행의 필드
구성이 달라 집계도 갈라진다.

RPC 경로(`crates/koharu-rpc/src/routes/pages.rs:234`)는 `lifecycle::import_pages`를 직접
부르므로 스팬이 그대로 걸린다. 데스크톱·RPC 양쪽에서 동일하게 이중 발행된다.

**제안:** 본문의 `metric = "import"` 이벤트를 지운다. 스팬이 `method`·`duration_ms`를,
`page_imported`가 `page_count`를 이미 담당한다. 잃는 정보는 없다.

---

## 4. 반쪽 발행: `export` — 스팬이 엉뚱한 구간을 잰다

`export`는 `import`와 성격이 다르다. 스팬과 이벤트가 **서로 다른 함수**에 붙어 있다.

- `crates/koharu-app/src/commands/output.rs`의 `export_pages`(Tauri 커맨드)에 `export`
  스팬. 병합 때 upstream의 `fields(format = ?format)`을 우리 시그니처에 맞춰
  `export_formats = ?options.formats`로 고쳤는데, §2대로 이 필드는 버려진다.
- 실제 내보내기는 `start_export`가 `tokio::spawn`한 detached 태스크 안의
  `export_pages_to`에서 돈다. 집계 이벤트 `metric = "export"`도 거기 있다.

여기서 두 가지가 따라온다.

**(a) `duration_ms`가 내보내기 시간이 아니다.** `export_pages`는 폴더 선택창을 띄우고
Job을 spawn한 직후 반환한다. 스팬은 그때 닫히므로 `duration_ms`는 **사용자가 폴더를 고른
시간**을 잰다. 내보내기 자체의 소요 시간이 아니다. 스팬이 spawn된 태스크로 전파되지도 않아
`export` 이벤트는 이 스팬의 자식도 아니다.

**(b) RPC 경로에는 스팬이 없다.** `crates/koharu-rpc/src/routes/pages.rs:402,422`는
`output::start_export`를 직접 부른다. `export_pages`를 거치지 않으므로 `export` 스팬이
아예 발행되지 않는다. 그 경로에서 유일한 신호는 본문의 `metric = "export"` 이벤트다.

즉 `export` 이벤트를 §3처럼 그냥 지우면 **RPC 내보내기가 계측에서 통째로 사라진다.**
`import`와 똑같이 처리하면 안 된다.

**제안(둘 중 하나):**

1. `export_pages`의 스팬 어트리뷰트를 떼고, `export_pages_to` 쪽 이벤트 하나로 통일한다.
   `export_formats`는 `format`(화이트리스트)으로 바꾸거나, 형식별 집계는
   `page_exported`에 맡기고 그냥 지운다. 가장 단순하고 두 경로가 같아진다.
2. 스팬을 `export_pages_to`로 옮긴다. `duration_ms`가 진짜 내보내기 시간이 되고 RPC
   경로도 덮는다. 대신 `origin = "user"`를 인자로 받아 경로별로 구분해 넘겨야 한다.

1번을 권한다. 2번의 `origin` 배선은 이번 병합 범위를 넘는다.

확인 방법:

```sh
grep -n -B6 'pub async fn export_pages' crates/koharu-app/src/commands/output.rs
grep -n 'export_pages_to\|tokio::spawn' crates/koharu-app/src/commands/output.rs
grep -n 'start_export' crates/koharu-rpc/src/routes/pages.rs
```

---

## 5. 지금 상태 요약

| 이벤트 | 발행 횟수 | 실려 나가는 필드 | 판정 |
|---|---|---|---|
| `import` (스팬) | 가져오기당 1 | `origin`, `method`, `duration_ms` | 정상 |
| `import` (이벤트) | 가져오기당 1 | `page_count` (`import_source`는 유실) | **중복 — 제거 대상** |
| `page_imported` | 가져오기당 1 | `page_count` | 정상 (upstream 신규) |
| `export` (스팬) | 데스크톱 경로만 1 | `origin`, `duration_ms`(폴더 선택 시간) | **구간 오측 + RPC 미포함** |
| `export` (이벤트) | 양쪽 경로 1 | `page_count` (`export_formats`는 유실) | RPC의 유일한 신호 — 함부로 못 지움 |
| `page_exported` | 페이지×형식당 1 | `format` | 정상 (upstream 신규) |

병합 자체는 이 상태로 커밋했다(`09875684`). 정리는 별도 커밋으로 하되, §3과 §4를 한꺼번에
건드리지 말 것 — §4는 RPC 경로 계측이 걸려 있어 판단이 필요하다.

---

## 6. 아직 안 한 것

- 프론트엔드(`packages/koharu`) 검증. 병합 충돌은 없었으나 `bun test` / 타입체크 미실행.
- 런타임 확인. 위 주장은 전부 코드 독해다. 실제 전송 페이로드를 본 적이 없고, §1(3) 때문에
  콘솔 로그로는 볼 수도 없다. 확인하려면 `koharu-metrics`의 전송 지점에 임시 덤프를 넣거나
  `crates/koharu-metrics/src/lib.rs`의 테스트를 확장해야 한다.
