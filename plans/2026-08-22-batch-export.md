# 일괄 내보내기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 프로젝트의 페이지를 PNG·PSD로 한 번에 내보내되, 진행률을 보여주고 중단할 수 있게 한다.

**Architecture:** 내보내기를 기존 Job 인프라(`Processing`, `JobChannel`, SSE `job` 이벤트, `/process/stop`) 위의 작업으로 만든다. 새 개념은 파일명 템플릿 모듈과 브라우저 ZIP을 2단계로 나누기 위한 스테이징 상태 둘뿐이다. 프론트엔드는 형식·대상·파일명·폴더 구조를 고르는 대화상자를 추가하고, 진행 표시는 이미 있는 `ActivityCenter`를 재사용한다.

**Tech Stack:** Rust (axum, tokio, tauri, anyhow), TypeScript (Next.js 16, React 19, zustand, i18next), 테스트는 `cargo test`와 vitest.

**설계 문서:** `specs/2026-08-22-batch-export-design.md`

**계획 위치에 대한 메모:** 기본 경로는 `docs/superpowers/plans/`지만 `docs/`는 `docs_dir = "."`에 명시적 `nav`를 가진 발행용 문서 사이트다. 거기 두면 사용자용 사이트에 고아 페이지로 섞이므로 발행 대상이 아닌 `plans/`에 둔다.

---

## File Structure

**신규**

| 파일 | 책임 |
|---|---|
| `crates/koharu-app/src/commands/naming.rs` | 파일명 템플릿 파싱·렌더·충돌 해소. Tauri 의존성 없는 순수 로직 |
| `packages/koharu/components/app/ExportDialog.tsx` | 내보내기 옵션 대화상자 |
| `packages/koharu/tests/components/export-dialog.test.tsx` | 위 컴포넌트 테스트 |

**수정**

| 파일 | 변경 |
|---|---|
| `crates/koharu-app/src/commands/mod.rs` | `naming` 모듈 등록 |
| `crates/koharu-app/src/commands/output.rs` | `ExportOptions`, `export_pages_to` 시그니처, `start_export` job 래퍼 |
| `crates/koharu-app/src/commands/processing.rs` | `Job.kind`, `JobKind` |
| `crates/koharu-rpc/src/routes/pages.rs` | 내보내기 4개 라우트, `archive_directory` 재귀 |
| `crates/koharu-rpc/src/lib.rs` | `ExportStaging` 상태 관리 |
| `packages/bridge/src/protocol.ts` | `Job.kind`, `ExportOptions`, 내보내기 명령 시그니처 |
| `packages/koharu/lib/transfer.ts` | `runExport` 2단계 흐름 |
| `packages/koharu/components/app/TitleBar.tsx` | 메뉴 두 항목 → 대화상자 한 항목 |
| `packages/koharu/components/editor/ActivityCenter.tsx` | `job.kind`로 라벨 선택 |
| `packages/koharu/public/locales/*/translation.json` | 9개 로케일 신규 문구 |

---

### Task 1: 파일명 템플릿 모듈

자유 템플릿의 복잡도를 Tauri 의존성 없는 한 파일에 가둔다. 여기가 이 기능에서 유일하게 단위 테스트가 촘촘히 붙는 곳이다.

**Files:**
- Create: `crates/koharu-app/src/commands/naming.rs`
- Modify: `crates/koharu-app/src/commands/mod.rs`

- [ ] **Step 1: 모듈을 등록한다**

`crates/koharu-app/src/commands/mod.rs`의 다른 `pub mod` 선언 옆에 알파벳 순서를 지켜 추가한다.

```rust
pub mod naming;
```

- [ ] **Step 2: 실패하는 테스트를 쓴다**

`crates/koharu-app/src/commands/naming.rs`를 만들고 테스트만 먼저 넣는다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_pattern_reproduces_the_previous_names() {
        let template = Template::parse("{index:04}_{label}").unwrap();
        assert_eq!(template.render(1, "page-01").unwrap(), "0001_page-01");
        assert_eq!(template.render(42, "page-42").unwrap(), "0042_page-42");
    }

    #[test]
    fn an_index_without_a_width_is_not_padded() {
        let template = Template::parse("{index}").unwrap();
        assert_eq!(template.render(7, "ignored").unwrap(), "7");
    }

    #[test]
    fn a_width_pads_with_zeroes_and_never_truncates() {
        let template = Template::parse("{index:2}").unwrap();
        assert_eq!(template.render(5, "x").unwrap(), "05");
        assert_eq!(template.render(1234, "x").unwrap(), "1234");
    }

    #[test]
    fn a_label_loses_its_extension_and_its_forbidden_characters() {
        let template = Template::parse("{label}").unwrap();
        assert_eq!(template.render(1, "page:01.png").unwrap(), "page_01");
    }

    #[test]
    fn an_unknown_token_is_rejected() {
        let error = Template::parse("{page}").unwrap_err().to_string();
        assert!(error.contains("page"), "{error}");
    }

    #[test]
    fn an_unclosed_brace_is_rejected() {
        let error = Template::parse("{index").unwrap_err().to_string();
        assert!(error.contains("unclosed"), "{error}");
    }

    #[test]
    fn a_non_numeric_width_is_rejected() {
        assert!(Template::parse("{index:wide}").is_err());
    }

    #[test]
    fn a_separator_in_a_literal_is_rejected() {
        assert!(Template::parse("out/{label}").is_err());
        assert!(Template::parse("out\\{label}").is_err());
        assert!(Template::parse("..{label}").is_err());
    }

    #[test]
    fn an_empty_result_falls_back_to_page() {
        let template = Template::parse("{label}").unwrap();
        assert_eq!(template.render(1, "   ").unwrap(), "page");
    }

    #[test]
    fn colliding_names_get_a_numeric_suffix() {
        let mut names = Names::default();
        assert_eq!(names.unique("cover".to_owned()), "cover");
        assert_eq!(names.unique("cover".to_owned()), "cover_2");
        assert_eq!(names.unique("cover".to_owned()), "cover_3");
    }

    #[test]
    fn the_collision_check_ignores_case_because_windows_does() {
        let mut names = Names::default();
        assert_eq!(names.unique("Cover".to_owned()), "Cover");
        assert_eq!(names.unique("cover".to_owned()), "cover_2");
    }
}
```

- [ ] **Step 3: 테스트가 실패하는지 확인한다**

Run: `cargo test -p koharu-app naming`
Expected: FAIL — `cannot find type Template in this scope`

- [ ] **Step 4: 구현한다**

위 `mod tests` 블록 **앞에** 붙인다.

```rust
//! 내보내기 파일명 템플릿.
//!
//! 사용자가 자유 문자열로 패턴을 쓰기 때문에, 파싱과 검증이 한곳에 모여
//! 있어야 한다. 이 모듈은 Tauri에도 프로젝트 상태에도 의존하지 않으므로
//! 단위 테스트로 전부 덮을 수 있다.

use std::collections::HashSet;

use anyhow::{Result, anyhow};

/// 파일 이름에 쓸 수 없는 문자. Windows 기준이 가장 좁으므로 그것을 따른다.
const FORBIDDEN: [char; 9] = ['<', '>', ':', '"', '/', '\\', '|', '?', '*'];

#[derive(Debug)]
enum Piece {
    Literal(String),
    /// `{index}`는 width 0, `{index:04}`는 width 4.
    Index { width: usize },
    Label,
}

#[derive(Debug)]
pub struct Template(Vec<Piece>);

impl Template {
    /// 패턴을 파싱한다. 알 수 없는 토큰, 닫히지 않은 중괄호, 숫자가 아닌
    /// 자릿수, 경로 구분자가 섞인 리터럴을 여기서 모두 거른다. 대화상자가
    /// 입력할 때마다 이 함수를 불러 오류를 보여준다.
    pub fn parse(pattern: &str) -> Result<Self> {
        let mut pieces = Vec::new();
        let mut literal = String::new();
        let mut rest = pattern;
        while let Some(open) = rest.find('{') {
            literal.push_str(&rest[..open]);
            let after = &rest[open + 1..];
            let close = after
                .find('}')
                .ok_or_else(|| anyhow!("unclosed '{{' in the filename pattern"))?;
            let token = &after[..close];
            if !literal.is_empty() {
                pieces.push(Piece::Literal(std::mem::take(&mut literal)));
            }
            pieces.push(parse_token(token)?);
            rest = &after[close + 1..];
        }
        if rest.contains('}') {
            return Err(anyhow!("unmatched '}}' in the filename pattern"));
        }
        literal.push_str(rest);
        if !literal.is_empty() {
            pieces.push(Piece::Literal(literal));
        }
        for piece in &pieces {
            if let Piece::Literal(text) = piece {
                check_literal(text)?;
            }
        }
        Ok(Self(pieces))
    }

    /// 한 페이지의 파일 이름 줄기(확장자 없음)를 만든다.
    ///
    /// `index`는 형식과 무관한 페이지의 1-기반 순번이다. PNG와 PSD를 함께
    /// 내면 같은 페이지의 두 파일이 같은 번호를 갖는다.
    pub fn render(&self, index: usize, label: &str) -> Result<String> {
        let mut name = String::new();
        for piece in &self.0 {
            match piece {
                Piece::Literal(text) => name.push_str(text),
                Piece::Index { width } => {
                    name.push_str(&format!("{index:0width$}", index = index, width = width));
                }
                Piece::Label => name.push_str(&sanitize_label(label)),
            }
        }
        let name = name.trim().to_owned();
        // 리터럴은 parse에서 이미 걸렀지만, 라벨과 리터럴이 이어 붙어
        // `..`가 생기는 경우가 남는다. 최종 결과도 확인한다.
        check_literal(&name)?;
        Ok(if name.is_empty() {
            "page".to_owned()
        } else {
            name
        })
    }
}

fn parse_token(token: &str) -> Result<Piece> {
    let (name, width) = match token.split_once(':') {
        Some((name, width)) => {
            let width = width
                .parse::<usize>()
                .map_err(|_| anyhow!("'{width}' is not a number of digits"))?;
            (name, width)
        }
        None => (token, 0),
    };
    match name {
        "index" => Ok(Piece::Index { width }),
        "label" => Ok(Piece::Label),
        other => Err(anyhow!(
            "unknown token '{{{other}}}' in the filename pattern; use {{index}} or {{label}}"
        )),
    }
}

/// 경로 이탈을 막는다. 결과는 언제나 대상 폴더 안의 **한** 파일이어야 한다.
fn check_literal(text: &str) -> Result<()> {
    if text.contains('/') || text.contains('\\') {
        return Err(anyhow!(
            "the filename pattern cannot contain a path separator"
        ));
    }
    if text.contains("..") {
        return Err(anyhow!("the filename pattern cannot contain '..'"));
    }
    Ok(())
}

/// 페이지 라벨을 파일 이름에 쓸 수 있게 다듬는다.
///
/// 기존 `export_pages_to`의 규칙을 그대로 옮긴 것이다: 뒤쪽 점과 공백을
/// 떼고, 확장자를 떼고, 쓸 수 없는 문자를 `_`로 바꾼다.
pub fn sanitize_label(label: &str) -> String {
    let name = label
        .trim()
        .trim_end_matches(|character: char| character == '.' || character.is_whitespace());
    let name = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
    name.chars()
        .map(|character| {
            if FORBIDDEN.contains(&character) {
                '_'
            } else {
                character
            }
        })
        .collect()
}

/// 이미 쓴 이름을 기억해 충돌에 접미사를 붙인다.
///
/// `{label}`만 쓰는 패턴에서 같은 라벨이 두 번 나오면 두 번째가 첫 번째를
/// 덮어쓴다. 대소문자를 구분하지 않는 것은 Windows가 그러기 때문이다.
#[derive(Default)]
pub struct Names {
    used: HashSet<String>,
}

impl Names {
    pub fn unique(&mut self, name: String) -> String {
        let key = name.to_lowercase();
        if self.used.insert(key) {
            return name;
        }
        for suffix in 2.. {
            let candidate = format!("{name}_{suffix}");
            if self.used.insert(candidate.to_lowercase()) {
                return candidate;
            }
        }
        unreachable!("the suffix range is unbounded")
    }
}
```

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

Run: `cargo test -p koharu-app naming`
Expected: PASS — 11 tests

- [ ] **Step 6: 커밋**

```bash
git add crates/koharu-app/src/commands/naming.rs crates/koharu-app/src/commands/mod.rs
git commit -m "feat(export): filename templates for batch export"
```

---

### Task 2: `export_pages_to`에 옵션·진행률·취소를 넣는다

Job 배선은 아직 하지 않는다. 코어만 바꾸고 기존 호출부는 컴파일만 되도록 최소로 맞춘다.

**Files:**
- Modify: `crates/koharu-app/src/commands/output.rs:31-165`
- Modify: `crates/koharu-rpc/src/routes/pages.rs` (호출부 3곳)

- [ ] **Step 1: `ExportFormat`에 확장자와 하위 폴더 이름을 붙인다**

`crates/koharu-app/src/commands/output.rs`의 `ExportFormat` 정의 바로 아래에 추가한다.

```rust
impl ExportFormat {
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Psd => "psd",
        }
    }

    /// 형식별 하위 폴더 이름. 확장자와 같지만 뜻이 달라 따로 둔다.
    #[must_use]
    pub fn subfolder(self) -> &'static str {
        self.extension()
    }
}
```

- [ ] **Step 2: `ExportOptions`를 정의한다**

같은 파일, `ExportFormat` 아래에 추가한다.

```rust
/// 한 번의 내보내기 실행이 받는 선택지.
#[derive(Clone, Debug, Deserialize, Type)]
pub struct ExportOptions {
    /// 최소 하나. 둘을 함께 주면 페이지마다 두 파일이 나온다.
    pub formats: Vec<ExportFormat>,
    /// `crate::commands::naming::Template`이 파싱하는 패턴.
    pub pattern: String,
    /// 형식이 둘일 때 `png/`, `psd/`로 나눌지.
    pub subfolders: bool,
}

impl ExportOptions {
    /// 대상 폴더 아래에서 이 형식이 쓰일 경로.
    fn directory(&self, root: &std::path::Path, format: ExportFormat) -> std::path::PathBuf {
        if self.subfolders {
            root.join(format.subfolder())
        } else {
            root.to_owned()
        }
    }
}
```

- [ ] **Step 3: `export_pages_to`를 새 시그니처로 바꾼다**

`export_pages_to` 전체를 아래로 교체한다. 이름 계산은 `naming`으로 넘기고, 형식마다 파일을 하나씩 쓰며, 각 항목 앞에서 취소를 확인하고 완료마다 진행률을 알린다.

```rust
/// [`export_pages`]의 코어. 네이티브 대화상자 대신 명시적 출력 폴더를 받는다.
///
/// `pages`가 비어 있으면 프로젝트의 모든 페이지가 대상이다. 진행률의 분모는
/// `pages × formats`지만 파일 이름의 번호는 형식과 무관한 페이지 순번이다 —
/// 두 값은 서로 다른 것을 센다.
pub async fn export_pages_to(
    directory: std::path::PathBuf,
    pages: Vec<EntityId>,
    options: ExportOptions,
    progress: Arc<dyn Fn(usize, usize, EntityId) + Send + Sync>,
    stop: StopToken,
    project: State<'_, CurrentProject>,
    desktop: State<'_, Desktop>,
) -> std::result::Result<(), Error> {
    if options.formats.is_empty() {
        return Err(anyhow::anyhow!("no export format was selected").into());
    }
    let template = crate::commands::naming::Template::parse(&options.pattern)?;
    let snapshot = {
        let project = project.project.lock().await;
        let project = project.as_ref().context("no project is open")?;
        project.snapshot()
    };
    let pages = if pages.is_empty() {
        snapshot.pages().map(|page| page.id()).collect()
    } else {
        pages
    };
    if pages.is_empty() {
        return Err(anyhow::anyhow!("there are no pages to export").into());
    }
    let page_count = pages.len();

    // 이름은 형식과 무관하게 페이지마다 한 번 정해진다. 충돌 해소도 여기서
    // 끝나야 PNG와 PSD가 같은 줄기를 공유한다.
    let mut names = crate::commands::naming::Names::default();
    let jobs = pages
        .into_iter()
        .enumerate()
        .map(|(index, page_id)| {
            let page = snapshot.page(page_id)?.page()?;
            let stem = template.render(index + 1, &page.label)?;
            Ok::<_, anyhow::Error>((page_id, names.unique(stem)))
        })
        .collect::<Result<Vec<_>>>()?;

    for format in &options.formats {
        let target = options.directory(&directory, *format);
        tokio::fs::create_dir_all(&target)
            .await
            .with_context(|| format!("failed to create {}", target.display()))?;
    }

    let total = page_count.saturating_mul(options.formats.len());
    let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let renderer = desktop.renderer();
    let rasterizer = desktop.rasterizer().await?;
    let units: Vec<_> = jobs
        .into_iter()
        .flat_map(|(page_id, stem)| {
            options
                .formats
                .iter()
                .map(move |format| (page_id, stem.clone(), *format))
                .collect::<Vec<_>>()
        })
        .collect();

    stream::iter(units)
        .map(|(page_id, stem, format)| {
            let renderer = renderer.clone();
            let rasterizer = Arc::clone(&rasterizer);
            let snapshot = snapshot.clone();
            let target = options.directory(&directory, format);
            let stop = stop.clone();
            let progress = Arc::clone(&progress);
            let completed = Arc::clone(&completed);
            async move {
                // 취소는 협조적이다. 이미 시작된 최대 4건은 마저 끝난다.
                if stop.stopped() {
                    return Ok::<_, anyhow::Error>(());
                }
                let frame = renderer.render(&snapshot, page_id).await?;
                match format {
                    ExportFormat::Png => {
                        let image =
                            rasterize(Arc::clone(&rasterizer), &frame, RasterOptions::default())
                                .await?
                                .image;
                        let path = target.join(format!("{stem}.png"));
                        tokio::task::spawn_blocking(move || -> Result<()> {
                            let file = std::fs::File::create(path)?;
                            PngEncoder::new_with_quality(
                                file,
                                CompressionType::Best,
                                FilterType::Adaptive,
                            )
                            .write_image(
                                image.as_raw(),
                                image.width(),
                                image.height(),
                                ExtendedColorType::Rgba8,
                            )?;
                            Ok(())
                        })
                        .await
                        .context("PNG export worker stopped unexpectedly")??;
                    }
                    ExportFormat::Psd => {
                        let bytes = export_page(
                            Arc::clone(&rasterizer),
                            &snapshot,
                            &frame,
                            &PsdExportOptions::default(),
                        )
                        .await?;
                        tokio::fs::write(target.join(format!("{stem}.psd")), bytes).await?;
                    }
                }
                let done =
                    completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                progress(done, total, page_id);
                Ok(())
            }
        })
        .buffer_unordered(4)
        .try_collect::<Vec<_>>()
        .await?;

    tracing::info!(
        target: "koharu_metrics",
        metric = "export",
        export_formats = ?options.formats,
        page_count,
    );
    Ok(())
}
```

파일 상단 `use`에 `use koharu_pipeline::StopToken;`을 추가한다.

- [ ] **Step 4: 네이티브 대화상자 명령을 맞춘다**

같은 파일의 `export_pages` 커맨드에서 `format: ExportFormat` 인자를 `options: ExportOptions`로 바꾸고, 마지막 호출을 다음으로 교체한다. 진행률과 취소는 Task 3에서 붙이므로 여기서는 무해한 기본값을 준다.

```rust
    export_pages_to(
        directory,
        pages,
        options,
        Arc::new(|_, _, _| {}),
        StopToken::default(),
        project,
        desktop,
    )
    .await
```

- [ ] **Step 5: rpc 호출부 3곳을 맞춘다**

`crates/koharu-rpc/src/routes/pages.rs`에서 `ExportPagesRequest`, `ExportDialogRequest`, `ExportDownloadRequest`의 `format: ExportFormat` 필드를 `options: ExportOptions`로 바꾸고, 세 핸들러의 `request.format` 인자를 `request.options`로 바꾼 뒤 `export_pages_to` 호출마다 `Arc::new(|_, _, _| {})`와 `StopToken::default()`를 넘긴다. `use koharu_app::commands::output::{self, ExportFormat};`를 `use koharu_app::commands::output::{self, ExportOptions};`로 바꾼다.

- [ ] **Step 6: 컴파일과 기존 테스트를 확인한다**

Run: `cargo check -p koharu-app -p koharu-rpc && cargo test -p koharu-app`
Expected: 오류 없음, 기존 테스트 전부 통과

- [ ] **Step 7: 커밋**

```bash
git add crates/koharu-app/src/commands/output.rs crates/koharu-rpc/src/routes/pages.rs
git commit -m "feat(export): multiple formats, subfolders and cooperative cancellation"
```

---

### Task 3: 내보내기를 Job으로 만든다

**Files:**
- Modify: `crates/koharu-app/src/commands/processing.rs:39-50`
- Modify: `crates/koharu-app/src/commands/output.rs`

- [ ] **Step 1: `JobKind`를 추가한다**

`crates/koharu-app/src/commands/processing.rs`의 `Job` 정의 바로 위에 넣고, `Job`에 필드를 더한다.

```rust
/// 작업의 종류. `stage`와 `model`은 파이프라인 전용이라 내보내기에서는
/// 비므로, 프론트엔드가 라벨을 고를 근거가 따로 필요하다.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Processing,
    Export,
}
```

`Job` 구조체의 `id` 다음 줄에 추가한다.

```rust
    pub kind: JobKind,
```

`process()`가 만드는 `Job` 리터럴(`processing.rs:118` 부근)에 `kind: JobKind::Processing,`을 넣는다.

- [ ] **Step 2: 내보내기 Job 래퍼를 쓴다**

`crates/koharu-app/src/commands/output.rs` 끝에 추가한다. 순서는 `process()`(`processing.rs:110-130`)를 그대로 따른다.

```rust
/// 내보내기를 백그라운드 Job으로 시작하고 그 id를 즉시 돌려준다.
///
/// `Processing.stops`의 단일 작업 제약을 그대로 쓴다. 내보내기와 파이프라인은
/// 둘 다 GPU를 많이 쓰므로 동시에 돌 이유가 없고, 이 제약 덕분에 스테이징
/// 디렉터리도 한 번에 하나만 존재한다.
pub async fn start_export(
    handle: AppHandle<Cef>,
    directory: std::path::PathBuf,
    pages: Vec<EntityId>,
    options: ExportOptions,
) -> std::result::Result<JobId, Error> {
    // 패턴은 여기서 한 번 검증한다. 대화상자가 이미 막지만 HTTP API는
    // 신뢰 경계이고, 경로 이탈을 클라이언트 검증에만 맡길 수 없다.
    crate::commands::naming::Template::parse(&options.pattern)?;

    let id = JobId::new();
    let stop = StopToken::default();
    {
        let processing = handle.state::<Processing>();
        let mut stops = processing.stops.lock();
        if !stops.is_empty() {
            return Err(anyhow::anyhow!("another process is already running").into());
        }
        stops.insert(id, stop.clone());
    }
    let job = Job {
        id,
        kind: JobKind::Export,
        state: JobState::Running,
        completed: 0,
        total: 0,
        page: None,
        stage: None,
        model: None,
        error: None,
    };
    handle.state::<Processing>().jobs.lock().insert(id, job.clone());
    handle.state::<JobChannel>().publish(job);

    let task_handle = handle.clone();
    let task_stop = stop.clone();
    drop(tokio::spawn(async move {
        let progress_handle = task_handle.clone();
        let progress: Arc<dyn Fn(usize, usize, EntityId) + Send + Sync> =
            Arc::new(move |completed, total, page| {
                let job = {
                    let processing = progress_handle.state::<Processing>();
                    let mut jobs = processing.jobs.lock();
                    jobs.get_mut(&id).map(|job| {
                        job.completed = completed;
                        job.total = total;
                        job.page = Some(page);
                        job.clone()
                    })
                };
                if let Some(job) = job {
                    progress_handle.state::<JobChannel>().publish(job);
                }
            });
        let result = export_pages_to(
            directory,
            pages,
            options,
            progress,
            task_stop.clone(),
            task_handle.state::<CurrentProject>(),
            task_handle.state::<Desktop>(),
        )
        .await;
        let (stopped, error) = match result {
            Ok(()) => (task_stop.stopped(), None),
            Err(error) => {
                tracing::error!(%error, "export failed");
                (false, Some(format!("{error}")))
            }
        };
        task_handle.state::<Processing>().stops.lock().remove(&id);
        let job = task_handle
            .state::<Processing>()
            .jobs
            .lock()
            .remove(&id)
            .map(|mut job| {
                job.state = if stopped {
                    JobState::Stopped
                } else if error.is_some() {
                    JobState::Failed
                } else {
                    JobState::Finished
                };
                job.error = error;
                job
            });
        if let Some(job) = job {
            task_handle.state::<JobChannel>().publish(job);
        }
    }));
    Ok(id)
}
```

파일 상단 `use`에 다음을 추가한다.

```rust
use tauri::{AppHandle, Manager as _};

use super::ChannelExt as _;
use super::processing::{Job, JobChannel, JobId, JobKind, JobState, Processing};
```

- [ ] **Step 3: 컴파일을 확인한다**

Run: `cargo check -p koharu-app`
Expected: 오류 없음

- [ ] **Step 4: 커밋**

```bash
git add crates/koharu-app/src/commands/processing.rs crates/koharu-app/src/commands/output.rs
git commit -m "feat(export): run exports as cancellable background jobs"
```

---

### Task 4: `archive_directory`를 재귀시킨다

형식별 하위 폴더를 켜면 지금 구현은 파일을 하나도 찾지 못해 `"the export produced no files"`로 실패한다. 렌더링은 멀쩡히 끝난 뒤라 오류 문구가 원인을 정반대로 가리킨다.

**Files:**
- Modify: `crates/koharu-rpc/src/routes/pages.rs:429-462`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`crates/koharu-rpc/src/routes/pages.rs` 맨 끝에 붙인다.

```rust
#[cfg(test)]
mod tests {
    use super::archive_directory;

    #[test]
    fn a_nested_export_keeps_its_subfolders_in_the_archive() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("png")).unwrap();
        std::fs::create_dir(root.path().join("psd")).unwrap();
        std::fs::write(root.path().join("png/0001_a.png"), b"png").unwrap();
        std::fs::write(root.path().join("psd/0001_a.psd"), b"psd").unwrap();

        let archive = archive_directory(root.path()).unwrap();
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive)).unwrap();
        let mut names: Vec<_> = (0..zip.len())
            .map(|index| zip.by_index(index).unwrap().name().to_owned())
            .collect();
        names.sort();
        assert_eq!(names, vec!["png/0001_a.png", "psd/0001_a.psd"]);
    }

    #[test]
    fn a_flat_export_is_unchanged() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("0002_b.png"), b"b").unwrap();
        std::fs::write(root.path().join("0001_a.png"), b"a").unwrap();

        let archive = archive_directory(root.path()).unwrap();
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive)).unwrap();
        assert_eq!(zip.len(), 2);
        assert_eq!(zip.by_index(0).unwrap().name(), "0001_a.png");
    }

    #[test]
    fn an_empty_export_is_still_an_error() {
        let root = tempfile::tempdir().unwrap();
        assert!(archive_directory(root.path()).is_err());
    }
}
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `cargo test -p koharu-rpc archive`
Expected: FAIL — `a_nested_export_keeps_its_subfolders_in_the_archive`가 `the export produced no files`로 실패

- [ ] **Step 3: 재귀 수집으로 바꾼다**

`archive_directory`를 아래로 교체하고 헬퍼를 그 아래 둔다.

```rust
/// `export_pages_to`가 방금 쓴 파일들을 ZIP으로 묶는다.
///
/// 형식별 하위 폴더를 쓰면 출력이 한 겹 더 깊어지므로 재귀해야 한다. 엔트리
/// 이름은 `root` 기준 상대 경로라 압축을 풀면 폴더 구조가 그대로 살아난다.
fn archive_directory(root: &std::path::Path) -> anyhow::Result<Vec<u8>> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    // Deflate earns little on PNG and a lot on PSD. One method for both keeps
    // this to a single path; the cost next to rendering the pages is noise.
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    let mut entries = Vec::new();
    collect_files(root, &mut entries)?;
    if entries.is_empty() {
        anyhow::bail!("the export produced no files");
    }
    entries.sort();
    for path in entries {
        let name = path
            .strip_prefix(root)
            .context("an export file escaped the staging directory")?
            .to_string_lossy()
            // ZIP은 언제나 '/'를 쓴다. Windows의 '\'를 그대로 두면 풀 때
            // 폴더가 아니라 이름에 역슬래시가 든 파일이 된다.
            .replace('\\', "/");
        let bytes = std::fs::read(&path)
            .with_context(|| format!("failed to read the rendered page {name}"))?;
        writer
            .start_file(&name, options)
            .with_context(|| format!("failed to add {name} to the export archive"))?;
        writer
            .write_all(&bytes)
            .with_context(|| format!("failed to write {name} into the export archive"))?;
    }
    Ok(writer
        .finish()
        .context("failed to finalize the export archive")?
        .into_inner())
}

/// `directory` 아래의 모든 파일 경로를 깊이 우선으로 모은다.
fn collect_files(
    directory: &std::path::Path,
    out: &mut Vec<std::path::PathBuf>,
) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(directory).context("failed to list the rendered export")? {
        let path = entry.context("failed to list the rendered export")?.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}
```

`tempfile`과 `zip`은 `koharu-rpc`의 일반 의존성이므로 같은 크레이트의 단위 테스트에서 그대로 쓸 수 있다. `[dev-dependencies]`에 더할 것은 없다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cargo test -p koharu-rpc archive`
Expected: PASS — 3 tests

- [ ] **Step 5: 커밋**

```bash
git add crates/koharu-rpc/src/routes/pages.rs
git commit -m "fix(export): archive nested export directories, not just the top level"
```

---

### Task 5: 라우트를 Job과 2단계 다운로드로 바꾼다

**Files:**
- Modify: `crates/koharu-app/src/commands/output.rs`
- Modify: `crates/koharu-rpc/src/routes/pages.rs`
- Modify: `crates/koharu-rpc/src/lib.rs`

- [ ] **Step 0: 폴더 선택 커맨드를 Job 방식으로 바꾼다**

`crates/koharu-app/src/commands/output.rs`의 `export_pages` Tauri 커맨드를 아래로 교체한다.

선택창은 `koharu-app`에 남는다. `rfd`는 이 크레이트의 의존성이고 `koharu-rpc`의 것이 아니므로 라우트 쪽으로 옮기면 의존성이 하나 늘어난다. 단일 작업 가드도 선택창 바로 옆에 있어야 한다 — 거절할 것이면 사용자를 폴더 선택으로 붙잡아두기 전에 거절해야 하기 때문이다.

폴더를 고른 뒤 `start_export`에 넘기므로 `project`/`desktop` State 인자는 더 이상 필요 없다. `start_export`가 `AppHandle`에서 직접 가져간다.

```rust
/// 네이티브 폴더 선택창을 띄우고 내보내기 Job을 시작한다.
///
/// 선택창을 띄우기 **전에** 다른 작업이 도는지 확인한다. 거절할 것이면
/// 사용자를 폴더 선택으로 붙잡아두기 전에 거절해야 한다.
#[tauri::command]
#[specta::specta]
pub async fn export_pages(
    window: WebviewWindow<Cef>,
    pages: Vec<EntityId>,
    options: ExportOptions,
) -> std::result::Result<Option<JobId>, Error> {
    let handle = window.app_handle().clone();
    if !handle.state::<Processing>().stops.lock().is_empty() {
        return Err(anyhow::anyhow!("another process is already running").into());
    }
    let Some(directory) = rfd::AsyncFileDialog::new()
        .set_parent(&window)
        .pick_folder()
        .await
        .map(|directory| directory.path().to_owned())
    else {
        return Ok(None);
    };
    Ok(Some(start_export(handle, directory, pages, options).await?))
}
```

- [ ] **Step 1: 스테이징 상태를 정의한다**

`crates/koharu-rpc/src/routes/pages.rs` 상단, `router()` 위에 넣는다.

```rust
/// 브라우저용 ZIP이 만들어지는 임시 디렉터리.
///
/// 다운로드가 2단계가 되면서 임시 디렉터리가 요청보다 오래 살아야 한다.
/// 단일 작업 제약 덕에 동시에 하나뿐이므로 맵이 아니라 슬롯 하나면 된다.
/// 새 내보내기가 시작되면 이전 것이 교체되며 `TempDir`의 Drop이 지운다.
/// `parking_lot`이 아니라 `std::sync::Mutex`인 것은 `koharu-rpc`가
/// `parking_lot`에 의존하지 않기 때문이다. 잠금 구간에 `await`가 없으므로
/// 표준 뮤텍스로 충분하다.
#[derive(Default)]
pub struct ExportStaging(std::sync::Mutex<Option<(JobId, tempfile::TempDir)>>);

impl ExportStaging {
    fn put(&self, job: JobId, directory: tempfile::TempDir) {
        *self.0.lock().expect("the export staging lock is never poisoned") = Some((job, directory));
    }

    /// 이 job의 스테이징을 꺼내 소유권을 넘긴다. 호출자가 놓으면 지워진다.
    fn take(&self, job: JobId) -> Option<tempfile::TempDir> {
        let mut slot = self.0.lock().expect("the export staging lock is never poisoned");
        match slot.take() {
            Some((held, directory)) if held == job => Some(directory),
            other => {
                *slot = other;
                None
            }
        }
    }
}
```

`use koharu_app::commands::processing::JobId;`를 상단 `use`에 추가한다.

- [ ] **Step 2: `koharu-rpc`가 이 상태를 관리하게 한다**

`crates/koharu-rpc/src/lib.rs`의 `serve` 안, `tokio::spawn` 앞에 넣는다.

```rust
    app.manage(routes::pages::ExportStaging::default());
```

`use tauri::Manager as _;`를 상단 `use`에 추가한다. `routes/pages.rs`의 `ExportStaging`은 이미 `pub`이다.

- [ ] **Step 3: 라우트를 등록한다**

`router()`의 `.route("/pages/export/download", post(export_download))` 다음 줄에 추가한다.

```rust
        .route("/pages/export/download/{job}", get(export_download_archive))
```

- [ ] **Step 4: 세 핸들러를 Job 방식으로 바꾼다**

`export_pages`, `export_dialog`, `export_download`를 아래로 교체한다.

```rust
#[derive(Deserialize)]
struct ExportPagesRequest {
    pages: Vec<EntityId>,
    options: ExportOptions,
    directory: String,
}

async fn export_pages(
    State(app): State<AppState>,
    Json(request): Json<ExportPagesRequest>,
) -> ApiResult<Json<JobId>> {
    let directory = PathBuf::from(request.directory);
    if !directory.is_dir() {
        return Err(anyhow::anyhow!("the export directory does not exist").into());
    }
    Ok(Json(
        output::start_export(app.clone(), directory, request.pages, request.options).await?,
    ))
}

#[derive(Deserialize)]
struct ExportDialogRequest {
    pages: Vec<EntityId>,
    options: ExportOptions,
}

/// 네이티브 폴더 선택은 `koharu-app`에 있다. `rfd`가 그 크레이트의 의존성이고
/// `koharu-rpc`의 것이 아니므로, 선택창을 여기로 옮기면 의존성이 하나 늘어난다.
/// 단일 작업 가드도 선택창 바로 옆에 있는 편이 낫다.
async fn export_dialog(
    State(app): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Json(request): Json<ExportDialogRequest>,
) -> ApiResult<Json<Option<JobId>>> {
    let window = require_local_window(&app, peer)?;
    Ok(Json(
        output::export_pages(window, request.pages, request.options).await?,
    ))
}

#[derive(Deserialize)]
struct ExportDownloadRequest {
    pages: Vec<EntityId>,
    options: ExportOptions,
}

/// 브라우저용 내보내기를 임시 디렉터리에 시작한다.
///
/// 요청 하나로 렌더링까지 끝내면 응답이 마지막에야 나가 진행률을 보낼 길이
/// 없다. 그래서 여기서는 Job만 시작하고, 클라이언트가 job이 끝나는 것을 SSE로
/// 본 뒤 `GET /pages/export/download/{job}`으로 ZIP을 받는다.
async fn export_download(
    State(app): State<AppState>,
    Json(request): Json<ExportDownloadRequest>,
) -> ApiResult<Json<JobId>> {
    let staging = tempfile::tempdir().context("failed to create a staging directory")?;
    let job = output::start_export(
        app.clone(),
        staging.path().to_owned(),
        request.pages,
        request.options,
    )
    .await?;
    app.state::<ExportStaging>().put(job, staging);
    Ok(Json(job))
}

/// 끝난 내보내기의 스테이징을 ZIP으로 넘기고 지운다.
async fn export_download_archive(
    State(app): State<AppState>,
    Path(job): Path<JobId>,
) -> ApiResult<impl IntoResponse> {
    let staging = app
        .state::<ExportStaging>()
        .take(job)
        .context("there is no finished export waiting for this job")?;
    let root = staging.path().to_owned();
    let archive = tokio::task::spawn_blocking(move || archive_directory(&root))
        .await
        .context("export archive worker stopped unexpectedly")??;
    drop(staging);
    Ok((
        [
            (header::CONTENT_TYPE, "application/zip"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"koharu-export.zip\"",
            ),
        ],
        Bytes::from(archive),
    ))
}
```

`use koharu_app::commands::processing::{JobId, Processing};`로 합치고, `use koharu_app::commands::output::{self, ExportOptions};`인지 확인한다.

- [ ] **Step 5: 컴파일과 테스트를 확인한다**

Run: `cargo test -p koharu-rpc`
Expected: 컴파일 성공, 기존 7개 + Task 4의 3개 테스트 통과

- [ ] **Step 6: 커밋**

```bash
git add crates/koharu-rpc/src/routes/pages.rs crates/koharu-rpc/src/lib.rs
git commit -m "feat(export): return a job id and hand the browser its archive in two steps"
```

---

### Task 6: `protocol.ts`를 맞춘다

이 파일은 생성되지 않고 손으로 쓴 것이다. 자동 생성을 기다릴 게 없다.

**Files:**
- Modify: `packages/bridge/src/protocol.ts:605-618` (타입), `commands` 객체의 내보내기 항목

- [ ] **Step 1: 타입을 갱신한다**

`export type Job`에 `kind`를 넣고, 바로 아래에 `JobKind`와 `ExportOptions`를 추가한다.

```ts
export type Job = {
	id: JobId,
	kind: JobKind,
	state: JobState,
	completed: number,
	total: number,
	page: EntityId | null,
	stage: Stage | null,
	model: string | null,
	error: string | null,
};

export type JobKind = "processing" | "export";

export type ExportOptions = {
	formats: ExportFormat[],
	pattern: string,
	subfolders: boolean,
};
```

- [ ] **Step 2: 명령을 갱신한다**

`commands` 객체의 `exportPages`, `exportPagesDialog`, `exportPagesDownload`를 아래로 교체한다.

```ts
	// 셋 다 즉시 `JobId`를 돌려주고 렌더링은 백그라운드에서 돈다. 진행률은
	// `openEventStream`의 job 이벤트로 오고, 중단은 `stopJob`이다.
	exportPages: (pages: EntityId[], options: ExportOptions, directory: string) =>
		post<JobId>("/pages/export", { pages, options, directory }),
	// The desktop path: native folder picker, server-side, loopback only.
	// 사용자가 선택창을 닫으면 `null`이다.
	exportPagesDialog: (pages: EntityId[], options: ExportOptions) =>
		post<JobId | null>("/pages/export/dialog", { pages, options }),
	// The remote path. Rendered output has to reach the user's machine, so it
	// comes back as one ZIP rather than being written somewhere they cannot see.
	// 렌더링과 전송이 나뉘어 있다: 이것이 job을 시작하고,
	// `getExportArchive`가 끝난 뒤의 ZIP을 가져온다.
	exportPagesDownload: (pages: EntityId[], options: ExportOptions) =>
		post<JobId>("/pages/export/download", { pages, options }),
	getExportArchive: (job: JobId) => downloadGet(`/pages/export/download/${job}`),
```

- [ ] **Step 3: GET 다운로드 헬퍼를 추가한다**

기존 `download` 함수 바로 아래에 넣는다. 기존 `download`는 POST 전용이라 재사용할 수 없다.

```ts
/** `download`의 GET판. 본문 없는 이진 응답을 Blob으로 가져온다. */
async function downloadGet(path: string): Promise<Blob> {
	const response = await fetch(`${API_BASE_URL}${path}`, { headers: authHeaders() });
	if (!response.ok) throw new Error(await errorMessage(response));
	return response.blob();
}
```

- [ ] **Step 4: 타입 검사**

Run: `bun run --filter @koharu/bridge typecheck`
Expected: exit 0

- [ ] **Step 5: 커밋**

```bash
git add packages/bridge/src/protocol.ts
git commit -m "feat(bridge): export options, job kind and the two-step archive fetch"
```

---

### Task 7: `runExport`를 2단계 흐름으로 바꾼다

**Files:**
- Modify: `packages/koharu/lib/transfer.ts:57-64`
- Test: `packages/koharu/tests/lib/transfer.test.ts`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`packages/koharu/tests/lib/transfer.test.ts`에 추가한다.

```ts
describe('runExport', () => {
  const options = { formats: ['png'] as const, pattern: '{index:04}_{label}', subfolders: false }

  it('데스크톱에서는 대화상자 명령만 부르고 job id를 돌려준다', async () => {
    vi.stubGlobal('window', { __TAURI_INTERNALS__: {} })
    const dialog = vi.spyOn(commands, 'exportPagesDialog').mockResolvedValue('job-1')
    const download = vi.spyOn(commands, 'exportPagesDownload')

    await expect(runExport(['page-1'], { ...options, formats: ['png'] })).resolves.toBe('job-1')
    expect(dialog).toHaveBeenCalledWith(['page-1'], { ...options, formats: ['png'] })
    expect(download).not.toHaveBeenCalled()
  })

  it('브라우저에서는 다운로드 job을 시작하고 아카이브는 아직 받지 않는다', async () => {
    vi.stubGlobal('window', {})
    const download = vi.spyOn(commands, 'exportPagesDownload').mockResolvedValue('job-2')
    const archive = vi.spyOn(commands, 'getExportArchive')

    await expect(runExport([], { ...options, formats: ['psd'] })).resolves.toBe('job-2')
    expect(download).toHaveBeenCalledWith([], { ...options, formats: ['psd'] })
    expect(archive).not.toHaveBeenCalled()
  })
})

describe('finishExport', () => {
  it('브라우저에서만 아카이브를 받아 저장한다', async () => {
    vi.stubGlobal('window', {})
    const blob = new Blob(['zip'])
    const archive = vi.spyOn(commands, 'getExportArchive').mockResolvedValue(blob)

    await finishExport('job-3')
    expect(archive).toHaveBeenCalledWith('job-3')
  })

  it('데스크톱에서는 아무것도 하지 않는다', async () => {
    vi.stubGlobal('window', { __TAURI_INTERNALS__: {} })
    const archive = vi.spyOn(commands, 'getExportArchive')

    await finishExport('job-4')
    expect(archive).not.toHaveBeenCalled()
  })
})
```

파일 상단 import에 `finishExport`를 더한다.

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `cd packages/koharu && bun run test -- transfer`
Expected: FAIL — `finishExport` is not exported

- [ ] **Step 3: 구현한다**

`packages/koharu/lib/transfer.ts`의 `runExport`를 아래로 교체한다.

```ts
/**
 * 내보내기 Job을 시작한다. 렌더링은 백그라운드에서 돌고, 진행률은 job
 * 이벤트로 온다. 데스크톱에서 사용자가 폴더 선택을 취소하면 `null`이다.
 */
export async function runExport(
  pages: EntityId[],
  options: ExportOptions,
): Promise<JobId | null> {
  if (isEmbedded()) return commands.exportPagesDialog(pages, options)
  return commands.exportPagesDownload(pages, options)
}

/**
 * Job이 끝난 뒤 결과를 사용자에게 넘긴다.
 *
 * 데스크톱은 이미 사용자가 고른 폴더에 파일이 들어 있으므로 할 일이 없다.
 * 브라우저는 여기서 ZIP을 받아 저장한다 — 서버가 임시 디렉터리를 붙들고
 * 있으므로 이 호출이 그것을 비우는 역할도 한다.
 */
export async function finishExport(job: JobId): Promise<void> {
  if (isEmbedded()) return
  saveBlob(await commands.getExportArchive(job), 'koharu-export.zip')
}
```

import에 `type ExportOptions`와 `type JobId`를 더하고, 더 이상 쓰지 않는 `type ExportFormat`을 뺀다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cd packages/koharu && bun run test -- transfer`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add packages/koharu/lib/transfer.ts packages/koharu/tests/lib/transfer.test.ts
git commit -m "feat(app): start exports as jobs and fetch the archive when they finish"
```

---

### Task 8: 내보내기 대화상자

**Files:**
- Create: `packages/koharu/components/app/ExportDialog.tsx`
- Create: `packages/koharu/tests/components/export-dialog.test.tsx`
- Modify: `packages/koharu/components/app/TitleBar.tsx:104-120, 278-281`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`packages/koharu/tests/components/export-dialog.test.tsx`를 만든다.

```tsx
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { ExportDialog } from '@/components/app/ExportDialog'

function open(onStart = vi.fn()) {
  render(
    <ExportDialog open onOpenChange={() => {}} pages={['a', 'b']} selected={['a']} onStart={onStart} />,
  )
  return onStart
}

describe('ExportDialog', () => {
  it('형식을 모두 끄면 시작할 수 없다', async () => {
    const user = userEvent.setup()
    open()
    await user.click(screen.getByRole('checkbox', { name: /png/i }))
    expect(screen.getByRole('button', { name: /export/i })).toBeDisabled()
  })

  it('잘못된 패턴은 오류를 보여주고 시작을 막는다', async () => {
    const user = userEvent.setup()
    open()
    const pattern = screen.getByLabelText(/pattern/i)
    await user.clear(pattern)
    await user.type(pattern, '{{page}')
    expect(screen.getByRole('button', { name: /export/i })).toBeDisabled()
    expect(screen.getByRole('alert')).toBeInTheDocument()
  })

  it('기본 패턴의 미리보기를 보여준다', () => {
    open()
    expect(screen.getByText('0001_page-01.png')).toBeInTheDocument()
  })

  it('고른 옵션으로 시작한다', async () => {
    const user = userEvent.setup()
    const onStart = open()
    await user.click(screen.getByRole('button', { name: /export/i }))
    expect(onStart).toHaveBeenCalledWith(['a'], {
      formats: ['png'],
      pattern: '{index:04}_{label}',
      subfolders: false,
    })
  })
})
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `cd packages/koharu && bun run test -- export-dialog`
Expected: FAIL — `Failed to resolve import "@/components/app/ExportDialog"`

- [ ] **Step 3: 컴포넌트를 만든다**

`packages/koharu/components/app/ExportDialog.tsx`:

```tsx
'use client'

import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import type { EntityId, ExportFormat, ExportOptions } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import { Checkbox } from '@koharu/ui/components/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import { Input } from '@koharu/ui/components/input'
import { Label } from '@koharu/ui/components/label'
import { RadioGroup, RadioGroupItem } from '@koharu/ui/components/radio-group'

const DEFAULT_PATTERN = '{index:04}_{label}'

/**
 * `naming::Template::parse`의 클라이언트 쪽 짝.
 *
 * 서버가 다시 검증하므로 이것은 보안 장치가 아니라, 사용자가 시작 버튼을
 * 누른 뒤가 아니라 타이핑하는 동안 오류를 보게 하려는 것이다. 규칙이
 * 어긋나면 서버가 잡는다.
 */
export function previewPattern(pattern: string): { name: string } | { error: string } {
  let out = ''
  let rest = pattern
  while (rest.includes('{')) {
    const open = rest.indexOf('{')
    out += rest.slice(0, open)
    const after = rest.slice(open + 1)
    const close = after.indexOf('}')
    if (close === -1) return { error: 'unclosed' }
    const token = after.slice(0, close)
    const [name, width] = token.includes(':') ? token.split(':', 2) : [token, undefined]
    if (width !== undefined && !/^\d+$/.test(width)) return { error: 'width' }
    if (name === 'index') out += '1'.padStart(Number(width ?? 0), '0')
    else if (name === 'label') out += 'page-01'
    else return { error: 'token' }
    rest = after.slice(close + 1)
  }
  if (rest.includes('}')) return { error: 'unclosed' }
  out += rest
  if (out.includes('/') || out.includes('\\') || out.includes('..')) return { error: 'separator' }
  if (!out.trim()) return { name: 'page' }
  return { name: out.trim() }
}

export function ExportDialog({
  open,
  onOpenChange,
  pages,
  selected,
  onStart,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  pages: EntityId[]
  selected: EntityId[]
  onStart: (pages: EntityId[], options: ExportOptions) => void
}) {
  const { t } = useTranslation()
  const [png, setPng] = useState(true)
  const [psd, setPsd] = useState(false)
  const [scope, setScope] = useState<'all' | 'selected'>(selected.length ? 'selected' : 'all')
  const [pattern, setPattern] = useState(DEFAULT_PATTERN)
  const [subfolders, setSubfolders] = useState(false)

  const formats = useMemo(() => {
    const chosen: ExportFormat[] = []
    if (png) chosen.push('png')
    if (psd) chosen.push('psd')
    return chosen
  }, [png, psd])

  const preview = useMemo(() => previewPattern(pattern), [pattern])
  const bothFormats = formats.length === 2
  const invalid = 'error' in preview
  const canStart = formats.length > 0 && !invalid && pages.length > 0

  const start = () => {
    onStart(scope === 'selected' ? selected : [], {
      formats,
      pattern,
      // 하위 폴더 체크박스가 비활성일 때는 체크 상태와 무관하게 false다.
      subfolders: bothFormats && subfolders,
    })
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className='sm:max-w-md'>
        <DialogHeader>
          <DialogTitle>{t('export.title')}</DialogTitle>
          <DialogDescription>{t('export.description')}</DialogDescription>
        </DialogHeader>

        <div className='space-y-4'>
          <fieldset className='space-y-2'>
            <legend className='text-xs font-medium'>{t('export.scope')}</legend>
            <RadioGroup value={scope} onValueChange={(value) => setScope(value as 'all' | 'selected')}>
              <div className='flex items-center gap-2'>
                <RadioGroupItem value='all' id='export-scope-all' />
                <Label htmlFor='export-scope-all'>
                  {t('export.scopeAll', { count: pages.length })}
                </Label>
              </div>
              <div className='flex items-center gap-2'>
                <RadioGroupItem
                  value='selected'
                  id='export-scope-selected'
                  disabled={selected.length === 0}
                />
                <Label htmlFor='export-scope-selected'>
                  {t('export.scopeSelected', { count: selected.length })}
                </Label>
              </div>
            </RadioGroup>
          </fieldset>

          <fieldset className='space-y-2'>
            <legend className='text-xs font-medium'>{t('export.formats')}</legend>
            <div className='flex items-center gap-2'>
              <Checkbox id='export-png' checked={png} onCheckedChange={(v) => setPng(v === true)} />
              <Label htmlFor='export-png'>PNG</Label>
            </div>
            <div className='flex items-center gap-2'>
              <Checkbox id='export-psd' checked={psd} onCheckedChange={(v) => setPsd(v === true)} />
              <Label htmlFor='export-psd'>PSD</Label>
            </div>
          </fieldset>

          <div className='space-y-1'>
            <Label htmlFor='export-pattern'>{t('export.pattern')}</Label>
            <Input
              id='export-pattern'
              value={pattern}
              onChange={(event) => setPattern(event.target.value)}
              aria-invalid={invalid}
            />
            {/* `invalid` 대신 여기서 다시 좁히는 것은 TypeScript가 별도
                boolean으로는 유니온을 좁혀 주지 않기 때문이다. 미리보기는
                텍스트 노드가 쪼개지지 않도록 템플릿 문자열 하나로 낸다 —
                테스트가 문자열 전체로 찾는다. */}
            {'error' in preview ? (
              <p role='alert' className='text-[11px] text-destructive'>
                {t(`export.patternError.${preview.error}`)}
              </p>
            ) : (
              <p className='text-[11px] text-muted-foreground'>
                {`${preview.name}.${formats[0] ?? 'png'}`}
              </p>
            )}
          </div>

          <div className='flex items-center gap-2'>
            <Checkbox
              id='export-subfolders'
              checked={bothFormats && subfolders}
              disabled={!bothFormats}
              onCheckedChange={(v) => setSubfolders(v === true)}
            />
            <Label htmlFor='export-subfolders'>{t('export.subfolders')}</Label>
          </div>
        </div>

        <DialogFooter>
          <Button variant='ghost' onClick={() => onOpenChange(false)}>
            {t('export.cancel')}
          </Button>
          <Button disabled={!canStart} onClick={start}>
            {t('export.start')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `cd packages/koharu && bun run test -- export-dialog`
Expected: PASS — 4 tests

- [ ] **Step 5: 메뉴를 교체한다**

`packages/koharu/components/app/TitleBar.tsx`에서 `menu.exportPng`와 `menu.exportPsd` 두 `MenubarItem`을 하나로 바꾼다.

```tsx
              <MenubarItem
                disabled={!project || pages.length === 0}
                onClick={() => setExportOpen(true)}
              >
                {t('menu.export')}
              </MenubarItem>
```

`exportSelection` 함수(`TitleBar.tsx:278-281`)를 지운다. 대상 선택이 대화상자로 옮겨갔다.

컴포넌트 상단에 상태를 더한다.

```tsx
  const [exportOpen, setExportOpen] = useState(false)
```

`AboutDialog`를 렌더링하는 곳 옆에 추가한다.

```tsx
      <ExportDialog
        open={exportOpen}
        onOpenChange={setExportOpen}
        pages={pages.map((page) => page.id)}
        selected={selectedPages}
        onStart={(target, options) => {
          void call(runExport, target, options)
            .then((job) => (job ? finishExportWhenDone(job) : undefined))
            .catch(() => undefined)
        }}
      />
```

import을 정리한다. `import { finishExportWhenDone, runExport } from '@/lib/transfer'`로 바꾸고 `import { ExportDialog } from '@/components/app/ExportDialog'`를 더한다. `finishExport`가 아니라 `finishExportWhenDone`이다 — 전자는 job이 이미 끝난 뒤에 쓰는 것이고, TitleBar는 끝나기를 기다려야 한다. `useState`가 아직 import되지 않았다면 더한다.

- [ ] **Step 6: 완료 감시를 붙인다**

`packages/koharu/lib/transfer.ts`에 추가한다. job이 끝나는 것을 스토어에서 지켜보다 브라우저면 ZIP을 받는다.

```ts
/**
 * Job이 끝날 때까지 기다렸다가 `finishExport`를 부른다.
 *
 * 진행률은 이미 `ActivityCenter`가 보여주므로 여기서는 완료만 본다. 스토어
 * 구독을 쓰는 것은 job 이벤트가 SSE로 오기 때문이다 — 폴링할 것이 없다.
 */
export function finishExportWhenDone(job: JobId): Promise<void> {
  return new Promise((resolve) => {
    const check = (state: { jobs: Record<string, { state: string }> }) => {
      const current = state.jobs[job]
      if (!current || current.state === 'running') return
      unsubscribe()
      if (current.state === 'finished') void finishExport(job).then(resolve, () => resolve())
      else resolve()
    }
    const unsubscribe = useKoharuStore.subscribe(check)
    check(useKoharuStore.getState())
  })
}
```

`import { useKoharuStore } from '@/lib/store'`를 더한다.

- [ ] **Step 7: 전체 프론트 테스트를 돌린다**

Run: `bun run test`
Expected: 기존 테스트 + 신규 테스트 전부 통과

- [ ] **Step 8: 커밋**

```bash
git add packages/koharu/components/app/ExportDialog.tsx packages/koharu/components/app/TitleBar.tsx packages/koharu/lib/transfer.ts packages/koharu/tests/components/export-dialog.test.tsx
git commit -m "feat(app): batch export dialog with format, scope and filename options"
```

---

### Task 9: `ActivityCenter`가 내보내기를 내보내기라고 부르게 한다

**Files:**
- Modify: `packages/koharu/components/editor/ActivityCenter.tsx:91-97`

- [ ] **Step 1: 라벨 선택을 고친다**

`JobItem` 안의 라벨 `<span>`을 교체한다.

```tsx
          <span className='block truncate text-[12px] font-medium capitalize'>
            {job.kind === 'export'
              ? t('activity.exporting')
              : job.stage
                ? t(`phase.${job.stage}`, { defaultValue: job.stage })
                : t('activity.processing')}
          </span>
```

실패 문구도 종류에 맞춘다.

```tsx
  if (job.state === 'failed') {
    return (
      <Failure
        message={
          job.error ||
          t(job.kind === 'export' ? 'activity.exportFailed' : 'activity.processingFailed')
        }
        onDismiss={() => dismiss(job.id)}
      />
    )
  }
```

- [ ] **Step 2: 테스트를 돌린다**

Run: `bun run test`
Expected: 통과. 기존 테스트가 `Job` 객체를 만든다면 `kind: 'processing'`을 더해야 컴파일된다 — 실패하면 그 픽스처들을 고친다.

- [ ] **Step 3: 커밋**

```bash
git add packages/koharu/components/editor/ActivityCenter.tsx packages/koharu/tests
git commit -m "feat(app): label export jobs as exports in the activity center"
```

---

### Task 10: 9개 로케일

**Files:**
- Modify: `packages/koharu/public/locales/{en-US,es-ES,ja-JP,ko-KR,pt-BR,ru-RU,tr-TR,zh-CN,zh-TW}/translation.json`

- [ ] **Step 1: `menu` 키를 바꾼다**

9개 파일 모두에서 `menu.exportPng`와 `menu.exportPsd`를 지우고 `menu.export`를 넣는다 (알파벳 순서상 `menu.file` 앞).

| 로케일 | `menu.export` |
|---|---|
| en-US | `Export…` |
| es-ES | `Exportar…` |
| ja-JP | `書き出し…` |
| ko-KR | `내보내기…` |
| pt-BR | `Exportar…` |
| ru-RU | `Экспорт…` |
| tr-TR | `Dışa Aktar…` |
| zh-CN | `导出…` |
| zh-TW | `匯出…` |

- [ ] **Step 2: `activity` 키를 더한다**

| 키 | en-US | ko-KR |
|---|---|---|
| `activity.exporting` | `Exporting` | `내보내는 중` |
| `activity.exportFailed` | `Export failed.` | `내보내기에 실패했습니다.` |

| 로케일 | `exporting` | `exportFailed` |
|---|---|---|
| es-ES | `Exportando` | `Error al exportar.` |
| ja-JP | `書き出し中` | `書き出しに失敗しました。` |
| pt-BR | `Exportando` | `Falha ao exportar.` |
| ru-RU | `Экспорт` | `Не удалось выполнить экспорт.` |
| tr-TR | `Dışa aktarılıyor` | `Dışa aktarma başarısız oldu.` |
| zh-CN | `正在导出` | `导出失败。` |
| zh-TW | `正在匯出` | `匯出失敗。` |

- [ ] **Step 3: `export` 블록을 더한다**

9개 파일 모두에 `export` 블록을 넣는다. 각 파일의 기존 키 정렬 방식(알파벳 순)을 그대로 지킨다.

i18next는 `{{...}}`만 보간하므로 오류 문구 안의 `{index}`, `{label}`은 이스케이프 없이 그대로 쓴다.

en-US:

```json
  "export": {
    "cancel": "Cancel",
    "description": "Render the pages to image files.",
    "formats": "Formats",
    "pattern": "Filename pattern",
    "patternError": {
      "separator": "The pattern cannot contain a path separator or '..'.",
      "token": "Unknown token. Use {index} or {label}.",
      "unclosed": "Unmatched brace in the pattern.",
      "width": "The digit count has to be a number."
    },
    "scope": "Pages",
    "scopeAll": "All pages ({{count}})",
    "scopeSelected": "Selected ({{count}})",
    "start": "Export",
    "subfolders": "Separate folders per format",
    "title": "Export pages"
  },
```

ko-KR:

```json
  "export": {
    "cancel": "취소",
    "description": "페이지를 이미지 파일로 렌더링합니다.",
    "formats": "형식",
    "pattern": "파일명 패턴",
    "patternError": {
      "separator": "패턴에 경로 구분자나 '..'를 쓸 수 없습니다.",
      "token": "알 수 없는 토큰입니다. {index} 또는 {label}을 쓰세요.",
      "unclosed": "패턴의 중괄호가 맞지 않습니다.",
      "width": "자릿수는 숫자여야 합니다."
    },
    "scope": "페이지",
    "scopeAll": "전체 페이지 ({{count}})",
    "scopeSelected": "선택한 페이지 ({{count}})",
    "start": "내보내기",
    "subfolders": "형식별로 폴더 나누기",
    "title": "페이지 내보내기"
  },
```

es-ES:

```json
  "export": {
    "cancel": "Cancelar",
    "description": "Renderiza las páginas como archivos de imagen.",
    "formats": "Formatos",
    "pattern": "Patrón de nombre",
    "patternError": {
      "separator": "El patrón no puede contener un separador de ruta ni '..'.",
      "token": "Token desconocido. Usa {index} o {label}.",
      "unclosed": "Llave sin cerrar en el patrón.",
      "width": "El número de dígitos tiene que ser un número."
    },
    "scope": "Páginas",
    "scopeAll": "Todas las páginas ({{count}})",
    "scopeSelected": "Seleccionadas ({{count}})",
    "start": "Exportar",
    "subfolders": "Carpetas separadas por formato",
    "title": "Exportar páginas"
  },
```

ja-JP:

```json
  "export": {
    "cancel": "キャンセル",
    "description": "ページを画像ファイルとして書き出します。",
    "formats": "形式",
    "pattern": "ファイル名パターン",
    "patternError": {
      "separator": "パターンにパス区切り文字や '..' は使えません。",
      "token": "不明なトークンです。{index} または {label} を使ってください。",
      "unclosed": "パターンの波かっこが対応していません。",
      "width": "桁数は数値で指定してください。"
    },
    "scope": "ページ",
    "scopeAll": "すべてのページ ({{count}})",
    "scopeSelected": "選択中 ({{count}})",
    "start": "書き出す",
    "subfolders": "形式ごとにフォルダーを分ける",
    "title": "ページの書き出し"
  },
```

pt-BR:

```json
  "export": {
    "cancel": "Cancelar",
    "description": "Renderiza as páginas como arquivos de imagem.",
    "formats": "Formatos",
    "pattern": "Padrão de nome",
    "patternError": {
      "separator": "O padrão não pode conter separador de caminho nem '..'.",
      "token": "Token desconhecido. Use {index} ou {label}.",
      "unclosed": "Chave sem correspondência no padrão.",
      "width": "A quantidade de dígitos precisa ser um número."
    },
    "scope": "Páginas",
    "scopeAll": "Todas as páginas ({{count}})",
    "scopeSelected": "Selecionadas ({{count}})",
    "start": "Exportar",
    "subfolders": "Pastas separadas por formato",
    "title": "Exportar páginas"
  },
```

ru-RU:

```json
  "export": {
    "cancel": "Отмена",
    "description": "Отрисовывает страницы в файлы изображений.",
    "formats": "Форматы",
    "pattern": "Шаблон имени файла",
    "patternError": {
      "separator": "Шаблон не может содержать разделитель пути или '..'.",
      "token": "Неизвестный токен. Используйте {index} или {label}.",
      "unclosed": "Непарная фигурная скобка в шаблоне.",
      "width": "Количество цифр должно быть числом."
    },
    "scope": "Страницы",
    "scopeAll": "Все страницы ({{count}})",
    "scopeSelected": "Выбранные ({{count}})",
    "start": "Экспортировать",
    "subfolders": "Отдельные папки для каждого формата",
    "title": "Экспорт страниц"
  },
```

tr-TR:

```json
  "export": {
    "cancel": "İptal",
    "description": "Sayfaları görüntü dosyaları olarak işler.",
    "formats": "Biçimler",
    "pattern": "Dosya adı deseni",
    "patternError": {
      "separator": "Desen yol ayırıcısı veya '..' içeremez.",
      "token": "Bilinmeyen belirteç. {index} veya {label} kullanın.",
      "unclosed": "Desende eşleşmeyen süslü parantez.",
      "width": "Basamak sayısı bir sayı olmalı."
    },
    "scope": "Sayfalar",
    "scopeAll": "Tüm sayfalar ({{count}})",
    "scopeSelected": "Seçili ({{count}})",
    "start": "Dışa aktar",
    "subfolders": "Her biçim için ayrı klasör",
    "title": "Sayfaları dışa aktar"
  },
```

zh-CN:

```json
  "export": {
    "cancel": "取消",
    "description": "将页面渲染为图像文件。",
    "formats": "格式",
    "pattern": "文件名模式",
    "patternError": {
      "separator": "模式不能包含路径分隔符或 '..'。",
      "token": "未知标记。请使用 {index} 或 {label}。",
      "unclosed": "模式中存在不匹配的花括号。",
      "width": "位数必须为数字。"
    },
    "scope": "页面",
    "scopeAll": "全部页面（{{count}}）",
    "scopeSelected": "所选页面（{{count}}）",
    "start": "导出",
    "subfolders": "按格式分文件夹",
    "title": "导出页面"
  },
```

zh-TW:

```json
  "export": {
    "cancel": "取消",
    "description": "將頁面算繪為影像檔案。",
    "formats": "格式",
    "pattern": "檔案名稱模式",
    "patternError": {
      "separator": "模式不能包含路徑分隔符號或 '..'。",
      "token": "未知的標記。請使用 {index} 或 {label}。",
      "unclosed": "模式中存在未配對的大括號。",
      "width": "位數必須為數字。"
    },
    "scope": "頁面",
    "scopeAll": "全部頁面（{{count}}）",
    "scopeSelected": "所選頁面（{{count}}）",
    "start": "匯出",
    "subfolders": "按格式分資料夾",
    "title": "匯出頁面"
  },
```

- [ ] **Step 4: 형식을 확인한다**

Run: `bun run check && bun run test`
Expected: 포맷 검사 통과, 테스트 통과

- [ ] **Step 5: 커밋**

```bash
git add packages/koharu/public/locales
git commit -m "i18n: strings for the batch export dialog"
```

---

## 마무리 확인

- [ ] `cargo test -p koharu-app -p koharu-rpc` 통과
- [ ] `bun run test` 통과
- [ ] `cd packages/koharu && bunx tsc --noEmit` 통과 — **테스트만으로는 부족하다.**
      `TitleBar`의 테스트는 명령을 목으로 대체하므로 `runExport(pages, 'png')`처럼
      타입이 어긋난 호출도 런타임에서는 그냥 지나간다. `Job`에 필드를 더한 것도
      마찬가지로 테스트 픽스처에서 `tsc`로만 드러난다.
- [ ] `bun run lint` 통과
- [ ] `cargo build --release && bun run ui:build` 후 앱을 띄워 실제로 확인
  - 전체 페이지를 PNG+PSD로 내보내고 `png/`, `psd/`가 생기는지
  - 진행률이 `ActivityCenter`에 "내보내는 중"으로 뜨고 중단이 먹는지
  - 브라우저(`http://127.0.0.1:47823`)에서 ZIP이 하위 폴더를 보존하는지
  - 번역 파이프라인이 도는 중에 내보내기를 누르면 폴더 선택창이 **뜨지 않고** 바로 거절되는지
