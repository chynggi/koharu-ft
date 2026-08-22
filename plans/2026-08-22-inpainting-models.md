# 인페인팅 모델 확장 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LaMa 체크포인트를 설정으로 교체할 수 있게 하고, MI-GAN을 인페인팅 모델로 추가한다. 파이썬 런타임 의존은 도입하지 않는다.

**Architecture:** `koharu-torch`(tch-rs 인트리 포크)의 TorchScript 실행 경로(`CModule`)를 연다. IOPaint의 `big-lama.pt`·`anime-manga-big-lama.pt`·`migan_traced.pt`가 모두 TorchScript로 배포되므로 아키텍처 포팅이 없다. 기존 safetensors 경로(`nn::VarStore`)와 공존시키고, `lama/processor.rs`에 이미 있는 전처리 헬퍼를 공용 모듈로 끌어올려 MI-GAN이 재사용한다.

**Tech Stack:** Rust (`koharu-torch`/libtorch 2.12.1, anyhow, image, fast_image_resize, serde, specta), TypeScript (Next.js 16, React 19, zustand, i18next). 테스트는 `cargo test`, 벤치는 criterion.

**설계 문서:** `specs/2026-08-22-inpainting-models-design.md`

**계획 위치에 대한 메모:** 기본 경로는 `docs/superpowers/plans/`지만 `docs/`는 발행용 문서 사이트다. 기존 관행(`plans/2026-08-22-batch-export.md`)을 따라 `plans/`에 둔다.

---

## 전제 확인 (이미 검증됨)

착수 전에 다시 조사할 필요가 없도록 기록해 둔다.

| 사실 | 근거 |
|---|---|
| `CModule::load_on_device` / `forward_ts` / `set_eval`이 존재한다 | `koharu-torch/src/wrappers/jit.rs:459,498,590`, `lib.rs:12`에서 재수출 |
| JIT 심볼이 실제 빌드 산출물에 바인딩되어 있다 | `target/debug/build/koharu-torch-sys-*/out/torch_api.rs:1364,1409,1500` |
| libtorch는 자동 프로비저닝된다 (사용자 설치 없음) | `koharu-runtime/src/runtime/packages/torch.rs:22,101-203` |
| LaMa TorchScript는 `forward(image, mask)` 2입력이다 | IOPaint `iopaint/model/lama.py:46-63` |
| MI-GAN TorchScript는 `forward(cat([0.5-mask, erased], 1))` 1입력이다 | IOPaint `iopaint/model/mi_gan.py:82-110` |
| 모델 설정은 `ProcessorConfig`의 `Option<T>` 필드에 저장된다 | `koharu-pipeline/src/config.rs:240-252` |

**LaMa 전처리는 이미 TorchScript와 일치한다.** `lama/processor.rs:126-127`이
`pad(image/255.0, 8)`과 `pad((mask>0) as f32, 8)`을 만드는데, 이는
`lama.py`의 `norm_img(image)` + `(mask > 0) * 1`과 같다. 따라서 LaMa는
백엔드 교체만으로 동작하고 전처리를 건드리지 않는다.

---

## File Structure

**신규**

| 파일 | 책임 |
|---|---|
| `crates/koharu-runtime/src/source/remote.rs` | 임의 URL 파일을 BLAKE3 다이제스트로 못박아 가져온다 |
| `crates/koharu-ml/src/torchscript.rs` | `CModule` 래퍼. 로드·디바이스 배치·`set_eval`·에러 문맥. 모델 지식 없음 |
| `crates/koharu-ml/src/source.rs` | `ComponentSource` 공용화. `flux2_klein/source.rs`에서 이동 |
| `crates/koharu-ml/src/inpaint_ops.rs` | `lama/processor.rs`에서 끌어올린 공용 헬퍼(bbox·crop·resize·pad·post-process) |
| `crates/koharu-ml/src/lama/backend.rs` | `Backend` 트레이트와 두 구현(safetensors / TorchScript) |
| `crates/koharu-ml/src/mi_gan/mod.rs` | MI-GAN 로드·추론 진입점 |
| `crates/koharu-ml/src/mi_gan/processor.rs` | MI-GAN 전용 512 crop·정규화·후처리 |
| `crates/koharu-ml/src/bin/mi_gan.rs` | 단일 이미지 CLI |
| `crates/koharu-ml/benches/mi_gan.rs` | LaMa 대비 지연 시간 벤치 |

**수정**

| 파일 | 변경 |
|---|---|
| `crates/koharu-runtime/src/source/mod.rs` | `remote` 등록, `RemoteFile`·`PinnedFile` 재수출 |
| `crates/koharu-runtime/src/lib.rs` | 재수출 |
| `crates/koharu-runtime/Cargo.toml` | `blake3` 추가 (워크스페이스에 이미 있음) |
| `crates/koharu-ml/src/lib.rs` | `torchscript`·`source`·`inpaint_ops`·`mi_gan` 모듈 등록, `remote_repository!` 매크로 |
| `crates/koharu-ml/src/flux2_klein/source.rs` | `ComponentSource`를 `pub use`로 대체 |
| `crates/koharu-ml/src/lama/mod.rs` | `load`가 소스·형식을 받는다 |
| `crates/koharu-ml/src/lama/processor.rs` | `Model` 직접 참조 → `Backend` 트레이트, 헬퍼는 `inpaint_ops`에서 |
| `crates/koharu-ml/src/bin/lama.rs` | 소스·형식 인자 |
| `crates/koharu-ml/Cargo.toml` | `mi_gan` bench·bin 등록 |
| `crates/koharu-pipeline/src/stages/inpainting.rs` | `LaMaConfig`·`MiGanConfig`, `Model` 변형 2개 |
| `crates/koharu-pipeline/src/stages/mod.rs` | 새 config 재수출 |
| `crates/koharu-pipeline/src/config.rs` | `InpaintingModel::LaMa(LaMaConfig)`·`MiGan(MiGanConfig)`, `ProcessorConfig` 2개 필드 |
| `crates/koharu-pipeline/src/lib.rs` | 재수출 |
| `crates/koharu-pipeline/src/bin/run.rs` | `InpaintingChoice::MiGan` |
| `packages/bridge/src/protocol.ts` | 생성된 타입 갱신 |
| `packages/koharu/components/.../인페인팅 설정 패널` | 소스·형식 선택 UI |
| `packages/koharu/public/locales/*/translation.json` | 9개 로케일 신규 문구 |

---

## Task 0: URL 체크포인트 소스

`HuggingFaceFile`은 Hugging Face만 해석한다
(`koharu-runtime/src/source/hugging_face.rs:99-143`). IOPaint 체크포인트는
GitHub 릴리스에 있으므로, **원본 저장소에서 직접 받는다.** 미러링하지 않는다 —
이 저장소는 포크이고, 상류의 `mayocream/` 재배포 관행을 따를 이유가 없다.
미러는 원본이 갱신될 때 조용히 낡고, 유지 책임만 늘린다.

필요한 것은 임의 URL을 받는 소스 하나다. `Store::file`이 `pub(crate)`이므로
`koharu-runtime` 안에 만든다.

**불변성 보장.** `HuggingFaceFile`은 커밋 해시로 파일을 못박는다. URL에는
그런 것이 없으므로 **다이제스트로 못박는다.** 워크스페이스에 `blake3`가 이미
있으므로(`Cargo.toml:74`) 새 의존성이 없다. 다운로드 후 해시를 검증하고,
어긋나면 실패한다. `Store::file`이 스테이징 파일을 통해서만 발행하므로
잘못된 파일이 캐시에 남지 않는다. HF 경로보다 오히려 강한 보장이다.

**Files:**
- Create: `crates/koharu-runtime/src/source/remote.rs`
- Modify: `crates/koharu-runtime/src/source/mod.rs`, `crates/koharu-runtime/src/lib.rs`, `crates/koharu-runtime/Cargo.toml`

- [ ] **Step 1: 의존성을 추가한다**

`crates/koharu-runtime/Cargo.toml`의 `[dependencies]`에 알파벳 순서로 넣는다.

```toml
blake3 = { workspace = true }
```

- [ ] **Step 2: 실패하는 테스트를 쓴다**

`crates/koharu-runtime/src/source/remote.rs`를 만들고 테스트만 먼저 넣는다.
네트워크 없이 검증 로직만 시험한다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_malformed_digest_is_rejected_before_the_network() {
        assert!(RemoteFile::pinned("https://example.com/model.pt", "not-a-hash")
            .validate()
            .is_err());
        assert!(
            RemoteFile::pinned("https://example.com/model.pt", &"a".repeat(64))
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn a_non_https_url_is_rejected() {
        assert!(
            RemoteFile::pinned("http://example.com/model.pt", &"a".repeat(64))
                .validate()
                .is_err()
        );
    }

    #[test]
    fn the_cache_path_is_addressed_by_the_digest() {
        let digest = "b".repeat(64);
        let file = RemoteFile::pinned(
            "https://github.com/Sanster/models/releases/download/migan/migan_traced.pt",
            &digest,
        );
        let path = file.cache_path().unwrap();

        assert!(path.ends_with(std::path::Path::new(&digest).join("migan_traced.pt")));
    }
}
```

- [ ] **Step 3: 테스트가 실패하는 것을 확인한다**

Run: `cargo test -p koharu-runtime remote`
Expected: FAIL — `cannot find type RemoteFile in this scope`

- [ ] **Step 4: 최소 구현을 쓴다**

`crates/koharu-runtime/src/source/remote.rs`의 테스트 위에 넣는다.

```rust
//! 임의 URL에 있는 파일을 다이제스트로 못박아 가져온다.
//!
//! Hugging Face에 없는 체크포인트 — IOPaint가 GitHub 릴리스로 배포하는
//! TorchScript 아카이브 같은 것 — 를 원본에서 직접 받기 위한 소스다.
//! URL에는 커밋 해시에 해당하는 것이 없으므로 BLAKE3 다이제스트가 그 역할을
//! 한다. 캐시 경로도 다이제스트로 주소를 매기므로, 상류가 같은 URL에 다른
//! 파일을 올려도 기존 캐시를 오염시키지 않는다.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use crate::{downloads::Transfer, store::Store};

/// URL 하나에 있는 불변 파일.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RemoteFile<'a> {
    url: &'a str,
    digest: &'a str,
}

impl<'a> RemoteFile<'a> {
    /// `digest`는 파일 내용의 BLAKE3 해시를 16진수 64자로 적은 것이다.
    /// 대소문자는 가리지 않되, 캐시 경로에는 소문자로 정규화해 넣는다 —
    /// 같은 파일이 표기만 달라 캐시에 두 번 들어가지 않게 하기 위해서다.
    #[must_use]
    pub const fn pinned(url: &'a str, digest: &'a str) -> Self {
        Self { url, digest }
    }

    /// 네트워크를 건드리지 않고 드러나는 실수를 거른다.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.url.starts_with("https://"),
            "model URL must be https: {}",
            self.url
        );
        ensure!(
            self.file_name().is_some(),
            "model URL has no file name: {}",
            self.url
        );
        ensure!(
            self.digest.len() == 64 && self.digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "digest must be 64 hex characters: {}",
            self.digest
        );
        Ok(())
    }

    /// URL의 마지막 경로 조각. 쿼리 문자열은 버린다.
    fn file_name(&self) -> Option<&'a str> {
        let path = self.url.split(['?', '#']).next()?;
        let name = path.rsplit('/').next()?;
        (!name.is_empty()
            && name != "."
            && name != ".."
            && name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')))
        .then_some(name)
    }

    fn cache_path(&self) -> Result<PathBuf> {
        self.validate()?;
        let name = self.file_name().context("model URL has no file name")?;
        Ok(Store::root().join("remote").join(self.digest).join(name))
    }

    #[tracing::instrument(skip_all)]
    pub async fn resolve(self) -> Result<PathBuf> {
        let target = self.cache_path()?;
        let url = self.url.to_owned();
        let digest = self.digest.to_owned();
        Store::file(target, move |stage| async move {
            Transfer::new()?.fetch(&url, &stage).await?;
            verify(&stage, &digest)
                .with_context(|| format!("{url} did not match its pinned digest"))
        })
        .await
    }
}

fn verify(path: &Path, expected: &str) -> Result<()> {
    let mut hasher = blake3::Hasher::new();
    hasher
        .update_mmap_rayon(path)
        .with_context(|| format!("failed to hash {}", path.display()))?;
    let actual = hasher.finalize().to_hex();
    ensure!(
        actual.as_str().eq_ignore_ascii_case(expected),
        "expected {expected}, got {actual}"
    );
    Ok(())
}
```

- [ ] **Step 5: 모듈을 등록한다**

`crates/koharu-runtime/src/source/mod.rs`:

```rust
mod archive;
mod hugging_face;
mod pypi;
mod remote;

pub use hugging_face::HuggingFaceFile;
pub use remote::RemoteFile;

pub(crate) use archive::extract;
pub(crate) use pypi::{Platform, wheel};
```

`crates/koharu-runtime/src/lib.rs`의 `HuggingFaceFile` 재수출 옆에
`RemoteFile`을 추가한다.

Run: `rg -n "pub use source::" crates/koharu-runtime/src/lib.rs`

- [ ] **Step 6: 두 소스를 하나로 받는 타입을 만든다**

`ComponentSource::resolve`는 지금 `HuggingFaceFile`만 받는다. 내장 기본값이
둘 중 어느 쪽일 수도 있으므로 `remote.rs` 아래, `source/mod.rs`에 넣는다.

```rust
/// 내장 기본값이 어디서 오는지. 소스가 `Builtin`일 때만 쓰인다.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PinnedFile<'a> {
    HuggingFace(HuggingFaceFile<'a>),
    Remote(RemoteFile<'a>),
}

impl PinnedFile<'_> {
    pub async fn resolve(self) -> anyhow::Result<std::path::PathBuf> {
        match self {
            Self::HuggingFace(file) => file.resolve().await,
            Self::Remote(file) => file.resolve().await,
        }
    }
}

impl<'a> From<HuggingFaceFile<'a>> for PinnedFile<'a> {
    fn from(value: HuggingFaceFile<'a>) -> Self {
        Self::HuggingFace(value)
    }
}

impl<'a> From<RemoteFile<'a>> for PinnedFile<'a> {
    fn from(value: RemoteFile<'a>) -> Self {
        Self::Remote(value)
    }
}
```

`pub use remote::RemoteFile;` 옆에 `PinnedFile`도 재수출한다.

- [ ] **Step 7: 테스트를 통과시킨다**

Run: `cargo test -p koharu-runtime`
Expected: PASS

- [ ] **Step 8: 실제 다이제스트를 구한다**

세 파일의 BLAKE3 해시를 계산한다. **자격 증명이 필요 없다 — 누구나 실행할 수
있다.** 결과는 Task 3과 Task 7의 상수에 그대로 넣는다.

```bash
cd "$(mktemp -d)"
curl -sL -O https://github.com/Sanster/models/releases/download/migan/migan_traced.pt
curl -sL -O https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt
curl -sL -O https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
b3sum migan_traced.pt anime-manga-big-lama.pt big-lama.pt
```

`b3sum`이 없으면 `cargo install b3sum`으로 넣거나, 위 `verify`를 쓰는
한 줄짜리 테스트로 대신 구한다.

IOPaint가 못박아 둔 `migan_traced.pt`의 MD5
`76eb3b1a71c400ee3290524f7a11b89c`(`iopaint/model/mi_gan.py:21`)로 받은
파일이 맞는지 교차 확인한다.

```bash
md5sum migan_traced.pt
```

Expected: `76eb3b1a71c400ee3290524f7a11b89c  migan_traced.pt`

**측정 결과 (2026-08-22, 이미 계획에 반영됨).** 자리표시자는 남아 있지 않다.

| 파일 | 크기 | BLAKE3 |
|---|---|---|
| `migan_traced.pt` | 26 MB | `fde1e5f7c6b6a48082f8eff36b9117e64b8c14ea4d1a76af508e29d357b28cbd` |
| `anime-manga-big-lama.pt` | 197 MB | `9213532a6e9990afcd0c9f3f31da82cc4c8c1ec86a13641e3ec37648d5e75f8b` |
| `big-lama.pt` | 197 MB | `1e3e5989dae88d561f1c8e8456c8fe9595739aaa9898862d004abf192c6d9e76` |

`migan_traced.pt`의 MD5는 `76eb3b1a71c400ee3290524f7a11b89c`로 IOPaint가
못박아 둔 값과 일치하는 것을 확인했다.

`big-lama.pt`(원본, 만화 파인튜닝이 아닌 쪽)는 내장 기본값으로 쓰지 않지만,
설정에서 `Url` 갈래로 지정할 때 쓰라고 기록해 둔다.

- [ ] **Step 9: 커밋한다**

```bash
git add crates/koharu-runtime
git commit -m "feat(runtime): resolve pinned files from arbitrary URLs"
```

---

## Task 1: TorchScript 로더

`CModule`을 얇게 감싸 로드 실패에 파일 경로 문맥을 붙이고, `set_eval`과
디바이스 배치를 한곳에서 보장한다. 모델 지식은 넣지 않는다.

**Files:**
- Create: `crates/koharu-ml/src/torchscript.rs`
- Modify: `crates/koharu-ml/src/lib.rs`

- [ ] **Step 1: 모듈을 등록한다**

`crates/koharu-ml/src/lib.rs`의 `mod backend;` 아래에 추가한다.

```rust
mod backend;

pub mod source;
pub mod torchscript;
```

그리고 `pub mod aot_inpainting;`로 시작하는 알파벳 순 목록은 그대로 둔다.

- [ ] **Step 2: 실패하는 테스트를 쓴다**

`crates/koharu-ml/src/torchscript.rs`를 만들고 테스트만 먼저 넣는다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_archive_file_fails_with_the_path_in_the_message() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("not-an-archive.pt");
        std::fs::write(&path, b"this is not a torchscript archive").unwrap();

        let error = TorchScript::load(&path, koharu_torch::Device::Cpu).unwrap_err();
        assert!(error.to_string().contains("not-an-archive.pt"));
    }
}
```

- [ ] **Step 3: 테스트가 실패하는 것을 확인한다**

Run: `cargo test -p koharu-ml torchscript`
Expected: FAIL — `cannot find type TorchScript in this scope`

- [ ] **Step 4: 최소 구현을 쓴다**

`crates/koharu-ml/src/torchscript.rs`의 테스트 위에 넣는다.

```rust
//! TorchScript 아카이브 실행 경로.
//!
//! `nn::VarStore`로 읽는 safetensors state_dict와 달리, TorchScript
//! 아카이브는 아키텍처를 파일 안에 담고 있으므로 Rust 쪽에 대응하는 모듈
//! 정의가 필요하지 않다. IOPaint가 `load_jit_model`로 읽는 체크포인트가
//! 모두 이 형식이다.

use std::path::Path;

use anyhow::{Context as _, Result};
use koharu_torch::{CModule, Device, Tensor};

#[derive(Debug)]
pub struct TorchScript {
    module: CModule,
}

impl TorchScript {
    /// 아카이브를 주어진 디바이스에 올리고 평가 모드로 고정한다.
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let path = path.as_ref();
        let mut module = CModule::load_on_device(path, device)
            .with_context(|| format!("failed to load TorchScript archive {}", path.display()))?;
        module.set_eval();
        Ok(Self { module })
    }

    /// 트레이싱 시점과 입력 개수·형상이 다르면 실패한다.
    pub fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        self.module
            .forward_ts(inputs)
            .context("TorchScript forward failed")
    }
}
```

- [ ] **Step 5: 테스트가 통과하는 것을 확인한다**

Run: `cargo test -p koharu-ml torchscript`
Expected: PASS

- [ ] **Step 6: 실제 아카이브로 스모크 테스트를 한다**

이 단계는 네트워크와 GPU가 필요하므로 `#[ignore]`를 붙인다.
`crates/koharu-ml/src/torchscript.rs`의 `mod tests`에 추가한다.

```rust
    /// 실제 체크포인트가 필요하므로 기본 실행에서 제외한다.
    /// `KOHARU_BIG_LAMA_TS`에 `big-lama.pt` 경로를 주고 실행한다.
    #[test]
    #[ignore = "requires a local big-lama.pt"]
    fn big_lama_accepts_an_image_and_a_mask() {
        let path = std::env::var("KOHARU_BIG_LAMA_TS").unwrap();
        let device = koharu_torch::Device::Cpu;
        let model = TorchScript::load(&path, device).unwrap();

        let image = koharu_torch::Tensor::zeros([1, 3, 512, 512], (koharu_torch::Kind::Float, device));
        let mask = koharu_torch::Tensor::zeros([1, 1, 512, 512], (koharu_torch::Kind::Float, device));
        let output = model.forward(&[&image, &mask]).unwrap();

        assert_eq!(output.size(), vec![1, 3, 512, 512]);
    }
```

Run: `KOHARU_BIG_LAMA_TS=/path/to/big-lama.pt cargo test -p koharu-ml torchscript -- --ignored`
Expected: PASS

- [ ] **Step 7: 커밋한다**

```bash
git add crates/koharu-ml/src/torchscript.rs crates/koharu-ml/src/lib.rs
git commit -m "feat(ml): add a TorchScript archive loader"
```

---

## Task 2: ComponentSource 공용화

`ComponentSource`는 이름과 달리 FLUX와 무관한 세 갈래(`Builtin` /
`LocalFile` / `HuggingFace`)일 뿐이다. LaMa와 MI-GAN이 같은 것을 필요로 하므로
크레이트 공용으로 올린다. 기존 테스트를 그대로 옮겨 회귀를 막는다.

**Files:**
- Create: `crates/koharu-ml/src/source.rs`
- Modify: `crates/koharu-ml/src/flux2_klein/source.rs`

- [ ] **Step 1: 파일을 이동한다**

`crates/koharu-ml/src/flux2_klein/source.rs`에서 `ComponentSource`의
정의·`impl`·관련 테스트 3개(`a_relative_local_path_is_rejected`,
`a_local_file_must_exist`, `a_hugging_face_override_is_checked_before_the_network`)를
`crates/koharu-ml/src/source.rs`로 통째로 옮긴다. 본문은 한 글자도 바꾸지
않는다 — 단, `pub(super) async fn resolve`의 가시성을 `pub`으로 올리고,
내장 기본값의 타입을 Task 0의 `PinnedFile`로 넓힌다. `flux2_klein` 밖에서
호출해야 하고, LaMa TorchScript와 MI-GAN의 내장값이 URL이기 때문이다.

```rust
    pub async fn resolve(&self, builtin: PinnedFile<'_>) -> Result<PathBuf> {
```

`Self::Builtin` 팔은 `builtin.resolve().await`로 그대로 동작한다. FLUX의
호출부는 `HuggingFaceFile`을 넘기는데 `From`이 있으므로
`WEIGHTS.into()` 한 번만 붙이면 된다.

- [ ] **Step 1b: URL 갈래를 추가한다**

원본 저장소에서 직접 받는 것이 이 태스크의 요지이므로, 사용자도 임의 URL을
지정할 수 있어야 한다. `ComponentSource`에 넣는다.

```rust
    /// 임의 URL. 다이제스트로 못박는다 — 없으면 캐시가 오염될 수 있으므로
    /// 필수다. 검증은 `RemoteFile::validate`에 위임하므로 형식 규칙이
    /// 한 곳에만 있다.
    Url { url: String, digest: String },
```

`resolve`에 팔을 넣는다.

```rust
            Self::Url { url, digest } => {
                self.validate()?;
                koharu_runtime::RemoteFile::pinned(url, digest).resolve().await
            }
```

`validate`에 팔을 넣는다.

```rust
            Self::Url { url, digest } => {
                koharu_runtime::RemoteFile::pinned(url, digest).validate()
            }
```

테스트를 추가한다.

```rust
    #[test]
    fn a_url_override_requires_a_digest() {
        let source = ComponentSource::Url {
            url: "https://example.com/model.pt".to_owned(),
            digest: "short".to_owned(),
        };
        assert!(source.validate().unwrap_err().to_string().contains("64 hex"));

        ComponentSource::Url {
            url: "https://example.com/model.pt".to_owned(),
            digest: "a".repeat(64),
        }
        .validate()
        .unwrap();
    }
```

`ComponentSourceConfig`(`koharu-pipeline/src/stages/inpainting.rs:38-72`)에도
같은 갈래와 `From` 팔을 추가한다.

```rust
    /// 임의 URL. `digest`는 BLAKE3 16진수 64자다 (대소문자 무관).
    Url { url: String, digest: String },
```

- [ ] **Step 2: FLUX 쪽을 재수출로 바꾼다**

`crates/koharu-ml/src/flux2_klein/source.rs`의 맨 위에서 이동한 정의를 지우고
넣는다. `Flux2KleinSource`와 그 테스트(`the_default_source_uses_every_builtin_repository`)는
남긴다.

```rust
use std::path::PathBuf;

use anyhow::{Context as _, Result};

pub use crate::source::ComponentSource;
```

`koharu_runtime::HuggingFaceFile`과 `ensure` import는 이동한 코드와 함께
빠지므로 지운다.

- [ ] **Step 3: URL 내장값을 위한 매크로를 만든다**

`crates/koharu-ml/src/lib.rs`의 `model_repository!` 정의 바로 아래에 넣는다.
Hugging Face 저장소가 아니라 URL이 내장 기본값인 모델을 위한 것이다.

```rust
macro_rules! remote_repository {
    ($($name:ident = $url:literal @ $digest:literal),+ $(,)?) => {
        $(
            const $name: koharu_runtime::RemoteFile<'static> =
                koharu_runtime::RemoteFile::pinned($url, $digest);
        )+
    };
}
```

- [ ] **Step 4: 전체 테스트를 돌린다**

Run: `cargo test -p koharu-ml`
Expected: PASS — 이동한 3개 테스트, 새 `a_url_override_requires_a_digest`,
`the_default_source_uses_every_builtin_repository`가 모두 통과

- [ ] **Step 5: 커밋한다**

```bash
git add crates/koharu-ml/src/source.rs crates/koharu-ml/src/flux2_klein/source.rs crates/koharu-ml/src/lib.rs crates/koharu-pipeline/src/stages/inpainting.rs
git commit -m "refactor(ml): lift ComponentSource out of flux2_klein and add a URL source"
```

---

## Task 3: LaMa 백엔드 추상화

`lama/processor.rs`가 요구하는 모델 인터페이스는 정확히 하나다 —
`forward(&self, image: &Tensor, mask: &Tensor) -> Tensor`
(`processor.rs:127`). 이 지점을 트레이트로 만들어 safetensors와 TorchScript
두 구현을 꽂는다. 전처리는 한 줄도 바뀌지 않는다.

**Files:**
- Create: `crates/koharu-ml/src/lama/backend.rs`
- Modify: `crates/koharu-ml/src/lama/mod.rs`, `crates/koharu-ml/src/lama/processor.rs`

- [ ] **Step 1: 트레이트와 두 구현을 쓴다**

`crates/koharu-ml/src/lama/backend.rs`를 만든다.

```rust
//! LaMa 가중치의 두 배포 형식.
//!
//! safetensors는 state_dict라 `FFCResNetGenerator`가 Rust에 있어야 하고,
//! TorchScript 아카이브는 아키텍처를 파일 안에 담는다. 전처리는 두 형식이
//! 동일하므로 `processor`는 이 트레이트만 본다.

use anyhow::Result;
use koharu_torch::Tensor;

use crate::torchscript::TorchScript;

use super::model::Model;

pub(super) trait Backend: std::fmt::Debug + Send {
    /// `image`는 [1,3,H,W] 0..1, `mask`는 [1,1,H,W] 0 또는 1.
    fn forward(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor>;
}

impl Backend for Model {
    fn forward(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor> {
        Ok(Model::forward(self, image, mask))
    }
}

impl Backend for TorchScript {
    fn forward(&self, image: &Tensor, mask: &Tensor) -> Result<Tensor> {
        TorchScript::forward(self, &[image, mask])
    }
}
```

- [ ] **Step 2: processor를 트레이트에 맞춘다**

`crates/koharu-ml/src/lama/processor.rs`에서 `Model` import를 바꾼다.

```rust
use super::{
    backend::Backend,
    config::{HDStrategy, InpaintRequest},
};
```

`call`·`pad_forward`의 `model: &Model` 매개변수를 `model: &dyn Backend`로
바꾼다. `pad_forward`의 `model.forward(...)` 호출은 이제 `Result`를 내므로
`?`를 붙인다.

```rust
        let output = model
            .forward(&model_image, &model_mask)?
            .narrow(2, 0, i64::from(height))
            .narrow(3, 0, i64::from(width))
            .clamp(0.0, 1.0)
            * 255.0;
```

- [ ] **Step 3: 형식을 설정에 넣는다**

`crates/koharu-ml/src/lama/config.rs`의 `HDStrategy` 위에 추가한다.

```rust
/// LaMa 가중치 파일의 형식. 두 형식은 서로 대체하지 않으므로 명시적으로 고른다.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WeightsFormat {
    /// `nn::VarStore`로 읽는 state_dict. `mayocream/lama-manga`가 이것이다.
    #[default]
    SafeTensors,
    /// `CModule`로 읽는 TorchScript 아카이브. IOPaint 체크포인트가 이것이다.
    TorchScript,
}
```

- [ ] **Step 4: 진입점을 바꾼다**

`crates/koharu-ml/src/lama/mod.rs`를 다음으로 바꾼다. `LaMa`의 필드에서
`model: Model`이 `backend: Box<dyn Backend>`가 된다.

```rust
//! LaMa inference with IOPaint-compatible orchestration.

mod backend;
mod config;
mod model;
mod processor;

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_torch::Device;

use crate::{backend::TryIntoDevice, source::ComponentSource, torchscript::TorchScript};

pub use self::config::{HDStrategy, InpaintRequest, WeightsFormat};
use self::{
    backend::Backend, config::FFCResNetGeneratorConfig, model::Model, processor::InpaintModel,
};

model_repository!("mayocream/lama-manga" @ "f91c85b26913b3e83f9877867b4c336da3675238" {
    WEIGHTS = "lama-manga.safetensors"
});

// TorchScript 기본값은 상류 원본에서 직접 받는다. 만화 파인튜닝이 이
// 파이프라인의 대상에 더 맞으므로 `big-lama.pt`가 아니라 이쪽을 기본으로
// 삼는다. 원본을 쓰려면 설정에서 URL을 바꾼다.
remote_repository! {
    TORCHSCRIPT_WEIGHTS =
        "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt"
        @ "9213532a6e9990afcd0c9f3f31da82cc4c8c1ec86a13641e3ec37648d5e75f8b",
}

#[derive(Debug)]
pub struct LaMa {
    backend: Box<dyn Backend>,
    processor: InpaintModel,
}

impl LaMa {
    pub async fn load(
        device: crate::Device,
        source: &ComponentSource,
        format: WeightsFormat,
    ) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let backend: Box<dyn Backend> = match format {
            WeightsFormat::SafeTensors => {
                let path = source
                    .resolve(WEIGHTS.into())
                    .await
                    .context("failed to resolve LaMa weights")?;
                let mut model = Model::new(&FFCResNetGeneratorConfig::default(), device);
                model
                    .load(&path)
                    .context("failed to load LaMa safetensors")?;
                Box::new(model)
            }
            WeightsFormat::TorchScript => {
                let path = source
                    .resolve(TORCHSCRIPT_WEIGHTS.into())
                    .await
                    .context("failed to resolve LaMa weights")?;
                Box::new(TorchScript::load(&path, device)?)
            }
        };
        Ok(Self {
            backend,
            processor: InpaintModel::new(device),
        })
    }

    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &GrayImage,
        config: &InpaintRequest,
    ) -> Result<RgbImage> {
        koharu_torch::no_grad(|| {
            self.processor
                .call(self.backend.as_ref(), image, mask, config)
        })
    }
}
```

다이제스트는 Task 0 Step 8에서 측정한 실제 값이다.

- [ ] **Step 5: CLI를 맞춘다**

`crates/koharu-ml/src/bin/lama.rs`의 `LaMa::load(device)` 호출을 바꾼다.
`clap` 인자 두 개를 추가한다.

```rust
    /// TorchScript 아카이브로 읽는다. 기본은 safetensors.
    #[arg(long)]
    torchscript: bool,

    /// 가중치 파일 경로. 생략하면 내장 저장소를 쓴다.
    #[arg(long)]
    weights: Option<std::path::PathBuf>,
```

호출부는 다음과 같다.

```rust
    let source = match args.weights {
        Some(path) => koharu_ml::source::ComponentSource::LocalFile(path),
        None => koharu_ml::source::ComponentSource::Builtin,
    };
    let format = if args.torchscript {
        koharu_ml::lama::WeightsFormat::TorchScript
    } else {
        koharu_ml::lama::WeightsFormat::SafeTensors
    };
    let model = LaMa::load(device, &source, format).await?;
```

- [ ] **Step 6: 벤치를 맞춘다**

`crates/koharu-ml/benches/lama.rs`의 `LaMa::load(device)` 호출을 바꾼다.

```rust
    let model = LaMa::load(
        device,
        &koharu_ml::source::ComponentSource::Builtin,
        koharu_ml::lama::WeightsFormat::SafeTensors,
    )
```

- [ ] **Step 7: 빌드와 테스트를 확인한다**

Run: `cargo test -p koharu-ml && cargo build -p koharu-ml --benches --bins`
Expected: PASS — 컴파일 에러 없음. 이 시점에 `koharu-pipeline`은 아직
깨져 있다(Task 4에서 고친다). 그래서 `-p koharu-ml`로 한정한다.

- [ ] **Step 8: safetensors 경로 무회귀를 확인한다**

Run: `cargo run -p koharu-ml --bin lama --release -- --image crates/koharu-ml/benches/fixtures/inpaint/image_4k.jpg --mask crates/koharu-ml/benches/fixtures/inpaint/mask_4k.png`
Expected: 이전과 같은 출력 이미지가 생성된다.

- [ ] **Step 9: TorchScript 경로를 확인한다**

Run: `cargo run -p koharu-ml --bin lama --release -- --torchscript --weights /path/to/anime-manga-big-lama.pt --image crates/koharu-ml/benches/fixtures/inpaint/image_4k.jpg --mask crates/koharu-ml/benches/fixtures/inpaint/mask_4k.png`
Expected: 마스크 영역이 지워진 이미지가 생성된다.

- [ ] **Step 10: 커밋한다**

```bash
git add crates/koharu-ml/src/lama crates/koharu-ml/src/bin/lama.rs crates/koharu-ml/benches/lama.rs
git commit -m "feat(ml): load LaMa from either safetensors or TorchScript"
```

---

## Task 4: LaMa 설정 배선

`ProcessorConfig`가 모델별 설정을 `Option<T>`로 들고, 파일에는 선택된 모델
이름만 저장된다(`config.rs:240-252`). 이 구조 덕분에 기존 `koharu.toml`은
`lama` 항목이 없으므로 `unwrap_or_default()`로 자동 하위 호환된다.

**Files:**
- Modify: `crates/koharu-pipeline/src/stages/inpainting.rs`, `crates/koharu-pipeline/src/stages/mod.rs`, `crates/koharu-pipeline/src/config.rs`, `crates/koharu-pipeline/src/lib.rs`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`crates/koharu-pipeline/src/config.rs`의 `mod tests`에 추가한다.

```rust
    #[test]
    fn an_existing_file_without_a_lama_section_keeps_the_builtin_checkpoint() {
        let config: PipelineConfig = toml::from_str(
            r#"
            [detection]
            model = "koharu-layout-rfdetr-seg-2xl"
            [ocr]
            model = "paddleocr-vl-1.6"
            [inpainting]
            model = "lama"
            "#,
        )
        .unwrap();

        let InpaintingModel::LaMa(lama) = config.inpainting().unwrap() else {
            panic!("expected LaMa");
        };
        assert_eq!(lama.source, ComponentSourceConfig::Builtin);
        assert_eq!(lama.format, WeightsFormatConfig::SafeTensors);
    }

    #[test]
    fn a_torchscript_lama_selection_round_trips() {
        let config = PipelineConfig {
            inpainting: InpaintingModel::LaMa(LaMaConfig {
                source: ComponentSourceConfig::Builtin,
                format: WeightsFormatConfig::TorchScript,
            }),
            ..PipelineConfig::default()
        };

        let text = toml::to_string(&config).unwrap();
        let parsed: PipelineConfig = toml::from_str(&text).unwrap();

        assert!(matches!(
            parsed.inpainting(),
            Ok(InpaintingModel::LaMa(config))
                if config.format == WeightsFormatConfig::TorchScript
        ));
    }
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

Run: `cargo test -p koharu-pipeline lama`
Expected: FAIL — `LaMaConfig` 미정의

- [ ] **Step 3: 설정 타입을 만든다**

`crates/koharu-pipeline/src/stages/inpainting.rs`의 `ComponentSourceConfig`
정의 아래에 추가한다. `ComponentSourceConfig`는 이미 존재하므로 다시 만들지
않는다.

```rust
/// LaMa 가중치 파일의 형식. `koharu_ml::lama::WeightsFormat`의 설정 표현이다.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum WeightsFormatConfig {
    #[default]
    SafeTensors,
    TorchScript,
}

impl From<WeightsFormatConfig> for koharu_ml::lama::WeightsFormat {
    fn from(value: WeightsFormatConfig) -> Self {
        match value {
            WeightsFormatConfig::SafeTensors => Self::SafeTensors,
            WeightsFormatConfig::TorchScript => Self::TorchScript,
        }
    }
}

/// LaMa 체크포인트 선택. 기본값은 `mayocream/lama-manga` safetensors로,
/// 이 필드가 도입되기 전의 동작과 같다.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct LaMaConfig {
    pub source: ComponentSourceConfig,
    pub format: WeightsFormatConfig,
}

impl LaMaConfig {
    pub fn validate(&self) -> Result<()> {
        koharu_ml::source::ComponentSource::from(self.source.clone())
            .validate()
            .context("LaMa weights")
    }
}
```

- [ ] **Step 4: 로드 분기를 고친다**

`crates/koharu-pipeline/src/stages/inpainting.rs`의 `Model::load`에서
`InpaintingModel::LaMa {}` 팔을 바꾼다.

```rust
            InpaintingModel::LaMa(config) => Ok(Self::LaMa(Arc::new(Mutex::new(
                LaMa::load(
                    device,
                    &koharu_ml::source::ComponentSource::from(config.source.clone()),
                    config.format.into(),
                )
                .await?,
            )))),
```

`Processor::new`의 검증 팔도 바꾼다.

```rust
            InpaintingModel::LaMa(settings) => settings.validate()?,
            InpaintingModel::AotInpainting {} => {}
```

- [ ] **Step 5: config.rs를 고친다**

`crates/koharu-pipeline/src/config.rs`에서 네 곳을 바꾼다.

`InpaintingModel` 정의:

```rust
    #[serde(rename = "lama")]
    LaMa(LaMaConfig),
```

`ProcessorConfig`에 필드 추가:

```rust
    #[serde(rename = "lama")]
    pub lama: Option<LaMaConfig>,
```

`Serialize`의 `match &self.inpainting` 두 곳:

```rust
            InpaintingModel::LaMa(_) => "lama",
```

```rust
            InpaintingModel::LaMa(config) => {
                processor.lama.get_or_insert_with(|| config.clone());
            }
            InpaintingModel::AotInpainting {} => {}
```

`Deserialize`:

```rust
            "lama" => InpaintingModel::LaMa(file.processor.lama.clone().unwrap_or_default()),
```

`Default`:

```rust
            inpainting: InpaintingModel::LaMa(LaMaConfig::default()),
```

`inpainting()`:

```rust
            InpaintingModel::LaMa(config) => Ok(InpaintingModel::LaMa(
                self.processor
                    .lama
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
```

파일 상단 import에 추가한다.

```rust
use crate::stages::{
    Flux2KleinConfig, KoharuLayoutRFDetrSeg2XLConfig, LaMaConfig, RoremMixedConfig,
    WeightsFormatConfig,
};
```

- [ ] **Step 6: 재수출한다**

`crates/koharu-pipeline/src/stages/mod.rs:13`:

```rust
pub use inpainting::{
    ComponentSourceConfig, Flux2KleinConfig, LaMaConfig, RoremMixedConfig, WeightsFormatConfig,
};
```

`crates/koharu-pipeline/src/lib.rs:21`의 재수출 목록에 `LaMaConfig`와
`WeightsFormatConfig`를 알파벳 순서로 추가한다.

- [ ] **Step 7: 나머지 컴파일 에러를 따라간다**

Run: `cargo build --workspace`
Expected: `InpaintingModel::LaMa {}`를 쓰던 모든 지점이 에러로 드러난다 —
`config.rs:297,364`, `stages/inpainting.rs:1261`,
`bin/run.rs:118`. 각각 `InpaintingModel::LaMa(_)` 또는
`InpaintingModel::LaMa(LaMaConfig::default())`로 바꾼다. 컴파일러가 전부
짚어주므로 누락은 없다.

- [ ] **Step 8: 테스트를 통과시킨다**

Run: `cargo test -p koharu-pipeline`
Expected: PASS — Step 1의 두 테스트 포함 전체 통과

- [ ] **Step 9: 커밋한다**

```bash
git add crates/koharu-pipeline
git commit -m "feat(pipeline): make the LaMa checkpoint configurable"
```

---

## Task 5: LaMa 설정 UI

**Files:**
- Modify: `packages/bridge/src/protocol.ts`, 인페인팅 설정 패널, `packages/koharu/public/locales/*/translation.json`

- [ ] **Step 1: 생성된 타입을 갱신한다**

`specta`가 `LaMaConfig`·`WeightsFormatConfig`를 내보내므로 바인딩을 다시
만든다. 기존 생성 명령을 쓴다.

Run: `cargo test -p koharu-app export_bindings` (또는 저장소의 바인딩 생성 명령)
Expected: `packages/bridge/src/protocol.ts`에 `LaMaConfig`가 나타난다

- [ ] **Step 2: 패널을 찾는다**

Run: `rg -l "rorem-mixed|RoremMixed" packages/koharu/components`
Expected: FLUX·RORem 설정을 그리는 컴포넌트 경로가 나온다. LaMa 패널을 그
옆에 같은 모양으로 붙인다.

- [ ] **Step 3: 형식 셀렉트와 소스 입력을 추가한다**

FLUX가 이미 `ComponentSourceConfig`를 편집하는 UI를 갖고 있다면 그 컴포넌트를
재사용한다. 없으면 세 갈래를 그리는 최소 UI를 만든다 — 라디오(내장 / 로컬
파일 / Hugging Face) + 갈래별 입력.

형식 셀렉트는 두 항목이다.

```tsx
<select
  value={config.format}
  onChange={(event) =>
    onChange({ ...config, format: event.target.value as WeightsFormatConfig })
  }
>
  <option value="safe_tensors">{t('inpainting.lama.format.safetensors')}</option>
  <option value="torch_script">{t('inpainting.lama.format.torchscript')}</option>
</select>
```

- [ ] **Step 4: 9개 로케일에 문구를 넣는다**

Run: `ls packages/koharu/public/locales`
각 `translation.json`에 `inpainting.lama.format.safetensors`,
`inpainting.lama.format.torchscript`, `inpainting.lama.source.*` 키를 추가한다.
영어를 먼저 쓰고 나머지는 번역한다.

- [ ] **Step 5: 확인한다**

Run: `bun run dev` 후 설정에서 LaMa 형식을 TorchScript로 바꾸고 페이지 하나를
인페인팅한다.
Expected: `anime-manga-big-lama.pt`가 내려받아지고 말풍선이 지워진다.

- [ ] **Step 6: 커밋한다**

```bash
git add packages
git commit -m "feat(ui): let the LaMa checkpoint be chosen"
```

---

## Task 6: 공용 전처리 헬퍼 추출

MI-GAN의 orchestration(`mi_gan.py:41-80`)은 `boxes_from_mask` →
`_crop_box` → `resize_max_size` → `_pad_forward` 순서로, Koharu의
`lama/processor.rs`에 **이미 전부 있는** 헬퍼들이다. 복제하지 않고 끌어올린다.

**Files:**
- Create: `crates/koharu-ml/src/inpaint_ops.rs`
- Modify: `crates/koharu-ml/src/lama/processor.rs`, `crates/koharu-ml/src/lib.rs`

- [ ] **Step 1: 모듈을 등록한다**

`crates/koharu-ml/src/lib.rs`:

```rust
pub(crate) mod inpaint_ops;
```

- [ ] **Step 2: 함수를 이동한다**

`crates/koharu-ml/src/lama/processor.rs`에서 다음 자유 함수를
`crates/koharu-ml/src/inpaint_ops.rs`로 옮기고 `pub(crate)`로 만든다. 본문은
바꾸지 않는다.

- `resize_dimensions`
- `resize_rgb`
- `resize_gray`
- `boxes_from_mask`
- `crop_box`
- `post_process`
- `pad_img_to_modulo`
- `ceil_modulo`
- `symmetric_indices`

`post_process`의 에러 문맥은 LaMa를 언급하므로 일반화한다.

```rust
    RgbImage::from_raw(width, height, rgb).context("failed to convert output tensor to RGB image")
```

이 함수들에 붙어 있던 테스트도 함께 옮긴다.

- [ ] **Step 3: processor를 import로 바꾼다**

`crates/koharu-ml/src/lama/processor.rs` 상단에 넣는다.

```rust
use crate::inpaint_ops::{
    boxes_from_mask, crop_box, pad_img_to_modulo, post_process, resize_dimensions, resize_gray,
    resize_rgb,
};
```

이제 쓰이지 않는 `fast_image_resize`·`imageproc::contours` import를 지운다.

- [ ] **Step 4: 무회귀를 확인한다**

Run: `cargo test -p koharu-ml`
Expected: PASS — 옮긴 테스트 포함 전체 통과

Run: `cargo run -p koharu-ml --bin lama --release -- --image crates/koharu-ml/benches/fixtures/inpaint/image_4k.jpg --mask crates/koharu-ml/benches/fixtures/inpaint/mask_4k.png`
Expected: Task 3 Step 8과 같은 출력

- [ ] **Step 5: 커밋한다**

```bash
git add crates/koharu-ml/src/inpaint_ops.rs crates/koharu-ml/src/lama/processor.rs crates/koharu-ml/src/lib.rs
git commit -m "refactor(ml): share the inpainting preprocessing helpers"
```

---

## Task 7: MI-GAN

LaMa와 다른 점은 **입력 규약**뿐이다. 1입력 4채널이고, 정규화가 [-1,1]이며,
마스크 임계값이 120이고, 항상 512 정사각으로 맞춘다
(`mi_gan.py:25-27,82-110`).

**Files:**
- Create: `crates/koharu-ml/src/mi_gan/mod.rs`, `crates/koharu-ml/src/mi_gan/processor.rs`
- Modify: `crates/koharu-ml/src/lib.rs`

- [ ] **Step 1: 모듈을 등록한다**

`crates/koharu-ml/src/lib.rs`의 알파벳 순 목록에서 `manga_ocr` 위에 넣는다.

```rust
pub mod mi_gan;
```

- [ ] **Step 2: 진입점을 쓴다**

`crates/koharu-ml/src/mi_gan/mod.rs`:

```rust
//! MI-GAN inference through the TorchScript archive IOPaint distributes.
//!
//! Original preprocessing and inference:
//! https://github.com/Sanster/IOPaint/blob/main/iopaint/model/mi_gan.py

mod processor;

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_torch::Device;

use crate::{
    backend::TryIntoDevice, lama::InpaintRequest, source::ComponentSource,
    torchscript::TorchScript,
};

use self::processor::Processor;

// 상류 원본에서 직접 받는다. 미러링하지 않는다.
remote_repository! {
    WEIGHTS = "https://github.com/Sanster/models/releases/download/migan/migan_traced.pt"
        @ "fde1e5f7c6b6a48082f8eff36b9117e64b8c14ea4d1a76af508e29d357b28cbd",
}

#[derive(Debug)]
pub struct MiGan {
    model: TorchScript,
    processor: Processor,
}

impl MiGan {
    pub async fn load(device: crate::Device, source: &ComponentSource) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let path = source
            .resolve(WEIGHTS.into())
            .await
            .context("failed to resolve MI-GAN weights")?;
        Ok(Self {
            model: TorchScript::load(&path, device)?,
            processor: Processor::new(device),
        })
    }

    pub fn inference(
        &self,
        image: &DynamicImage,
        mask: &GrayImage,
        config: &InpaintRequest,
    ) -> Result<RgbImage> {
        koharu_torch::no_grad(|| self.processor.call(&self.model, image, mask, config))
    }
}
```

다이제스트는 Task 0 Step 8에서 측정한 실제 값이다.

- [ ] **Step 3: 프로세서를 쓴다**

`crates/koharu-ml/src/mi_gan/processor.rs`:

```rust
//! MI-GAN은 512 정사각 입력만 받는다. 큰 이미지는 마스크 바운딩 박스로
//! 잘라 512로 줄여 추론한 뒤 원래 크기로 되돌려 마스크 영역만 붙인다.

use anyhow::{Result, ensure};
use image::{DynamicImage, GrayImage, RgbImage};
use koharu_torch::{Device, Kind, Tensor};

use crate::{
    inpaint_ops::{
        boxes_from_mask, crop_box, pad_img_to_modulo, post_process, resize_dimensions, resize_gray,
        resize_rgb,
    },
    lama::InpaintRequest,
    torchscript::TorchScript,
};

/// `mi_gan.py`의 `min_size` / `pad_mod`.
const SIZE: u32 = 512;
/// `mi_gan.py`의 `config.hd_strategy_crop_margin = 128`.
const CROP_MARGIN: u32 = 128;
/// `mi_gan.py`의 `(mask > 120) * 255`.
const MASK_THRESHOLD: f64 = 120.0;

#[derive(Debug)]
pub(super) struct Processor {
    device: Device,
}

impl Processor {
    pub(super) fn new(device: Device) -> Self {
        Self { device }
    }

    pub(super) fn call(
        &self,
        model: &TorchScript,
        image: &DynamicImage,
        mask: &GrayImage,
        _config: &InpaintRequest,
    ) -> Result<RgbImage> {
        let image = image.to_rgb8();
        ensure!(
            image.dimensions() == mask.dimensions(),
            "image and mask dimensions differ: image={:?}, mask={:?}",
            image.dimensions(),
            mask.dimensions()
        );
        ensure!(
            image.width() > 0 && image.height() > 0,
            "image dimensions must be non-zero"
        );

        if image.width() == SIZE && image.height() == SIZE {
            return self.forward(model, &image, mask);
        }

        let mut result = image.clone();
        for bounding_box in boxes_from_mask(mask) {
            let [left, top, right, bottom] =
                crop_box(image.width(), image.height(), bounding_box, CROP_MARGIN);
            let (width, height) = (right - left, bottom - top);
            let crop_image =
                image::imageops::crop_imm(&image, left, top, width, height).to_image();
            let crop_mask = image::imageops::crop_imm(mask, left, top, width, height).to_image();

            let (small_width, small_height) = resize_dimensions(width, height, SIZE);
            let small_image = resize_rgb(&crop_image, small_width, small_height)?;
            let small_mask = resize_gray(&crop_mask, small_width, small_height)?;

            let inpainted = self.forward(model, &small_image, &small_mask)?;
            let mut restored = resize_rgb(&inpainted, width, height)?;

            // 마스크 밖은 원본 픽셀을 되돌린다. 리사이즈 왕복이 선화를
            // 뭉개기 때문이다. `mi_gan.py`의 `original_pixel_indices`.
            for (index, value) in crop_mask.as_raw().iter().enumerate() {
                if *value < 127 {
                    let offset = index * 3;
                    restored.as_mut()[offset..offset + 3]
                        .copy_from_slice(&crop_image.as_raw()[offset..offset + 3]);
                }
            }

            image::imageops::replace(&mut result, &restored, left.into(), top.into());
        }
        Ok(result)
    }

    /// `mi_gan.py`의 `forward`. 입력은 `cat([0.5 - mask, image * (1 - mask)], 1)`
    /// 4채널이고, 이미지는 [-1,1], 마스크는 0 또는 1이다.
    fn forward(&self, model: &TorchScript, image: &RgbImage, mask: &GrayImage) -> Result<RgbImage> {
        let width = image.width();
        let height = image.height();

        let image_tensor = Tensor::from_slice(image.as_raw())
            .view([i64::from(height), i64::from(width), 3])
            .to_device(self.device)
            .permute([2, 0, 1])
            .unsqueeze(0)
            .contiguous()
            .to_kind(Kind::Float)
            / 255.0;
        let mask_tensor = Tensor::from_slice(mask.as_raw())
            .view([i64::from(height), i64::from(width)])
            .to_device(self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .contiguous()
            .to_kind(Kind::Float)
            .gt(MASK_THRESHOLD)
            .to_kind(Kind::Float);

        let image_tensor = pad_img_to_modulo(image_tensor, SIZE);
        let mask_tensor = pad_img_to_modulo(mask_tensor, SIZE);

        let normalized = &image_tensor * 2.0 - 1.0;
        let erased = &normalized * (mask_tensor.ones_like() - &mask_tensor);
        let input = Tensor::cat(&[&mask_tensor * -1.0 + 0.5, erased], 1);

        let output = model.forward(&[&input])?;
        let output = (output * 127.5 + 127.5)
            .round()
            .clamp(0.0, 255.0)
            .narrow(2, 0, i64::from(height))
            .narrow(3, 0, i64::from(width))
            .to_kind(Kind::Uint8);

        post_process(&output, width, height)
    }
}
```

- [ ] **Step 4: CLI를 쓴다**

`crates/koharu-ml/src/bin/mi_gan.rs`를 만든다.
`crates/koharu-ml/src/bin/lama.rs`를 열어 그 구조를 그대로 따르되
`LaMa::load(device, &source, format)`를 `MiGan::load(device, &source)`로,
`use koharu_ml::lama::LaMa`를 `use koharu_ml::mi_gan::MiGan`으로 바꾼다.
`--torchscript` 인자는 MI-GAN에 형식 선택이 없으므로 넣지 않는다.

- [ ] **Step 5: 빌드하고 눈으로 확인한다**

`crates/koharu-ml/Cargo.toml`에는 `[[bin]]` 항목이 하나도 없다 — `src/bin/`이
자동 탐색되므로 등록 작업은 없다. 벤치는 `[[bench]]`가 필요하다(Task 10).

Run: `cargo run -p koharu-ml --bin mi_gan --release -- --image crates/koharu-ml/benches/fixtures/inpaint/image_4k.jpg --mask crates/koharu-ml/benches/fixtures/inpaint/mask_4k.png --output /tmp/migan.png`
Expected: 마스크 영역이 지워진 이미지가 생성되고, 마스크 밖 픽셀은 원본과
동일하다.

- [ ] **Step 6: 커밋한다**

```bash
git add crates/koharu-ml/src/mi_gan crates/koharu-ml/src/bin/mi_gan.rs crates/koharu-ml/src/lib.rs crates/koharu-ml/Cargo.toml
git commit -m "feat(ml): add MI-GAN inpainting"
```

---

## Task 8: MI-GAN 파이프라인 배선

**Files:**
- Modify: `crates/koharu-pipeline/src/stages/inpainting.rs`, `crates/koharu-pipeline/src/stages/mod.rs`, `crates/koharu-pipeline/src/config.rs`, `crates/koharu-pipeline/src/lib.rs`, `crates/koharu-pipeline/src/bin/run.rs`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`crates/koharu-pipeline/src/config.rs`의 `mod tests`에 추가한다.

```rust
    #[test]
    fn a_mi_gan_selection_round_trips() {
        let config = PipelineConfig {
            inpainting: InpaintingModel::MiGan(MiGanConfig::default()),
            ..PipelineConfig::default()
        };

        let text = toml::to_string(&config).unwrap();
        let parsed: PipelineConfig = toml::from_str(&text).unwrap();

        assert!(matches!(
            parsed.inpainting(),
            Ok(InpaintingModel::MiGan(_))
        ));
    }
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

Run: `cargo test -p koharu-pipeline mi_gan`
Expected: FAIL — `MiGanConfig` 미정의

- [ ] **Step 3: 설정 타입을 만든다**

`crates/koharu-pipeline/src/stages/inpainting.rs`의 `LaMaConfig` 아래에 넣는다.

```rust
/// MI-GAN 체크포인트 선택. 프롬프트가 없는 순수 소거 모델이라 소스만 갖는다.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct MiGanConfig {
    pub source: ComponentSourceConfig,
}

impl MiGanConfig {
    pub fn validate(&self) -> Result<()> {
        koharu_ml::source::ComponentSource::from(self.source.clone())
            .validate()
            .context("MI-GAN weights")
    }
}
```

- [ ] **Step 4: Model enum과 로드·추론을 확장한다**

`crates/koharu-ml` import에 `mi_gan::MiGan`을 추가한다.
`Model` enum(`stages/inpainting.rs:271`)에 변형을 넣는다.

```rust
    MiGan(Arc<Mutex<MiGan>>),
```

`Model::load`에 팔을 넣는다.

```rust
            InpaintingModel::MiGan(config) => Ok(Self::MiGan(Arc::new(Mutex::new(
                MiGan::load(
                    device,
                    &koharu_ml::source::ComponentSource::from(config.source.clone()),
                )
                .await?,
            )))),
```

`Processor::new` 검증에 팔을 넣는다.

```rust
            InpaintingModel::MiGan(settings) => settings.validate()?,
```

`model()`에 팔을 넣는다.

```rust
            InpaintingModel::MiGan(_) => "mi-gan",
```

추론을 부르는 `match`(`Model::LaMa(model) => ...`가 있는 곳)에 팔을 넣는다.
LaMa 팔과 같은 모양이다.

```rust
            Model::MiGan(model) => model
                .lock()
                .map_err(|_| anyhow!("MI-GAN mutex poisoned"))?
                .inference(image, mask, &InpaintRequest::default())?,
```

- [ ] **Step 5: config.rs에 배선한다**

Task 4와 같은 다섯 지점이다.

```rust
    #[serde(rename = "mi-gan")]
    MiGan(MiGanConfig),
```

```rust
    #[serde(rename = "mi-gan")]
    pub mi_gan: Option<MiGanConfig>,
```

```rust
            InpaintingModel::MiGan(_) => "mi-gan",
```

```rust
            InpaintingModel::MiGan(config) => {
                processor.mi_gan.get_or_insert_with(|| config.clone());
            }
```

```rust
            "mi-gan" => InpaintingModel::MiGan(file.processor.mi_gan.clone().unwrap_or_default()),
```

```rust
            InpaintingModel::MiGan(config) => Ok(InpaintingModel::MiGan(
                self.processor
                    .mi_gan
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
```

- [ ] **Step 6: 재수출과 CLI를 맞춘다**

`stages/mod.rs`와 `lib.rs`의 재수출 목록에 `MiGanConfig`를 알파벳 순서로
추가한다.

`crates/koharu-pipeline/src/bin/run.rs`의 `InpaintingChoice`에 변형을 넣고
(`bin/run.rs:118` 근방) 매핑을 추가한다.

```rust
                InpaintingChoice::MiGan => InpaintingModel::MiGan(MiGanConfig::default()),
```

`bin/run.rs:278`의 디퓨전 모델만 거르는 `matches!`는 MI-GAN을 포함하지
않는다 — 그대로 둔다.

- [ ] **Step 7: 테스트를 통과시킨다**

Run: `cargo test --workspace`
Expected: PASS

- [ ] **Step 8: 파이프라인으로 실제 확인한다**

Run: `cargo run -p koharu-pipeline --bin run --release -- --inpainting mi-gan <페이지 경로>`
Expected: 말풍선의 원문이 지워진 결과가 나온다.

- [ ] **Step 9: 커밋한다**

```bash
git add crates/koharu-pipeline
git commit -m "feat(pipeline): select MI-GAN for inpainting"
```

---

## Task 9: MI-GAN UI

**Files:**
- Modify: `packages/bridge/src/protocol.ts`, 인페인팅 모델 셀렉트, `packages/koharu/public/locales/*/translation.json`

- [ ] **Step 1: 타입을 다시 만든다**

Run: Task 5 Step 1과 같은 바인딩 생성 명령
Expected: `MiGanConfig`가 `protocol.ts`에 나타난다

- [ ] **Step 2: 모델 셀렉트에 항목을 넣는다**

인페인팅 모델을 고르는 셀렉트를 찾는다.

Run: `rg -n "aot-inpainting" packages/koharu`

`mi-gan` 항목을 `lama`와 `aot-inpainting` 사이에 넣고, 선택 시
`MiGanConfig`의 소스 편집 UI(Task 5에서 만든 컴포넌트 재사용)를 그린다.

- [ ] **Step 3: 9개 로케일에 문구를 넣는다**

`inpainting.model.mi-gan` 키를 추가한다. 설명 문구는 "빠르고 가벼운 소거
전용 모델" 취지로 쓴다.

- [ ] **Step 4: 확인한다**

Run: `bun run dev` 후 인페인팅 모델을 MI-GAN으로 바꾸고 페이지를 처리한다.
Expected: 체크포인트가 내려받아지고 결과가 나온다.

- [ ] **Step 5: 커밋한다**

```bash
git add packages
git commit -m "feat(ui): expose MI-GAN as an inpainting model"
```

---

## Task 10: 벤치

스펙의 성공 기준은 "MI-GAN이 더 빠르고 가볍다"는 **주장이 수치로 확인되거나
반증되는 것**이다. 벤치가 없으면 이 태스크 전체의 근거가 없다.

**Files:**
- Create: `crates/koharu-ml/benches/mi_gan.rs`
- Modify: `crates/koharu-ml/Cargo.toml`

- [ ] **Step 1: 기존 벤치를 본뜬다**

`crates/koharu-ml/benches/lama.rs`를 열어 그대로 따른다. 픽스처는 같은
것(`benches/fixtures/inpaint/image_4k.jpg`, `mask_4k.png`)을 쓴다.
`LaMa::load(...)`를 `MiGan::load(device, &ComponentSource::Builtin)`으로,
`model.inference(...)`는 시그니처가 같으므로 그대로 둔다.

- [ ] **Step 2: Cargo.toml에 등록한다**

`crates/koharu-ml/Cargo.toml`의 `[[bench]] name = "lama"` 아래에 넣는다.

```toml
[[bench]]
name = "mi_gan"
harness = false
```

- [ ] **Step 3: 두 벤치를 나란히 돌린다**

Run: `cargo bench -p koharu-ml --bench lama --bench mi_gan`
Expected: 두 모델의 4K 페이지 처리 시간이 출력된다.

- [ ] **Step 4: 결과를 기록한다**

`crates/koharu-ml/benches/fixtures/inpaint/README.md`에 측정한 하드웨어와
두 수치를 적는다. 사람이 나중에 "MI-GAN이 정말 빨랐는가"를 다시 재지 않고
확인할 수 있어야 한다.

- [ ] **Step 5: 커밋한다**

```bash
git add crates/koharu-ml/benches crates/koharu-ml/Cargo.toml
git commit -m "bench(ml): compare MI-GAN against LaMa"
```

---

## Self-Review 결과

**스펙 커버리지**

| 스펙 항목 | 태스크 |
|---|---|
| 1부: TorchScript 로더 | Task 1 |
| 2부: LaMa 체크포인트 소스 | Task 2, 3, 4, 5 |
| 3부: MI-GAN | Task 6, 7, 8, 9, 10 |
| 열린 질문: 체크포인트 재배포 | Task 0에서 **미러링하지 않고** `RemoteFile`로 원본 직접 참조로 확정 |
| 열린 질문: Manga 인페인터 | 범위 밖 유지. MI-GAN 벤치(Task 10) 결과를 보고 별도 판단 |

**타입 일관성**

`WeightsFormat`(ml) ↔ `WeightsFormatConfig`(pipeline),
`ComponentSource`(ml) ↔ `ComponentSourceConfig`(pipeline) 두 쌍이 `From`으로
연결된다. `ComponentSourceConfig → ComponentSource`의 `From`은
`stages/inpainting.rs:56-72`에 이미 존재하므로 새로 만들지 않는다 —
구현 시 그 `impl`이 `koharu_ml::source::ComponentSource`를 가리키도록
경로만 갱신한다(Task 2에서 타입이 이동했으므로).

**남은 위험**

- ~~Task 0의 자리표시자~~ **해소됨.** 세 다이제스트를 실측해 계획에 반영했다
  (Task 0 Step 8 표). 자리표시자는 남아 있지 않다.
- **이 계획서의 코드는 컴파일된 적이 없다.** Task 0에서 이미 두 건이
  드러났다 — `blake3::Hasher::update_mmap_rayon`은 워크스페이스가 해당
  피처를 켜지 않아 쓸 수 없었고(`update_reader`로 대체), 다이제스트
  대소문자 정책이 산문과 코드에서 어긋나 있었다(대소문자 허용 + 캐시
  경로 소문자 정규화로 확정). **뒤 태스크의 코드 블록도 같은 눈으로 볼
  것.** 특히 Task 3의 `Box<dyn Backend>`와 Task 7의 텐서 연산은 검증되지
  않은 추론이다. 구현자는 고치고 신고하도록 지시받는다.
- **상류 릴리스 자산의 가용성.** 원본을 미러링하지 않으므로 GitHub 릴리스가
  사라지면 내장 기본값이 죽는다. 세 URL 모두 현재 200으로 응답하는 것을
  확인했다. 다이제스트가 있으므로 상류가 같은 URL에 다른 파일을 올려도
  조용한 교체는 일어나지 않고, 명시적 실패가 된다. 그때는 설정의 `Url`
  갈래로 사용자가 직접 우회할 수 있다.
- **`pad_img_to_modulo(_, 512)`가 정사각을 보장하지 않는다.** IOPaint의
  `pad_to_square = True`는 짧은 변도 512로 채운다. `resize_dimensions`가
  긴 변을 512로 맞추므로 결과는 512×N(N≤512)이고, modulo 512 패딩이 N을
  512로 올려 결국 512×512가 된다. Task 7 Step 6에서 출력 크기를 확인해
  이 추론을 검증한다 — 어긋나면 `pad_to_square` 전용 헬퍼를
  `inpaint_ops`에 추가한다.
