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
