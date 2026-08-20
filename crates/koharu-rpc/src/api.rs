use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

use crate::AppState;
use crate::routes;

/// Compares in time that does not depend on how many leading bytes match, so
/// the token cannot be recovered one byte at a time. The length is not hidden;
/// that is the usual tradeoff and not worth the complexity of hiding.
fn secret_eq(presented: &str, expected: &str) -> bool {
    let (presented, expected) = (presented.as_bytes(), expected.as_bytes());
    presented.len() == expected.len()
        && presented
            .iter()
            .zip(expected)
            .fold(0u8, |difference, (a, b)| difference | (a ^ b))
            == 0
}

/// Undoes `%XX` escapes in a query value.
///
/// `openEventStream` builds its URL with `encodeURIComponent`, so a token
/// containing anything outside the unreserved set arrives escaped. Comparing
/// the raw form would reject a *correct* token — a failure that looks like a
/// misconfigured secret and is miserable to diagnose during a deployment.
fn percent_decode(value: &str) -> Cow<'_, str> {
    if !value.contains('%') {
        return Cow::Borrowed(value);
    }
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let escape = std::str::from_utf8(&bytes[index + 1..index + 3])
                .ok()
                .and_then(|escape| u8::from_str_radix(escape, 16).ok());
            if let Some(byte) = escape {
                decoded.push(byte);
                index += 3;
                continue;
            }
        }
        decoded.push(bytes[index]);
        index += 1;
    }
    String::from_utf8(decoded).map_or(Cow::Borrowed(value), Cow::Owned)
}

/// The token as a caller presented it: `Authorization: Bearer <token>`, or
/// `?token=<token>` for clients that cannot set headers — `EventSource`, and a
/// browser being pointed at the UI for the first time.
fn presented_token<'a>(headers: &'a axum::http::HeaderMap, uri: &'a Uri) -> Option<Cow<'a, str>> {
    let bearer = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(Cow::Borrowed);
    bearer.or_else(|| {
        let value = uri.query()?.split('&').find_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            (key == "token").then_some(value)
        })?;
        Some(percent_decode(value))
    })
}

fn authorized(headers: &axum::http::HeaderMap, uri: &Uri, expected: &str) -> bool {
    presented_token(headers, uri).is_some_and(|presented| secret_eq(&presented, expected))
}

async fn require_token(State(token): State<Arc<str>>, req: Request, next: Next) -> Response {
    if authorized(req.headers(), req.uri(), &token) {
        next.run(req).await
    } else {
        (StatusCode::UNAUTHORIZED, "missing or invalid API token").into_response()
    }
}

pub fn router(app: AppState, static_dir: Option<PathBuf>, api_token: Option<String>) -> Router {
    let mut api = routes::router();
    if let Some(token) = api_token.clone() {
        api = api.layer(middleware::from_fn_with_state(
            Arc::<str>::from(token),
            require_token,
        ));
    }
    let mut router = Router::new().nest("/api/v1", api);

    if let Some(dir) = static_dir {
        if dir.is_dir() {
            let index = dir.join("index.html");
            let serve_dir = ServeDir::new(&dir).not_found_service(ServeFile::new(index.clone()));
            let token = Arc::<str>::from(api_token.unwrap_or_default());
            let index_route = move |headers: axum::http::HeaderMap, uri: Uri| {
                let index = index.clone();
                let token = token.clone();
                async move { index_with_token(token, index, headers, uri).await }
            };
            router = router.route("/", get(index_route)).fallback_service(serve_dir);
        } else {
            tracing::warn!(
                path = %dir.display(),
                "static frontend directory does not exist; serving the API only"
            );
        }
    }

    router
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(app)
}

/// Serve the UI, with the API token injected for the page's own use.
///
/// **This route has to be authenticated exactly as strictly as the API is.**
/// It hands out the token, so leaving it open would have made the token
/// worthless: anyone who could reach the port could `curl /`, read
/// `window.__KOHARU_API_TOKEN__` out of the HTML, and then use the API freely.
/// With `CorsLayer::permissive` below that reached further still — any page
/// the operator visited could have read it cross-origin.
///
/// So a token-protected deployment is opened as `/?token=<token>` once, the
/// way Jupyter does it. There is deliberately no exemption for loopback
/// callers: a reverse proxy sharing the host would make every request look
/// local and quietly undo this. The desktop window is unaffected because it
/// does not set a token at all, and with none set nothing here is gated.
///
/// The static assets alongside this are *not* gated - they cannot be, since
/// the browser fetches them without the query string - but they carry no
/// secret. Only this response does.
async fn index_with_token(
    token: Arc<str>,
    index: PathBuf,
    headers: axum::http::HeaderMap,
    uri: Uri,
) -> Response {
    if !token.is_empty() && !authorized(&headers, &uri, &token) {
        return (
            StatusCode::UNAUTHORIZED,
            "missing or invalid API token; open this page as /?token=<token>",
        )
            .into_response();
    }
    let body = match tokio::fs::read_to_string(&index).await {
        Ok(body) => body,
        Err(error) => {
            tracing::error!(%error, path = %index.display(), "failed to read index.html");
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };

    let injected = if token.is_empty() {
        body
    } else {
        body.replacen(
            "<head>",
            &format!(
                "<head><script>window.__KOHARU_API_TOKEN__ = {token:?};</script>",
                token = token,
            ),
            1,
        )
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(Body::from(injected))
        .unwrap_or_else(|error| {
            tracing::error!(%error, "failed to build index.html response");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;

    fn bearer(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {value}").parse().expect("header value"),
        );
        headers
    }

    fn uri(path: &str) -> Uri {
        path.parse().expect("uri")
    }

    #[test]
    fn a_token_matches_only_itself() {
        assert!(secret_eq("s3cret", "s3cret"));
        assert!(!secret_eq("s3cret", "s3crea"));
        assert!(!secret_eq("s3cret", "s3cretx"), "a prefix is not a match");
        assert!(!secret_eq("", "s3cret"));
        assert!(secret_eq("", ""));
    }

    #[test]
    fn a_bearer_header_is_read() {
        assert_eq!(
            presented_token(&bearer("s3cret"), &uri("/")).as_deref(),
            Some("s3cret")
        );
    }

    #[test]
    fn a_query_parameter_is_read_for_clients_that_cannot_set_headers() {
        assert_eq!(
            presented_token(&HeaderMap::new(), &uri("/?token=s3cret")).as_deref(),
            Some("s3cret")
        );
        // EventSource appends its own parameters, so `token` cannot be assumed
        // to come first.
        assert_eq!(
            presented_token(&HeaderMap::new(), &uri("/api/v1/events?since=4&token=s3cret"))
                .as_deref(),
            Some("s3cret")
        );
    }

    #[test]
    fn an_escaped_query_token_is_decoded() {
        // What `encodeURIComponent("a+b/c=")` produces. Comparing this raw
        // would reject a correct token.
        assert_eq!(
            presented_token(&HeaderMap::new(), &uri("/?token=a%2Bb%2Fc%3D")).as_deref(),
            Some("a+b/c=")
        );
        assert!(authorized(
            &HeaderMap::new(),
            &uri("/?token=a%2Bb%2Fc%3D"),
            "a+b/c="
        ));
        // A stray `%` is not an escape and must not be swallowed or panic.
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("a%zzb"), "a%zzb");
        assert_eq!(percent_decode("plain"), "plain");
    }

    #[test]
    fn nothing_presented_is_not_an_empty_token() {
        assert_eq!(presented_token(&HeaderMap::new(), &uri("/")), None);
        assert_eq!(
            presented_token(&HeaderMap::new(), &uri("/?other=1")),
            None,
            "an unrelated parameter must not read as a token"
        );
        assert!(
            !authorized(&HeaderMap::new(), &uri("/"), ""),
            "an unset expected token must not let an unauthenticated caller through here"
        );
    }

    #[test]
    fn authorization_needs_the_right_token() {
        assert!(authorized(&bearer("s3cret"), &uri("/"), "s3cret"));
        assert!(authorized(&HeaderMap::new(), &uri("/?token=s3cret"), "s3cret"));
        assert!(!authorized(&bearer("wrong"), &uri("/"), "s3cret"));
        assert!(!authorized(&HeaderMap::new(), &uri("/?token=wrong"), "s3cret"));
        assert!(!authorized(&HeaderMap::new(), &uri("/"), "s3cret"));
    }

    #[test]
    fn a_malformed_header_does_not_authorize() {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            "s3cret".parse().expect("header value"),
        );
        assert!(
            !authorized(&headers, &uri("/"), "s3cret"),
            "the Bearer prefix is required"
        );
    }
}

