use std::sync::Arc;

use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use http_body_util::Full;
use tokio::net::TcpListener;
use tokio::sync::watch;

use crate::hub::Hub;

/// Shared application state for HTTP handlers.
pub struct HttpState {
    pub hub: Arc<Hub>,
    pub train_tx: watch::Sender<bool>,
}

/// Start the Hyper HTTP server on the given address.
pub async fn serve(state: Arc<HttpState>, addr: &str) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    log::info!("http: listening on http://{addr}");

    loop {
        let (stream, _) = listener.accept().await?;
        let state = state.clone();
        tokio::spawn(async move {
            let svc = service_fn(move |req| handle_request(state.clone(), req));
            let io = TokioIo::new(stream);
            if let Err(e) = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, svc)
                .await
            {
                log::warn!("http: connection error: {e}");
            }
        });
    }
}

async fn handle_request(
    state: Arc<HttpState>,
    req: Request<Incoming>,
) -> Result<Response<Full<bytes::Bytes>>, hyper::Error> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/") => serve_static("index.html").await,
        (&Method::GET, path) if path.starts_with("/record/") => {
            handle_record(state, req).await
        }
        _ => {
            let mut not_found = Response::new(Full::new(bytes::Bytes::from("not found")));
            *not_found.status_mut() = StatusCode::NOT_FOUND;
            Ok(not_found)
        }
    }
}

async fn serve_static(filename: &str) -> Result<Response<Full<bytes::Bytes>>, hyper::Error> {
    let path = format!("static/{}", filename);
    match tokio::fs::read(&path).await {
        Ok(data) => {
            let content_type = if filename.ends_with(".html") {
                "text/html; charset=utf-8"
            } else {
                "application/octet-stream"
            };
            let mut res = Response::new(Full::new(bytes::Bytes::from(data)));
            res.headers_mut()
                .insert("Content-Type", content_type.parse().unwrap());
            Ok(res)
        }
        Err(_) => {
            let mut res = Response::new(Full::new(bytes::Bytes::from("not found")));
            *res.status_mut() = StatusCode::NOT_FOUND;
            Ok(res)
        }
    }
}

async fn handle_record(
    _state: Arc<HttpState>,
    req: Request<Incoming>,
) -> Result<Response<Full<bytes::Bytes>>, hyper::Error> {
    // Placeholder — will be implemented in Phase 7 with the recorder
    let body = match req.method() {
        &Method::POST => "recording placeholder",
        _ => "method not allowed",
    };
    let mut res = Response::new(Full::new(bytes::Bytes::from(body)));
    if req.method() != Method::POST {
        *res.status_mut() = StatusCode::METHOD_NOT_ALLOWED;
    }
    Ok(res)
}

/// Return a JSON 200 response.
pub fn json_response<T: serde::Serialize>(
    data: &T,
) -> Result<Response<Full<bytes::Bytes>>, hyper::Error> {
    let json = serde_json::to_string(data).unwrap_or_default();
    let mut res = Response::new(Full::new(bytes::Bytes::from(json)));
    res.headers_mut()
        .insert("Content-Type", "application/json".parse().unwrap());
    Ok(res)
}
