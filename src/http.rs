use std::sync::Arc;

use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use http_body_util::{BodyExt, Full};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};

use crate::hub::Hub;
use crate::recorder::Recorder;
use crate::EmulatorCmd;

pub struct HttpState {
    pub hub: Arc<Hub>,
    pub train_tx: watch::Sender<bool>,
    pub rec: Arc<Recorder>,
    pub cmd_tx: mpsc::UnboundedSender<EmulatorCmd>,
}

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
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    match (&method, path.as_str()) {
        (&Method::GET, "/") => serve_static("index.html").await,
        (&Method::POST, "/train/start") => {
            if state.train_tx.send(true).is_err() {
                return Ok(error_response("channel closed"));
            }
            json_ok(&serde_json::json!({"status": "training"}))
        }
        (&Method::POST, "/train/stop") => {
            if state.train_tx.send(false).is_err() {
                return Ok(error_response("channel closed"));
            }
            json_ok(&serde_json::json!({"status": "stopped"}))
        }
        (&Method::POST, "/record/start") => {
            match state.rec.start(".") {
                Ok(dir) => json_ok(&serde_json::json!({"status": "ok", "dir": dir})),
                Err(e) => Ok(error_response(&e.to_string())),
            }
        }
        (&Method::POST, "/record/stop") => {
            let video = req.uri().query().map(|q| q.contains("video=true")).unwrap_or(false);
            match state.rec.stop(video) {
                Ok(result) => json_ok(&serde_json::json!(result)),
                Err(e) => Ok(error_response(&e.to_string())),
            }
        }
        (&Method::GET, "/record/status") => {
            json_ok(&serde_json::json!(state.rec.status()))
        }
        (&Method::POST, "/state/save") => {
            let body = read_body(req).await;
            if let Some(path) = body.get("path").and_then(|v| v.as_str()) {
                let _ = state.cmd_tx.send(EmulatorCmd::SaveState(path.to_string()));
                json_ok(&serde_json::json!({"status": "ok"}))
            } else {
                Ok(error_response("missing path"))
            }
        }
        (&Method::POST, "/state/load") => {
            let body = read_body(req).await;
            if let Some(path) = body.get("path").and_then(|v| v.as_str()) {
                let _ = state.cmd_tx.send(EmulatorCmd::LoadState(path.to_string()));
                json_ok(&serde_json::json!({"status": "ok"}))
            } else {
                Ok(error_response("missing path"))
            }
        }
        _ => {
            let mut res = Response::new(Full::new(bytes::Bytes::from("not found")));
            *res.status_mut() = StatusCode::NOT_FOUND;
            Ok(res)
        }
    }
}

async fn read_body(req: Request<Incoming>) -> serde_json::Value {
    let body_bytes = req.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
    serde_json::from_slice(&body_bytes).unwrap_or(serde_json::Value::Null)
}

async fn serve_static(filename: &str) -> Result<Response<Full<bytes::Bytes>>, hyper::Error> {
    let path = format!("static/{filename}");
    match tokio::fs::read(&path).await {
        Ok(data) => {
            let ct = if filename.ends_with(".html") { "text/html; charset=utf-8" }
            else { "application/octet-stream" };
            let mut res = Response::new(Full::new(bytes::Bytes::from(data)));
            res.headers_mut().insert("Content-Type", ct.parse().unwrap());
            Ok(res)
        }
        Err(_) => Ok(not_found()),
    }
}

fn json_ok<T: serde::Serialize>(val: &T) -> Result<Response<Full<bytes::Bytes>>, hyper::Error> {
    let json = serde_json::to_string(val).unwrap_or_default();
    let mut res = Response::new(Full::new(bytes::Bytes::from(json)));
    res.headers_mut().insert("Content-Type", "application/json".parse().unwrap());
    Ok(res)
}

fn error_response(msg: &str) -> Response<Full<bytes::Bytes>> {
    let mut res = Response::new(Full::new(bytes::Bytes::from(msg.to_string())));
    *res.status_mut() = StatusCode::CONFLICT;
    res
}

fn not_found() -> Response<Full<bytes::Bytes>> {
    let mut res = Response::new(Full::new(bytes::Bytes::from("not found")));
    *res.status_mut() = StatusCode::NOT_FOUND;
    res
}
