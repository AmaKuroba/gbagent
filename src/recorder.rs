use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

/// Recorder saves gameplay frames to disk as PNG + JSONL.
/// Optionally generates MP4 via ffmpeg on stop.
pub struct Recorder {
    state: Mutex<RecorderInner>,
}

struct RecorderInner {
    active: bool,
    output_dir: PathBuf,
    frames_dir: PathBuf,
    jsonl: Option<fs::File>,
    frame_count: usize,
    start_time: std::time::Instant,
}

impl Recorder {
    pub fn new() -> Self {
        Self { state: Mutex::new(RecorderInner {
            active: false,
            output_dir: PathBuf::new(),
            frames_dir: PathBuf::new(),
            jsonl: None,
            frame_count: 0,
            start_time: std::time::Instant::now(),
        })}
    }

    /// Start a new recording session.
    pub fn start(&self, base_dir: &str) -> anyhow::Result<String> {
        let mut inner = self.state.lock().unwrap();
        if inner.active {
            anyhow::bail!("already recording");
        }

        let ts = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
        let output_dir = PathBuf::from(base_dir).join("recordings").join(ts.to_string());
        let frames_dir = output_dir.join("frames");
        fs::create_dir_all(&frames_dir)?;

        let jsonl_path = output_dir.join("data.jsonl");
        let jsonl = fs::File::create(&jsonl_path)?;

        let dir_str = output_dir.to_string_lossy().to_string();
        log::info!("recording started: {dir_str}");

        *inner = RecorderInner {
            active: true,
            output_dir,
            frames_dir,
            jsonl: Some(jsonl),
            frame_count: 0,
            start_time: std::time::Instant::now(),
        };
        Ok(dir_str)
    }

    /// Record a single frame.
    pub fn record_frame(
        &self,
        png_data: &[u8],
        dpad: u8,
        btn: u8,
        reward: f64,
        loss: f64,
        epsilon: f64,
    ) {
        let inner = self.state.lock().unwrap();
        if !inner.active {
            return;
        }

        let n = inner.frame_count + 1;
        let frame_file = format!("{:06}.png", n);
        let frame_path = inner.frames_dir.join(&frame_file);

        // Write PNG
        if let Err(e) = fs::write(&frame_path, png_data) {
            log::warn!("recorder: write {frame_file}: {e}");
            return;
        }

        // Write JSONL line
        if let Some(ref file) = inner.jsonl {
            let entry = serde_json::json!({
                "frame": frame_file,
                "dpad": dpad,
                "btn": btn,
                "reward": reward,
                "loss": loss,
                "epsilon": epsilon,
                "t_ms": chrono::Utc::now().timestamp_millis(),
            });
            let mut file = file;
            if writeln!(file, "{}", entry).is_err() {
                log::warn!("recorder: jsonl write failed");
            }
        }

        drop(inner);
        // Update frame count in a separate lock to minimize contention
        let mut inner = self.state.lock().unwrap();
        inner.frame_count = n;
    }

    /// Stop recording. If `encode_video`, run ffmpeg to create MP4.
    pub fn stop(&self, encode_video: bool) -> anyhow::Result<RecorderResult> {
        let mut inner = self.state.lock().unwrap();
        if !inner.active {
            anyhow::bail!("not recording");
        }

        // Close JSONL
        inner.jsonl.take();

        let frame_count = inner.frame_count;
        let output_dir = inner.output_dir.clone();
        inner.active = false;

        let video_path = if encode_video && frame_count > 0 {
            let video = output_dir.join("recording.mp4");
            let frames_dir = output_dir.join("frames");
            match encode_mp4(&frames_dir, &video, frame_count) {
                Ok(()) => Some(video.to_string_lossy().to_string()),
                Err(e) => {
                    log::warn!("recorder: ffmpeg encode failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        log::info!("recording stopped: {frame_count} frames");
        Ok(RecorderResult {
            frame_count,
            output_dir: output_dir.to_string_lossy().to_string(),
            video_path,
        })
    }

    /// Return current recording status.
    pub fn status(&self) -> RecorderStatus {
        let inner = self.state.lock().unwrap();
        RecorderStatus {
            active: inner.active,
            frame_count: inner.frame_count,
            duration_ms: if inner.active {
                inner.start_time.elapsed().as_millis() as u64
            } else {
                0
            },
            output_dir: if inner.active {
                Some(inner.output_dir.to_string_lossy().to_string())
            } else {
                None
            },
        }
    }
}

#[derive(serde::Serialize)]
pub struct RecorderResult {
    pub frame_count: usize,
    pub output_dir: String,
    pub video_path: Option<String>,
}

#[derive(serde::Serialize)]
pub struct RecorderStatus {
    pub active: bool,
    pub frame_count: usize,
    pub duration_ms: u64,
    pub output_dir: Option<String>,
}

/// Encode a directory of PNG frames to MP4 using ffmpeg.
fn encode_mp4(frames_dir: &PathBuf, output: &PathBuf, _frame_count: usize) -> anyhow::Result<()> {
    let pattern = frames_dir.join("%06d.png");
    let pattern_str = pattern.to_string_lossy().to_string();

    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate", "60",
            "-i", &pattern_str,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
        ])
        .arg(output)
        .status()?;

    if !status.success() {
        anyhow::bail!("ffmpeg exited with status: {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_recorder_start_stop_no_video() {
        let r = Arc::new(Recorder::new());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        r.start(&path).unwrap();
        assert!(r.status().active);

        let result = r.stop(false).unwrap();
        assert!(result.frame_count == 0);
        assert!(result.video_path.is_none());
    }

    #[test]
    fn test_recorder_double_start_fails() {
        let r = Arc::new(Recorder::new());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        r.start(&path).unwrap();
        assert!(r.start(&path).is_err());
    }

    #[test]
    fn test_recorder_stop_without_start_fails() {
        let r = Recorder::new();
        assert!(r.stop(false).is_err());
    }

    #[test]
    fn test_recorder_record_frame() {
        let r = Arc::new(Recorder::new());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        r.start(&path).unwrap();
        r.record_frame(&[0u8, 1, 2], 1, 2, 3.0, 0.1, 0.01);
        let result = r.stop(false).unwrap();
        assert_eq!(result.frame_count, 1);
    }

    #[test]
    fn test_recorder_status_not_active() {
        let r = Recorder::new();
        let s = r.status();
        assert!(!s.active);
        assert_eq!(s.frame_count, 0);
    }
}
