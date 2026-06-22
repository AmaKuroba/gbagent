"""Frame-by-frame PNG + JSONL recording to disk, with optional ffmpeg MP4 export.

Matches the on-disk format of the Rust gbagent recording system so that
recordings are interchangeable between the two implementations:

::

    recordings/
     └── session_<timestamp>/
          ├── frames/
          │   ├── frame_000001.png
          │   ├── frame_000002.png
          │   └── …
          ├── metadata.jsonl      # one JSON object per frame (step, reward, …)
          └── meta.json           # summary dict after stop()

Usage
-----
    from gbagent.recorder import Recorder

    recorder = Recorder(output_dir="recordings")

    # In training loop:
    if recorder.is_recording:
        recorder.record_frame(
            frame=rgb_array,
            metadata={"step": 10, "reward": 0.5, "action": 3, …},
        )

    # From dashboard / CLI:
    recorder.start()
    recorder.stop(encode_mp4=True)   # optionally invoke ffmpeg
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("gbagent.recorder")


# ===================================================================
# Recorder
# ===================================================================


class Recorder:
    """Thread-safe frame recorder.

    The recorder writes PNG frames and JSONL metadata to a session
    directory under ``output_dir``.  It is designed to be used from
    the training loop (main thread) while start/stop/status calls can
    originate from the dashboard HTTP thread.
    """

    def __init__(
        self,
        output_dir: str | Path = "recordings",
        fps: int = 15,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._fps = fps

        # Mutable state — protected by _lock
        self._lock = threading.Lock()
        self._active: bool = False
        self._session_dir: Path | None = None
        self._frames_dir: Path | None = None
        self._jsonl_path: Path | None = None
        self._frame_count: int = 0
        self._start_step: int = 0
        self._encoded_path: Path | None = None

        # Pre-allocate a reusable resized frame buffer
        self._resized: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """Whether a recording session is currently active."""
        return bool(self._active)

    @property
    def frame_count(self) -> int:
        """Number of frames written in the current (or last) session."""
        with self._lock:
            return self._frame_count

    @property
    def session_dir(self) -> Path | None:
        """Current or last session directory."""
        with self._lock:
            return self._session_dir

    @property
    def encoded_path(self) -> Path | None:
        """Path to the generated MP4, if any."""
        with self._lock:
            return self._encoded_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, start_step: int = 0) -> str:
        """Start a new recording session.

        Creates a timestamped directory under ``output_dir`` and
        prepares frame + metadata files.

        Parameters
        ----------
        start_step : int
            Global training step at which recording began (for JSONL).

        Returns
        -------
        str
            The short session name (directory basename).

        Raises
        ------
        RuntimeError
            If a recording is already in progress.
        """
        with self._lock:
            if self._active:
                raise RuntimeError("Recording is already in progress")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"session_{timestamp}"
            self._session_dir = self._output_dir / session_name
            self._frames_dir = self._session_dir / "frames"
            self._jsonl_path = self._session_dir / "metadata.jsonl"
            self._frame_count = 0
            self._start_step = start_step
            self._encoded_path = None

            # Create directories
            self._frames_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Recording started → %s",
                self._session_dir,
            )

            # Write meta.json skeleton (will be completed on stop)
            meta = {
                "session": session_name,
                "start_time": timestamp,
                "fps": self._fps,
                "start_step": start_step,
                "total_frames": 0,
                "end_step": 0,
                "encoded": False,
                "encoded_path": None,
            }
            self._write_json(self._session_dir / "meta.json", meta)

            self._active = True

        return session_name

    def stop(self, encode_mp4: bool = True) -> dict[str, Any]:
        """Stop the current recording session.

        Parameters
        ----------
        encode_mp4 : bool
            Whether to call ffmpeg to produce an MP4 from the PNG frames.

        Returns
        -------
        dict
            Summary of the completed session:
            ``{"session": str, "frames": int, "duration_s": float,
              "encoded": bool, "encoded_path": str | None}``
        """
        with self._lock:
            if not self._active:
                logger.warning("stop() called but no recording is active")
                return {"status": "not_recording"}

            self._active = False
            assert self._session_dir is not None  # set by start()
            total_frames = self._frame_count
            end_time = datetime.now().isoformat()
            duration_s = total_frames / self._fps if self._fps > 0 else 0.0

            # Update meta.json
            meta_path = self._session_dir / "meta.json"
            try:
                meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            except Exception:
                meta = {}
            meta["total_frames"] = total_frames
            meta["frames"] = total_frames  # convenience alias
            meta["end_time"] = end_time
            meta["duration_s"] = round(duration_s, 2)
            meta["end_step"] = self._start_step + total_frames
            self._write_json(meta_path, meta)

            logger.info(
                "Recording stopped — %d frames (%.1f s) in %s",
                total_frames,
                duration_s,
                self._session_dir,
            )

            session_dir = self._session_dir
            encoded = False
            encoded_rel = None

        # ── ffmpeg encoding (outside lock — can be slow) ────────
        encode_mp4 = encode_mp4 and total_frames > 0
        if encode_mp4:
            self._encode_mp4(session_dir)

            with self._lock:
                if self._encoded_path:
                    encoded = True
                    encoded_rel = str(
                        self._encoded_path.relative_to(self._output_dir)
                        if self._encoded_path.is_relative_to(self._output_dir)
                        else str(self._encoded_path)
                    )

                    # Update meta.json with encoded info
                    try:
                        meta_path = session_dir / "meta.json"
                        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
                        meta["encoded"] = True
                        meta["encoded_path"] = encoded_rel
                        self._write_json(meta_path, meta)
                    except Exception:
                        pass

        return {
            "session": session_dir.name,
            "frames": total_frames,
            "duration_s": round(duration_s, 2),
            "encoded": encoded,
            "encoded_path": encoded_rel,
        }

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_frame(
        self,
        frame: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write one frame PNG + JSONL metadata line.

        Called from the training loop for every step the recorder
        is active.

        Parameters
        ----------
        frame : np.ndarray
            RGB frame, shape ``(H, W, 3)``, uint8 in ``[0, 255]``.
        metadata : dict, optional
            Per-frame data to append as a JSON line.  Typical keys:
            ``step``, ``reward``, ``action``, ``value``, ``logprob``,
            ``bonus``, ``episode_return``, etc.
        """
        with self._lock:
            if not self._active:
                return

            assert self._frames_dir is not None  # set by start()
            self._frame_count += 1
            frame_idx = self._frame_count

            # Write PNG (fast — uses OpenCV imencode + file write)
            frame_path = self._frames_dir / f"frame_{frame_idx:06d}.png"
            success = cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not success:
                logger.error("Failed to write frame %d → %s", frame_idx, frame_path)

            # Write JSONL line
            if self._jsonl_path is not None:
                line = {"frame_index": frame_idx}
                if metadata:
                    line.update(metadata)
                try:
                    with open(self._jsonl_path, "a") as f:
                        f.write(json.dumps(line, default=str) + "\n")
                except OSError as exc:
                    logger.error("JSONL write error: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Atomically write a JSON file."""
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(path)

    def _encode_mp4(self, session_dir: Path) -> None:
        """Run ffmpeg to produce an MP4 from the PNG frame sequence.

        ffmpeg must be installed on ``PATH``; otherwise encoding is
        skipped with a warning.
        """
        if not shutil.which("ffmpeg"):
            logger.warning(
                "ffmpeg not found on PATH — skipping MP4 encoding for %s",
                session_dir,
            )
            return

        frames_glob = str(session_dir / "frames" / "frame_%06d.png")
        output_path = session_dir / "recording.mp4"

        cmd = [
            "ffmpeg",
            "-y",  # overwrite without asking
            "-framerate",
            str(self._fps),
            "-i",
            frames_glob,
            "-c:v",
            "libx264",
            "-preset",
            "medium",  # balanced speed/size
            "-crf",
            "23",  # constant rate factor
            "-pix_fmt",
            "yuv420p",  # broad compatibility
            "-movflags",
            "+faststart",  # web-optimised
            str(output_path),
        ]

        logger.info("Encoding MP4 via ffmpeg …")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max
            )
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out after 10 min")
            return
        except FileNotFoundError:
            logger.warning("ffmpeg not found — skipping MP4 encoding")
            return

        if result.returncode != 0:
            logger.error(
                "ffmpeg failed (code %d): %s",
                result.returncode,
                result.stderr.strip() or result.stdout.strip(),
            )
            return

        if output_path.is_file():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(
                "MP4 encoded → %s (%.1f MB)",
                output_path,
                size_mb,
            )
            with self._lock:
                self._encoded_path = output_path
        else:
            logger.error("ffmpeg produced no output file at %s", output_path)

    # ------------------------------------------------------------------
    # List recordings
    # ------------------------------------------------------------------

    def list_recordings(self) -> list[dict[str, Any]]:
        """Return metadata for all completed recording sessions."""
        if not self._output_dir.is_dir():
            return []

        sessions: list[dict[str, Any]] = []
        for entry in sorted(self._output_dir.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("session_"):
                continue
            meta_path = entry / "meta.json"
            info: dict[str, Any] = {"session": entry.name}
            if meta_path.is_file():
                with contextlib.suppress(Exception):
                    info.update(json.loads(meta_path.read_text()))
            sessions.append(info)
        return sessions
