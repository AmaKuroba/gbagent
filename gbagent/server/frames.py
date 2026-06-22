"""Frame encoding helpers for the dashboard server.

Provides ``frame_to_png`` to encode a raw RGB numpy array as PNG bytes
suitable for broadcasting over WebSocket.
"""

from __future__ import annotations

import cv2
import numpy as np


def frame_to_png(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode an RGB (or grayscale) frame as PNG bytes.

    Parameters
    ----------
    frame : np.ndarray
        Shape ``(H, W, 3)`` for RGB or ``(H, W)`` / ``(H, W, 1)`` for
        grayscale.  Values in ``[0, 255]`` (uint8) or ``[0, 1]`` (float).
    quality : int
        PNG compression quality 0–100 (higher = larger file; default 85).

    Returns
    -------
    bytes
        Raw PNG bytes suitable for WebSocket broadcast.
    """
    # Normalise float → uint8
    if frame.dtype in (np.float32, np.float64):
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

    # Convert grayscale to RGB for PNG
    if frame.ndim == 2 or frame.shape[-1] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Encode
    success, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 9 - quality // 12])
    if not success:
        return b""
    return buf.tobytes()
