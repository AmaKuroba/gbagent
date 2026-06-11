"""WebSocket JSON-RPC client for gbagent.

Connects to a running gbagent serve instance via WebSocket.
No subprocess management -- the emulator runs standalone.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import websocket as ws_module  # type: ignore[import-untyped]
    from websocket import WebSocketApp, WebSocketTimeoutException  # type: ignore[import-untyped]
except ImportError:
    WebSocketApp = None  # type: ignore[misc]
    WebSocketTimeoutException = Exception
    ws_module = None


class GBAError(Exception):
    """Raised when gbagent returns an error response."""


class GBWSClient:
    """JSON-RPC 2.0 client for gbagent over WebSocket.

    Connects to a running gbagent serve instance via its JSON-RPC WebSocket endpoint.

    Args:
        url: WebSocket URL (e.g. ``ws://localhost:8767/ws``).
        timeout: Seconds to wait for RPC responses.
    """

    def __init__(self, url: str = "ws://localhost:8767/ws", timeout: float = 30.0) -> None:
        self._url = url
        self._timeout = timeout
        self._ws: WebSocketApp | None = None
        self._req_id = 0
        self._responses: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connected = threading.Event()

    # ── lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        """Connect to gbagent WebSocket."""
        if self._ws is not None:
            return

        self._stop_event.clear()
        self._responses.clear()
        self._req_id = 0

        self._ws = WebSocketApp(
            self._url,
            on_open=lambda _: self._connected.set(),
            on_message=self._on_message,
            on_error=lambda _, e: self._stop_event.set(),
            on_close=lambda _, __, ___: self._stop_event.set(),
        )

        t = threading.Thread(target=self._ws.run_forever, daemon=True)
        t.start()

        if not self._connected.wait(timeout=10):
            raise ConnectionError(f"Failed to connect to {self._url}")

    def stop(self) -> None:
        """Disconnect from gbagent."""
        self._stop_event.set()
        if self._ws:
            self._ws.close()
            self._ws = None

    def __enter__(self) -> GBWSClient:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    # ── JSON-RPC call ────────────────────────────────────────

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC request and return the result field."""
        if not self._ws:
            raise RuntimeError("GBWSClient not started")

        self._req_id += 1
        req_id = self._req_id
        request: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params:
            request["params"] = params

        with self._lock:
            self._responses[req_id] = None  # type: ignore[assignment]
            self._ws.send(json.dumps(request))

        return self._wait_for_response(req_id, method)

    def _wait_for_response(self, req_id: int, method: str) -> Any:
        """Block until the response arrives for the given request ID."""
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                raise ConnectionError(f"WebSocket disconnected while waiting for {method}")
            with self._lock:
                entry = self._responses.get(req_id)
                if entry is not None:
                    del self._responses[req_id]
                    if "error" in entry:
                        err = entry["error"]
                        raise GBAError(err.get("message", str(err)))
                    return entry.get("result")
            time.sleep(0.001)
        raise TimeoutError(f"RPC call {req_id} ({method}) timed out after {self._timeout}s")

    def _on_message(self, _ws: Any, message: str) -> None:
        """Handle incoming WebSocket text message (JSON-RPC response)."""
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict):
            return
        req_id = msg.get("id")
        if req_id is not None:
            with self._lock:
                if req_id in self._responses:
                    self._responses[req_id] = msg

    # ── game boy API ─────────────────────────────────────────

    def get_screen(self) -> Image.Image:
        """Get current screen as grayscale PIL Image (144x160). No frame advancement."""
        raw = self.call("get_screen")
        return self._parse_image(raw)

    def wait_frames(self, count: int) -> int:
        """Block until the emulator frame counter advances by ``count`` frames.

        Returns the new frame count.
        """
        state = self.call("get_state")
        target = state["ppu"]["frame_count"] + count
        # Sleep most of the expected time (60 fps ≈ 16.7ms per frame)
        # before polling to avoid hammering the server.
        time.sleep(count * 0.016)
        while True:
            state = self.call("get_state")
            if state["ppu"]["frame_count"] >= target:
                return state["ppu"]["frame_count"]
            time.sleep(0.001)

    def screenshot(self) -> Image.Image:
        """Alias for get_screen (matches legacy GBClient API)."""
        return self.get_screen()

    def press_button(self, button: str) -> None:
        """Press a button (held until released or next press)."""
        self.call("press_button", {"button": button.upper()})

    def release_button(self, button: str) -> None:
        """Release a button."""
        self.call("release_button", {"button": button.upper()})

    def release_all(self) -> None:
        """Release all held buttons."""
        self.call("release_all", {})

    def read_ram(self, addr: int) -> int:
        """Read one byte from RAM (0x0000-0xFFFF)."""
        result = self.call("read_ram", {"address": addr})
        if isinstance(result, dict):
            return int(result.get("value", 0))
        return int(result)

    def save_state(self, path: str | Path) -> None:
        """Save emulator state."""
        self.call("save_state", {"path": str(path)})

    def load_state(self, path: str | Path) -> None:
        """Load emulator state."""
        self.call("load_state", {"path": str(path)})

    def reset(self) -> None:
        """Reset the emulator (hard reset, not available to training)."""
        self.call("reset", {})

    def reset_state(self, path: str | None = None) -> None:
        """Reset to a saved start state (uses --load-state path if path is None)."""
        params: dict[str, Any] = {}
        if path:
            params["path"] = path
        self.call("reset_state", params)

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _parse_image(raw: Any) -> Image.Image:
        if isinstance(raw, str):
            return Image.open(io.BytesIO(base64.b64decode(raw))).convert("L")
        if isinstance(raw, dict):
            b64 = raw.get("image", raw.get("data", ""))
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
        raise GBAError(f"unexpected image response type: {type(raw)}")
