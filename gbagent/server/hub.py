"""Broadcast hub — thread-safe fan-out of frames + metrics to WebSocket clients.

Usage
-----
    hub = BroadcastHub()
    hub.register(ws)
    hub.unregister(ws)

    # From training thread:
    hub.broadcast_frame(png_bytes)
    hub.broadcast_metrics({"step": 1000, "return": 42.5})

    # Check control flags:
    if hub.stop_requested:
        break
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("gbagent.server.hub")


class BroadcastHub:
    """Manages WebSocket connections and broadcasts frames/metrics.

    Designed to be accessed from both the aiohttp async event loop thread
    and the synchronous training loop thread.  Frame/metric broadcast
    methods are intended to be called via ``asyncio.run_coroutine_threadsafe``
    from the training thread.

    Control flags (``stop_requested``, ``train_requested``) are plain Python
    attributes suitable for GIL-safe single-writer access from the training
    thread.
    """

    def __init__(self) -> None:
        # WebSocket clients (accessed from server event-loop thread via async methods)
        self._clients: set[Any] = set()

        # Cached latest frame / metrics
        self._latest_frame: bytes | None = None
        self._latest_metrics: dict[str, float] = {}

        # Joypad state
        self._joypad_state: dict[str, bool] = {}
        self._on_joypad: list[Callable] = []

        # Control flags (accessed from training thread, GIL-safe)
        self.stop_requested: bool = False
        self.train_requested: bool = False

    # ------------------------------------------------------------------
    # Client management (server thread only)
    # ------------------------------------------------------------------

    def register(self, ws: Any) -> None:
        """Register a WebSocket client for broadcasts."""
        self._clients.add(ws)

    def unregister(self, ws: Any) -> None:
        """Remove a WebSocket client."""
        self._clients.discard(ws)

    @property
    def num_clients(self) -> int:
        return len(self._clients)

    # ------------------------------------------------------------------
    # Joypad (server thread writes, training thread reads)
    # ------------------------------------------------------------------

    def set_joypad_state(self, state: dict[str, bool]) -> None:
        """Update joypad state and notify listeners."""
        self._joypad_state = state
        for cb in self._on_joypad:
            cb(state)

    def get_joypad_state(self) -> dict[str, bool]:
        """Return a copy of the current joypad state."""
        return dict(self._joypad_state)

    def on_joypad_change(self, callback: Callable[[dict[str, bool]], None]) -> None:
        """Register a joypad-change callback."""
        self._on_joypad.append(callback)

    # ------------------------------------------------------------------
    # Frame / metrics access (for late-joining clients)
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> bytes | None:
        return self._latest_frame

    def get_latest_metrics(self) -> dict[str, float]:
        return dict(self._latest_metrics)

    # ------------------------------------------------------------------
    # Broadcast methods (call from any thread via run_coroutine_threadsafe)
    # ------------------------------------------------------------------

    async def broadcast_frame(self, png_bytes: bytes) -> None:
        """Send a live frame (binary) to all connected WebSocket clients.

        Binary protocol: ``0x00`` prefix byte followed by raw PNG data.
        """
        self._latest_frame = png_bytes
        if not self._clients:
            return
        msg = b"\x00" + png_bytes
        dead: set[Any] = set()
        for ws in self._clients:
            try:
                await ws.send_bytes(msg)
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                dead.add(ws)
        self._clients -= dead

    async def broadcast_metrics(self, data: dict[str, float]) -> None:
        """Send a metrics update (JSON) to all connected WebSocket clients."""
        self._latest_metrics = data
        if not self._clients:
            return
        payload = json.dumps({"type": "metrics", "data": data})
        dead: set[Any] = set()
        for ws in self._clients:
            try:
                await ws.send_str(payload)
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                dead.add(ws)
        self._clients -= dead

    async def broadcast_event(self, event: str, **kwargs: Any) -> None:
        """Send a named event (train_start, train_stop, etc.)."""
        if not self._clients:
            return
        payload = json.dumps({"type": "event", "event": event, **kwargs})
        dead: set[Any] = set()
        for ws in self._clients:
            try:
                await ws.send_str(payload)
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                dead.add(ws)
        self._clients -= dead
