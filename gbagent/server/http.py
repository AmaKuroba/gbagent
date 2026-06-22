"""aiohttp HTTP server + WebSocket endpoint for the gbagent dashboard.

Provides:
- ``create_app`` — aiohttp ``web.Application`` factory.
- ``DashboardServer`` — runs the app in a background daemon thread.

Routes
------
- ``GET  /``              → static dashboard HTML
- ``GET  /ws``            → WebSocket (frame relay + joypad input)
- ``GET  /api/config``    → current training config (JSON)
- ``POST /api/train/start`` → signal training start
- ``POST /api/train/stop``  → signal training stop
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from gbagent.recorder import Recorder
from gbagent.server.hub import BroadcastHub

logger = logging.getLogger("gbagent.server.http")

# Where static assets live (relative to project root)
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


# ===================================================================
# aiohttp app factory
# ===================================================================


def create_app(
    hub: BroadcastHub | None = None,
    config: Any = None,
    recorder: Recorder | None = None,
) -> web.Application:
    """Create and configure the aiohttp ``web.Application``.

    Parameters
    ----------
    hub : BroadcastHub, optional
        Shared hub instance.  Created fresh if omitted.
    config : Config, optional
        Resolved gbagent config for the ``/api/config`` endpoint.
    recorder : Recorder, optional
        Frame recorder instance for the ``/api/recorder/*`` endpoints.
    """
    if hub is None:
        hub = BroadcastHub()

    app = web.Application()
    app["hub"] = hub
    if config is not None:
        app["config"] = config
    if recorder is not None:
        app["recorder"] = recorder

    # Static file serving — serve index.html at /, and other files under /static/
    if STATIC_DIR.is_dir():
        app.router.add_get("/", handle_index)
        app.router.add_static("/static", str(STATIC_DIR))
    else:
        app.router.add_get("/", handle_index)

    # WebSocket
    app.router.add_get("/ws", handle_websocket)

    # REST API — training
    app.router.add_get("/api/config", handle_config)
    app.router.add_post("/api/train/start", handle_train_start)
    app.router.add_post("/api/train/stop", handle_train_stop)

    # REST API — recording
    app.router.add_get("/api/recorder/status", handle_recorder_status)
    app.router.add_post("/api/recorder/start", handle_recorder_start)
    app.router.add_post("/api/recorder/stop", handle_recorder_stop)
    app.router.add_get("/api/recorder/files", handle_recorder_files)

    return app


# ===================================================================
# Route handlers
# ===================================================================


async def handle_index(request: web.Request) -> web.Response:
    """Serve ``static/index.html``."""
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return web.FileResponse(index_path)
    return web.Response(
        text="<h1>GBAGent Dashboard</h1><p>static/index.html not found</p>",
        content_type="text/html",
        status=404,
    )


async def handle_config(request: web.Request) -> web.Response:
    """Return the current training config as JSON."""
    cfg = request.app.get("config")
    if cfg is None:
        return web.json_response({"status": "no_config"})
    # Basic serialisable representation
    info = {
        "env": {
            "game": cfg.env.game,
            "state": cfg.env.state,
            "frame_skip": cfg.env.frame_skip,
            "frame_stack": cfg.env.frame_stack,
        },
        "agent": {
            "learning_rate": cfg.agent.learning_rate,
            "gamma": cfg.agent.gamma,
            "clip_epsilon": cfg.agent.clip_epsilon,
            "hidden_dim": cfg.agent.hidden_dim,
        },
        "train": {
            "total_timesteps": cfg.train.total_timesteps,
            "num_envs": cfg.train.num_envs,
            "n_steps": cfg.train.n_steps,
        },
    }
    return web.json_response({"status": "ok", "config": info})


async def handle_train_start(request: web.Request) -> web.Response:
    """Signal the training loop to start / resume."""
    hub: BroadcastHub = request.app["hub"]
    hub.train_requested = True
    return web.json_response({"status": "training_requested"})


async def handle_train_stop(request: web.Request) -> web.Response:
    """Signal the training loop to stop gracefully."""
    hub: BroadcastHub = request.app["hub"]
    hub.stop_requested = True
    return web.json_response({"status": "stop_requested"})


# ===================================================================
# Recorder API handlers
# ===================================================================


def _get_recorder(request: web.Request) -> Recorder:
    """Get the Recorder instance from the app, or raise 503."""
    recorder = request.app.get("recorder")
    if recorder is None:
        raise web.HTTPServiceUnavailable(
            text='{"status": "error", "message": "Recorder not available"}',
            content_type="application/json",
        )
    return recorder


async def handle_recorder_status(request: web.Request) -> web.Response:
    """Return current recording state.

    Returns
    -------
    JSON
        ``{"status": "ok", "recording": bool, "frame_count": int,
          "session_dir": str | None, "encoded_path": str | None}``
    """
    try:
        recorder = _get_recorder(request)
    except web.HTTPException:
        return web.json_response(
            {
                "status": "ok",
                "recording": False,
                "frame_count": 0,
                "session_dir": None,
                "encoded_path": None,
                "message": "Recorder not configured",
            }
        )

    sd = recorder.session_dir
    ep = recorder.encoded_path
    return web.json_response(
        {
            "status": "ok",
            "recording": recorder.is_recording,
            "frame_count": recorder.frame_count,
            "session_dir": str(sd) if sd else None,
            "encoded_path": str(ep) if ep else None,
        }
    )


async def handle_recorder_start(request: web.Request) -> web.Response:
    """Start a new recording session.

    Request body (optional JSON):
        ``{"start_step": 0}``

    Returns
    -------
    JSON
        ``{"status": "ok", "session": "session_20260101_120000",
          "message": "Recording started"}``
    ``{"status": "error", "message": "…"}`` on failure.
    """
    recorder = _get_recorder(request)

    if recorder.is_recording:
        return web.json_response(
            {"status": "error", "message": "Recording already in progress"},
            status=409,
        )

    # Parse optional body
    start_step = 0
    if request.can_read_body:
        try:
            body = await request.json()
            start_step = int(body.get("start_step", 0))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    try:
        session = recorder.start(start_step=start_step)
    except RuntimeError as exc:
        return web.json_response(
            {"status": "error", "message": str(exc)},
            status=409,
        )

    return web.json_response(
        {
            "status": "ok",
            "session": session,
            "message": "Recording started",
        }
    )


async def handle_recorder_stop(request: web.Request) -> web.Response:
    """Stop the current recording session and optionally encode MP4.

    Request body (optional JSON):
        ``{"encode_mp4": true}``   (default: true)

    Returns
    -------
    JSON
        ``{"status": "ok", "session": "…", "frames": N,
          "duration_s": 12.3, "encoded": bool,
          "encoded_path": "…" | null}``
    """
    recorder = _get_recorder(request)

    if not recorder.is_recording:
        return web.json_response(
            {"status": "error", "message": "No recording in progress"},
            status=409,
        )

    encode_mp4 = True
    if request.can_read_body:
        try:
            body = await request.json()
            encode_mp4 = bool(body.get("encode_mp4", True))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    summary = recorder.stop(encode_mp4=encode_mp4)
    summary["status"] = "ok"
    return web.json_response(summary)


async def handle_recorder_files(request: web.Request) -> web.Response:
    """List all completed recording sessions.

    Returns
    -------
    JSON
        ``{"status": "ok", "sessions": [{…}, …]}``
    """
    try:
        recorder = _get_recorder(request)
    except web.HTTPException:
        return web.json_response({"status": "ok", "sessions": []})

    sessions = recorder.list_recordings()
    return web.json_response({"status": "ok", "sessions": sessions})


# ===================================================================
# WebSocket handler
# ===================================================================


async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
    """WebSocket endpoint for live frame relay, metrics, and joypad input.

    Server → Client
    ---------------
    - Binary (0x00 + PNG data) — live frame
    - Text (JSON) — ``{"type": "metrics", "data": {…}}``
    - Text (JSON) — ``{"type": "event", "event": "…", …}``

    Client → Server
    ---------------
    - Text (JSON) — ``{"type": "joypad", "state": {"UP": true, …}}``
    - Text (JSON) — ``{"type": "command", "command": "train_start"}``
                     or ``{"type": "command", "command": "train_stop"}``
    """
    ws = web.WebSocketResponse(max_msg_size=0)  # no size limit for frames
    await ws.prepare(request)

    hub: BroadcastHub = request.app["hub"]
    hub.register(ws)

    logger.info("WebSocket client connected (%d total)", hub.num_clients)

    # Send cached state immediately
    latest_frame = hub.get_latest_frame()
    if latest_frame is not None:
        with contextlib.suppress(Exception):
            await ws.send_bytes(b"\x00" + latest_frame)

    latest_metrics = hub.get_latest_metrics()
    if latest_metrics:
        with contextlib.suppress(Exception):
            await ws.send_str(json.dumps({"type": "metrics", "data": latest_metrics}))

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                if not isinstance(data, dict):
                    continue

                msg_type = data.get("type", "")

                if msg_type == "joypad":
                    hub.set_joypad_state(data.get("state", {}))

                elif msg_type == "command":
                    cmd = data.get("command", "")
                    if cmd == "train_start":
                        hub.train_requested = True
                        await hub.broadcast_event("train_start", msg="Training requested")
                    elif cmd == "train_stop":
                        hub.stop_requested = True
                        await hub.broadcast_event("train_stop", msg="Stop requested")

            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error("WebSocket error: %s", ws.exception())

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.debug("WebSocket disconnected: %s", exc)
    finally:
        hub.unregister(ws)
        logger.info("WebSocket client disconnected (%d remaining)", hub.num_clients)

    return ws


# ===================================================================
# DashboardServer — runs in a background daemon thread
# ===================================================================


class DashboardServer:
    """aiohttp server running in a background daemon thread.

    Usage
    -----
        server = DashboardServer(host="127.0.0.1", port=8765)
        server.start(config=cfg, recorder=recorder)

        # In training loop (synchronous):
        server.broadcast_frame(png_bytes)
        server.broadcast_metrics({"step": 1000, "return": 42.5})
        if server.hub.stop_requested:
            break

        # Recording:
        if recorder.is_recording:
            recorder.record_frame(raw_frame, metadata={...})

        # Shutdown:
        server.stop()

    Parameters
    ----------
    host : str
        Bind address.
    port : int
        Bind port.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.host = host
        self.port = port
        self.hub = BroadcastHub()
        self.recorder: Recorder | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None
        self._started = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, config: Any = None, recorder: Recorder | None = None) -> None:
        """Start the server in a daemon thread (non-blocking)."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("DashboardServer is already running")
            return

        self.recorder = recorder

        self._thread = threading.Thread(
            target=self._run_event_loop,
            args=(config, recorder),
            daemon=True,
            name="gbagent-dashboard",
        )
        self._thread.start()
        self._started.wait(timeout=10)  # wait for startup

    def stop(self) -> None:
        """Shut down the server gracefully."""
        if self._loop is not None and not self._loop.is_closed():
            if self._runner is not None:
                fut = asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop)
                with contextlib.suppress(Exception):
                    fut.result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------------
    # Broadcasting (thread-safe, call from training loop)
    # ------------------------------------------------------------------

    def broadcast_frame(self, png_bytes: bytes) -> None:
        """Send a live frame to all connected WebSocket clients."""
        if self._loop is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.hub.broadcast_frame(png_bytes), self._loop)

    def broadcast_metrics(self, data: dict[str, float]) -> None:
        """Send metrics to all connected WebSocket clients."""
        if self._loop is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.hub.broadcast_metrics(data), self._loop)

    def broadcast_event(self, event: str, **kwargs: Any) -> None:
        """Send an event to all connected WebSocket clients."""
        if self._loop is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.hub.broadcast_event(event, **kwargs), self._loop)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_event_loop(self, config: Any, recorder: Recorder | None = None) -> None:
        """Daemon thread entry point."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_server(config, recorder))
        except Exception as exc:
            logger.error("Failed to start dashboard server: %s", exc)
            self._started.set()
            return
        self._started.set()
        try:
            self._loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._loop.close()

    async def _start_server(self, config: Any, recorder: Recorder | None = None) -> None:
        """Create the app and start the TCPSite."""
        app = create_app(hub=self.hub, config=config, recorder=recorder)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(
            "Dashboard server running on http://%s:%d",
            self.host,
            self.port,
        )
