"""HTTP server, WebSocket relay, and dashboard for gbagent.

Provides:
- ``BroadcastHub`` — thread-safe fan-out of frames + metrics to WS clients.
- ``DashboardServer`` — aiohttp server running in a background daemon thread.
- ``create_app`` — standalone aiohttp ``web.Application`` factory.
"""

from __future__ import annotations

from gbagent.config import RecorderConfig
from gbagent.recorder import Recorder
from gbagent.server.frames import frame_to_png
from gbagent.server.http import DashboardServer, create_app
from gbagent.server.hub import BroadcastHub

__all__ = [
    "BroadcastHub",
    "DashboardServer",
    "Recorder",
    "RecorderConfig",
    "create_app",
    "frame_to_png",
]
