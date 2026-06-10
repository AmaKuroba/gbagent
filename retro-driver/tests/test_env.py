from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from retro_driver.env import GBEnv, DPAD_MAP, BTN_MAP


@pytest.fixture
def mock_ws_client():
    """Patch GBWSClient so tests don't need a real emulator."""
    with patch("retro_driver.env.GBWSClient") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance

        from PIL import Image

        instance.get_screen.return_value = Image.fromarray(
            np.zeros((144, 160), dtype=np.uint8), mode="L"
        )
        yield instance


@pytest.fixture
def env(mock_ws_client):
    return GBEnv(gbagent_url="ws://localhost:8767/ws")


class TestEnvInitialization:
    def test_defaults(self, env):
        assert env.observation_space.shape == (4, 144, 160)
        assert env.action_space.shape == (2,)

    def test_frame_stack(self, mock_ws_client):
        e = GBEnv(gbagent_url="ws://localhost:8767/ws", frame_stack=8)
        assert e.frame_stack == 8
        assert e.observation_space.shape == (8, 144, 160)

    def test_reset_returns_correct_shape(self, mock_ws_client):
        e = GBEnv(gbagent_url="ws://localhost:8767/ws")
        obs, info = e.reset()
        assert obs.shape == (4, 144, 160)
        assert obs.dtype == np.uint8


class TestStep:
    def test_args_direction(self, mock_ws_client):
        e = GBEnv(gbagent_url="ws://localhost:8767/ws")
        e._frames = [np.zeros((144, 160), dtype=np.uint8) for _ in range(4)]
        e._prev_frame = np.zeros((144, 160), dtype=np.uint8)
        assert DPAD_MAP[0] == ""
        assert BTN_MAP[0] == ""


class TestCrashDetection:
    def test_white_screen_detected(self):
        with patch("retro_driver.env.GBWSClient") as mock_cls:
            instance = MagicMock()
            mock_cls.return_value = instance
            from PIL import Image

            instance.get_screen.return_value = Image.fromarray(
                np.full((144, 160), 255, dtype=np.uint8), mode="L"
            )

            env = GBEnv(gbagent_url="ws://localhost:8767/ws")
            env._frames = [np.full((144, 160), 255, dtype=np.uint8) for _ in range(4)]
            env._prev_frame = np.full((144, 160), 255, dtype=np.uint8)
            # reset sets up frames via get_screen — for this test we mock
            obs, reward, terminated, truncated, info = env.step((0, 0))
            assert terminated  # white screen = crash

    def test_black_screen_detected(self):
        with patch("retro_driver.env.GBWSClient") as mock_cls:
            instance = MagicMock()
            mock_cls.return_value = instance
            from PIL import Image

            instance.get_screen.return_value = Image.fromarray(
                np.zeros((144, 160), dtype=np.uint8), mode="L"
            )

            env = GBEnv(gbagent_url="ws://localhost:8767/ws")
            env._frames = [np.zeros((144, 160), dtype=np.uint8) for _ in range(4)]
            env._prev_frame = np.zeros((144, 160), dtype=np.uint8)
            obs, reward, terminated, truncated, info = env.step((0, 0))
            assert terminated

    def test_frozen_frame_detected(self):
        with patch("retro_driver.env.GBWSClient") as mock_cls:
            instance = MagicMock()
            mock_cls.return_value = instance
            from PIL import Image

            frame = Image.fromarray(np.full((144, 160), 128, dtype=np.uint8), mode="L")
            instance.get_screen.return_value = frame

            env = GBEnv(gbagent_url="ws://localhost:8767/ws", frame_skip=1)
            env._frames = [np.full((144, 160), 128, dtype=np.uint8) for _ in range(4)]
            env._prev_frame = np.full((144, 160), 128, dtype=np.uint8)
            terminated = False
            for _ in range(30):
                _, _, terminated, _, _ = env.step((0, 0))
                if terminated:
                    break
            assert terminated
