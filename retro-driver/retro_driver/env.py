"""GBEnv: Gymnasium environment wrapping gbagent via WebSocket JSON-RPC.

Observation: 4 stacked grayscale frames (144x160 each, uint8)
Action: MultiDiscrete -- dpad (5) x buttons (4)
Reward: Screen novelty baseline + optional RAM scanners
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from retro_driver.reward import RewardConfig, RewardSystem
from retro_driver.ws_client import GBWSClient

if TYPE_CHECKING:
    from retro_driver.dqn import FrameStore


DPAD_MAP = ["", "up", "down", "left", "right"]
BTN_MAP = ["", "a", "b", "start", "select"]


class GBEnv(gym.Env):
    """Gymnasium environment for Game Boy games via gbagent WebSocket JSON-RPC.

    Connects to a running gbagent serve instance. The emulator runs at a
    constant 60 fps — the agent reads frames and presses buttons but never
    controls frame advancement.
    """

    metadata: ClassVar[dict[str, Any]] = {"render.modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        gbagent_url: str = "ws://localhost:8767/ws",
        frame_stack: int = 4,
        frame_skip: int = 4,
        reward_config: RewardConfig | None = None,
        max_steps: int = 10_000,
        frame_store: FrameStore | None = None,
        boot_frames: int = 60,
    ) -> None:
        super().__init__()

        self.gbagent_url = gbagent_url
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.max_steps = max_steps

        self.client = GBWSClient(gbagent_url)
        self.reward = RewardSystem(reward_config or RewardConfig())
        self.frame_store = frame_store

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.frame_stack, 144, 160),
            dtype=np.uint8,
        )

        self.action_space = spaces.MultiDiscrete([5, 4])

        self._frames: list[np.ndarray] = []
        self._step_count = 0
        self._prev_frame: np.ndarray | None = None
        self._frozen_steps = 0
        self._crash_white_threshold = 0.95
        self._crash_black_threshold = 0.95
        self._freeze_steps_limit = 20

    # ---- reset ---------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.client.start()
        self.client.reset()
        self.client.wait_frames(60)
        self._skip_intro()

        self._step_count = 0
        self._prev_frame = None
        self._frozen_steps = 0
        self.reward.reset()

        self._frames = []
        for _ in range(self.frame_stack):
            img = self._get_screen()
            frame = np.array(img, dtype=np.uint8)
            if self.frame_store is not None:
                self.frame_store.add(frame)
            self._frames.append(frame)
            self._prev_frame = frame
            self.client.wait_frames(1)

        first_gid = (
            self.frame_store.next_gid - self.frame_stack if self.frame_store is not None else 0
        )

        return self._stack(), {"frame_gid": first_gid}

    # ---- step ----------------------------------------------------------

    def step(
        self, action: np.ndarray | tuple[int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        dpad_idx = int(action[0])
        btn_idx = int(action[1])

        buttons: list[str] = []
        if dpad_idx > 0:
            buttons.append(DPAD_MAP[dpad_idx])
        if btn_idx > 0:
            buttons.append(BTN_MAP[btn_idx])

        for btn in buttons:
            self.client.press_button(btn)
        self.client.wait_frames(self.frame_skip)
        img = self.client.get_screen()
        self.client.release_all()

        frame = np.array(img, dtype=np.uint8)

        if self.frame_store is not None:
            new_gid = self.frame_store.add(frame)
            current_stack_start_gid = new_gid - (self.frame_stack - 1)
        else:
            current_stack_start_gid = 0

        self._frames.pop(0)
        self._frames.append(frame)

        step_reward = self.reward.compute(frame, self._prev_frame, self.client)
        self._step_count += 1

        terminated = self._detect_game_over(frame)
        truncated = self._step_count >= self.max_steps

        self._prev_frame = frame

        info = {
            "step": self._step_count,
            "frames": self._step_count * self.frame_skip,
            "frame_gid": current_stack_start_gid,
            "reward_breakdown": self.reward.last_breakdown(),
        }

        return self._stack(), step_reward, terminated, truncated, info

    # ---- render --------------------------------------------------------

    def render(self) -> np.ndarray | None:
        if self._frames:
            gray = self._frames[-1]
            return np.stack([gray, gray, gray], axis=-1)
        return None

    # ---- helpers -------------------------------------------------------

    def _get_screen(self) -> Image.Image:
        return self.client.get_screen()

    def _stack(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)

    def _skip_intro(self) -> None:
        """Script through the Pokemon Red/Blue intro (title, Oak, starter, rival)."""
        import time

        a = lambda: (self.client.press_button("A"), self.client.wait_frames(4), self.client.release_all())
        wait = lambda f: self.client.wait_frames(f)

        wait(120)  # let boot ROM finish

        # Title screen → press Start
        self.client.press_button("START")
        wait(30)
        self.client.release_all()
        wait(30)

        # Oak intro dialog + name selection: mash A generously
        for _ in range(100):
            a()

        # Walk down into the tall grass (triggers Oak to stop you)
        self.client.press_button("DOWN")
        wait(60)
        self.client.release_all()
        wait(180)  # Oak cutscene + walk to lab

        # Lab dialog: mash A through Oak's explanations
        for _ in range(80):
            a()

        # Select starter + more dialog
        for _ in range(40):
            a()

        # Walk up to exit
        self.client.press_button("UP")
        wait(90)
        self.client.release_all()
        wait(30)

    def _detect_game_over(self, frame: np.ndarray) -> bool:
        h, w = frame.shape
        total_pixels = h * w

        white_pixels = int((frame > 250).sum())
        if white_pixels / total_pixels >= self._crash_white_threshold:
            return True

        black_pixels = int((frame < 5).sum())
        if black_pixels / total_pixels >= self._crash_black_threshold:
            return True

        if self._prev_frame is not None and np.array_equal(frame, self._prev_frame):
            self._frozen_steps += 1
            if self._frozen_steps >= self._freeze_steps_limit:
                return True
        else:
            self._frozen_steps = 0

        return False

    def close(self) -> None:
        self.client.stop()
