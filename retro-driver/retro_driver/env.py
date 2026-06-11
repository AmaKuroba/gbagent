"""GBEnv: Gymnasium environment wrapping gbagent via WebSocket JSON-RPC.

Observation: 4 stacked grayscale frames (144x160 each, uint8)
Action: MultiDiscrete -- dpad (5) x buttons (4)
Reward: Screen novelty baseline + optional RAM scanners
"""

from __future__ import annotations

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

        # Load the emulator's saved start state (--load-state on Go side)
        # instead of hard-resetting. The model never calls client.reset().
        self.client.reset_state()
        self.client.wait_frames(4)

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


class SyncVectorGBEnv:
    """Synchronous vectorized environment for PPO training.

    Runs multiple GBEnv instances in parallel, each connected to a separate
    emulator process. This enables efficient rollout collection for PPO.
    """

    def __init__(
        self,
        num_envs: int,
        gbagent_urls: list[str] | None = None,
        frame_stack: int = 4,
        frame_skip: int = 4,
        reward_config: RewardConfig | None = None,
        max_steps: int = 10_000,
    ):
        self.num_envs = num_envs

        # Default URLs: localhost:8767, 8768, 8769, ...
        if gbagent_urls is None:
            gbagent_urls = [f"ws://localhost:{8767 + i}/ws" for i in range(num_envs)]

        self.envs = [
            GBEnv(
                gbagent_url=gbagent_urls[i],
                frame_stack=frame_stack,
                frame_skip=frame_skip,
                reward_config=reward_config,
                max_steps=max_steps,
            )
            for i in range(num_envs)
        ]

        # Observation and action spaces match single env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset all environments.

        Returns:
            observations: (num_envs, C, H, W) stacked frames
            infos: List of info dicts
        """
        obs_list = []
        info_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)

        return np.stack(obs_list), {"infos": info_list}

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step all environments.

        Args:
            actions: (num_envs, 2) array of (dpad, button) actions

        Returns:
            observations: (num_envs, C, H, W)
            rewards: (num_envs,)
            terminateds: (num_envs,)
            truncateds: (num_envs,)
            infos: Dict with 'frame_gids' and 'reward_breakdowns'
        """
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        frame_gid_list = []
        breakdown_list = []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            frame_gid_list.append(info.get("frame_gid", 0))
            breakdown_list.append(info.get("reward_breakdown", {}))

        return (
            np.stack(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(terminated_list, dtype=bool),
            np.array(truncated_list, dtype=bool),
            {
                "frame_gids": frame_gid_list,
                "reward_breakdowns": breakdown_list,
            },
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()
