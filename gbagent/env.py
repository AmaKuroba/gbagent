"""Stable-Retro environment wrapper with frame stack, frame skip,
GBA L/R shoulder button support, and action discretisation."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import cv2
import gymnasium as gym
import numpy as np
import retro
import stable_retro.data as retro_data

# Track already-registered custom ROM directories (process-wide)
_registered_rom_dirs: set[str] = set()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBS_SHAPE = (84, 84, 1)  # height, width, channels (grayscale)
FRAME_SKIP = 4
FRAME_STACK = 4

# Standard retro button ordering (canonical positions)
BUTTON_ORDER: list[str] = [
    "B",
    "A",
    "MODE",
    "SELECT",
    "START",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "C",
    "Y",
    "X",
    "Z",
    "L",
    "R",
    "L2",
    "R2",
    "L3",
    "R3",
]

# ---------------------------------------------------------------------------
# Canonical action table builders
# ---------------------------------------------------------------------------

# Game Boy (B, A, SELECT, START, UP, DOWN, LEFT, RIGHT) → 12 actions
GB_ACTION_TABLE: list[list[str]] = [
    [],                  # 0: NOOP
    ["B"],               # 1: B
    ["A"],               # 2: A
    ["SELECT"],          # 3: SELECT
    ["START"],           # 4: START
    ["UP"],              # 5: UP
    ["DOWN"],            # 6: DOWN
    ["LEFT"],            # 7: LEFT
    ["RIGHT"],           # 8: RIGHT
    ["A", "B"],          # 9: A+B
    ["LEFT", "RIGHT"],   # 10: LEFT+RIGHT (neutral)
    ["UP", "DOWN"],      # 11: UP+DOWN
]

# GBA (B, A, SELECT, START, UP, DOWN, LEFT, RIGHT, L, R) → 14 actions
GBA_ACTION_TABLE: list[list[str]] = [
    [],                  # 0: NOOP
    ["B"],               # 1: B
    ["A"],               # 2: A
    ["SELECT"],          # 3: SELECT
    ["START"],           # 4: START
    ["UP"],              # 5: UP
    ["DOWN"],            # 6: DOWN
    ["LEFT"],            # 7: LEFT
    ["RIGHT"],           # 8: RIGHT
    ["L"],               # 9: Left shoulder
    ["R"],               # 10: Right shoulder
    ["A", "B"],          # 11: A+B
    ["LEFT", "RIGHT"],   # 12: LEFT+RIGHT (neutral)
    ["UP", "DOWN"],      # 13: UP+DOWN
]


def _build_discrete_actions(
    buttons: Sequence[str], gba_mode: bool = False
) -> list[list[bool]]:
    """Build a discrete action table from available button names.

    Uses canonical tables for GB and GBA, filtering out any actions
    that reference buttons not present in *buttons*.

    Parameters
    ----------
    buttons : sequence of str
        Button names available from the Retro environment.
    gba_mode : bool
        If True, use the 14-action GBA table (adds L, R shoulder buttons).

    Returns
    -------
    list[list[bool]]
        Boolean mask per action, suitable for ``env.step()``.
    """
    b_map = {name: i for i, name in enumerate(BUTTON_ORDER)}

    # Map button name to its index in the env's button array
    env_idx = {name: b_map[name] for name in buttons if name in b_map}

    table = GBA_ACTION_TABLE if gba_mode else GB_ACTION_TABLE

    out: list[list[bool]] = []
    for action_name_list in table:
        mask = [False] * len(buttons)
        for name in action_name_list:
            if name in env_idx:
                mask[env_idx[name]] = True
        out.append(mask)

    return out


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert RGB frame → grayscale, resize to 84×84, normalise to [0, 1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (OBS_SHAPE[1], OBS_SHAPE[0]), interpolation=cv2.INTER_AREA)
    return resized[..., np.newaxis].astype(np.float32) / 255.0  # (84, 84, 1)


# ---------------------------------------------------------------------------
# GBAGEnv — main env wrapper
# ---------------------------------------------------------------------------


class GBAGEnv(gym.Env):
    """Gymnasium wrapper around a Stable-Retro Game Boy / GBA environment.

    Features
    --------
    * Action discretisation via canonical GB or GBA action tables.
    * Frame-skip (*n* input steps per logical step; repeat action).
    * Frame-stack – last *k* observations stacked on channel dimension.
    * RGB → grayscale → 84×84 preprocessing.
    * Reward clipping to [-1, 1] for stability.
    * GBA mode adds L/R shoulder buttons to the action space.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ("human", "rgb_array"),
        "render_fps": 60,
    }

    # ------------------------------------------------------------------
    def __init__(
        self,
        game: str = "PokemonRed-GB",
        state: str = "Level1",
        frame_skip: int = FRAME_SKIP,
        frame_stack: int = FRAME_STACK,
        render_mode: str | None = None,
        gba_mode: bool = False,
        rom_dir: str | None = "roms",
        **scenario_kwargs: Any,
    ) -> None:
        super().__init__()

        self._game = game
        self._frame_skip = frame_skip
        self._frame_stack = frame_stack
        self._stack: deque[np.ndarray] = deque(maxlen=frame_stack)
        self._render_mode = render_mode

        # Register custom ROM directory and build integration type.
        # We keep the default inttype (STABLE) and layer CUSTOM_ONLY on top
        # so built-in game data (data.json, scenario.json) is found in the
        # stable path while the ROM is resolved from the custom directory.
        inttype = retro_data.Integrations.STABLE
        if rom_dir is not None:
            resolved = Path(rom_dir).resolve()
            if resolved.is_dir() and str(resolved) not in _registered_rom_dirs:
                _registered_rom_dirs.add(str(resolved))
                retro_data.Integrations.add_custom_path(str(resolved))
                inttype |= retro_data.Integrations.CUSTOM_ONLY
                print(f"  ROM dir: {resolved}")

        # Resolve state: if the file doesn't exist, fall back to power-on default
        # and auto-create the state file so it's available next time.
        state_to_use: retro.State | str = state
        auto_create_state = False
        if isinstance(state, str) and state not in ("__default__", "__none__"):
            if retro_data.get_file_path(game, f"{state}.state", inttype) is None:
                print(f"  ⚠ State '{state}' not found for '{game}', using power-on default")
                state_to_use = retro.State.DEFAULT
                auto_create_state = True

        # Create the underlying Retro environment
        self._env = retro.make(
            game=game,
            state=state_to_use,
            inttype=inttype,
            use_restricted_actions=retro.Actions.DISCRETE,
            render_mode=render_mode if render_mode == "human" else None,
            **scenario_kwargs,
        )

        # Auto-create the requested state file from the default state
        if auto_create_state:
            self._auto_create_state(game, state, rom_dir)

        # GBA mode enables L/R shoulder buttons
        self._gba_mode = gba_mode or self._detect_gba()

        # Build our discrete action space from available buttons
        self._buttons = self._env.buttons
        self._discrete_actions = _build_discrete_actions(self._buttons, gba_mode=self._gba_mode)
        self.action_space = gym.spaces.Discrete(len(self._discrete_actions))

        # Observation space: stack * 1-channel frames stacked on last axis
        obs_channels = frame_stack * OBS_SHAPE[-1]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_SHAPE[0], OBS_SHAPE[1], obs_channels),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed)

        # Initialise frame stack with the first preprocessed frame * k times
        preproc = _preprocess_frame(obs)
        self._stack.clear()
        for _ in range(self._frame_stack):
            self._stack.append(preproc)

        return self._stacked_obs(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply the discrete action for ``frame_skip`` steps.

        Returns the stacked observation, summed clipped reward, and terminal
        signals from the *last* environment step.
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        action_mask = self._discrete_actions[action]
        total_reward = 0.0
        terminated = truncated = False
        info: dict[str, Any] = {}

        for _ in range(self._frame_skip):
            obs, reward, terminated, truncated, info = self._env.step(action_mask)
            total_reward += reward
            preproc = _preprocess_frame(obs)
            self._stack.append(preproc)
            if terminated or truncated:
                break

        # Clip total reward per step
        total_reward = np.clip(total_reward, -1.0, 1.0)

        return self._stacked_obs(), total_reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self._render_mode == "rgb_array":
            return self._env.render(mode="rgb_array")
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_gba(self) -> bool:
        """Detect GBA games by checking for L/R buttons in the env."""
        return "L" in self._buttons or "R" in self._buttons

    def _auto_create_state(self, game: str, state: str, rom_dir: str | None) -> None:
        """Save the current (default) emulator state as the requested state file."""
        if rom_dir is not None:
            save_dir = Path(rom_dir).resolve() / game
        else:
            save_dir = Path(retro_data.path()) / "stable" / game
        save_dir.mkdir(parents=True, exist_ok=True)
        state_path = save_dir / f"{state}.state"
        self._env.save_state(str(state_path))
        print(f"  ✓ Created default state: {state_path}")

    def _stacked_obs(self) -> np.ndarray:
        """Concatenate the frame stack on the last axis → (*H*, *W*, *C*·*k*)."""
        return np.concatenate(list(self._stack), axis=-1)

    @property
    def raw_env(self) -> retro.RetroEnv:
        """Access the underlying ``retro.RetroEnv`` for diagnostics."""
        return self._env

    @property
    def buttons(self) -> tuple[str, ...]:
        """Buttons available in this game (read-only)."""
        return tuple(self._buttons)

    @property
    def discrete_actions(self) -> list[list[bool]]:
        """Return the full discrete action table (read-only)."""
        return list(self._discrete_actions)

    @property
    def gba_mode(self) -> bool:
        """Whether this env is using GBA-style action space (with L/R)."""
        return self._gba_mode
