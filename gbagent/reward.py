"""Shaped rewards — screen novelty, stale penalty, and YAML-defined RAM scanners.

RewardSystem
============
Augments the environment's native reward with three bonus components:

1. **Screen novelty** – reward proportional to the mean absolute pixel
   difference between consecutive preprocessed frames.  Encourages the agent
   to move around and change the screen content.

2. **Stale penalty** – a negative reward applied each step when the screen
   has been nearly static for a configurable number of consecutive steps.
   Discourages the agent from getting stuck.

3. **RAM scanners** – user-defined watchers on Game Boy memory addresses.
   Three scanner types are supported:

   - ``discrete``: reward when RAM[address] equals *target_value*.
   - ``delta``: reward when the value at *address* changes in a specific
     direction (positive, negative, or any).
   - ``multi_byte``: combine *n_bytes* consecutive bytes into a single
     integer (little- or big-endian) and compare against *target_value*.

Scanner state is tracked step-to-step, so a ``discrete`` scanner only
rewards the *first* step it sees a given target value (avoiding repeated
rewards for holding a value).

Usage
-----
    reward_system = RewardSystem(env, cfg.reward.to_dict())
    obs, env_reward, term, trunc, info = env.step(action)
    extra = reward_system.step(obs)
    total = env_reward + extra
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Scanner state keys
# ---------------------------------------------------------------------------

_ST_PREV_VALUE = "prev_value"
_ST_LAST_REWARDED = "last_rewarded"


# ---------------------------------------------------------------------------
# RewardSystem
# ---------------------------------------------------------------------------


class RewardSystem:
    """Compute per-step shaped rewards from screen and RAM signals.

    Parameters
    ----------
    env : object
        Wrapped environment.  Must expose a ``raw_env`` attribute that
        in turn has a ``get_ram()`` method returning a numpy array of
        uint8 RAM bytes (as provided by ``stable-retro``).
    config : dict
        Reward sub-section of the YAML config (see ``configs/default.yaml``).
        Expected keys:

        * ``screen_novelty_scale`` — multiplier for pixel-diff novelty
          (default ``0.0`` → disabled).
        * ``stale_penalty`` — negative reward per stale step
          (default ``0.0`` → disabled).
        * ``stale_threshold`` — consecutive non-novel steps before penalty
          kicks in (default ``100``).
        * ``stale_diff_threshold`` — MSE threshold for "novel" vs "stale"
          (default ``0.001``).
        * ``ram_scanners`` — list of scanner dicts (default ``[]``).
    """

    def __init__(self, env, config: dict[str, Any]) -> None:
        self._env = env

        # Screen novelty
        self._novelty_scale = float(config.get("screen_novelty_scale", 0.0))

        # Stale penalty
        self._stale_penalty = float(config.get("stale_penalty", 0.0))
        self._stale_threshold = int(config.get("stale_threshold", 100))
        self._stale_diff_threshold = float(config.get("stale_diff_threshold", 0.001))

        # RAM scanner definitions
        self._scanners: list[dict[str, Any]] = config.get("ram_scanners", [])

        # ── Reset-able state ─────────────────────────────────────────
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, obs: np.ndarray) -> float:
        """Compute the shaped reward bonus for the current step.

        Parameters
        ----------
        obs : np.ndarray
            Preprocessed observation from :class:`GBAGEnv` (84, 84, *C*).
            The most recent frame (channel -1) is used for pixel diff.

        Returns
        -------
        float
            Bonus reward to **add** to the environment's native reward.
        """
        bonus = 0.0

        # Extract the latest single frame from the stacked observation
        #   obs shape: (84, 84, 4)  →  latest frame at obs[:, :, -1]
        frame = obs[:, :, -1] if obs.ndim == 3 else obs

        bonus += self._screen_novelty(frame)
        bonus += self._stale_penalty_step(frame)
        bonus += self._ram_scanner_step()

        # Save current frame as previous for next step
        self._prev_frame = frame.copy()

        return float(bonus)

    def reset(self) -> None:
        """Clear all internal state (call when the environment resets)."""
        self._prev_frame: np.ndarray | None = None
        self._stale_steps = 0
        self._scanner_state: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Screen novelty
    # ------------------------------------------------------------------

    def _screen_novelty(self, frame: np.ndarray) -> float:
        """Reward proportional to the mean absolute pixel change.

        Returns ``novelty_scale * mean_diff``, or 0 on the first frame.
        """
        if self._novelty_scale == 0.0 or self._prev_frame is None:
            return 0.0

        diff = float(np.mean(np.abs(frame - self._prev_frame)))
        return self._novelty_scale * diff

    # ------------------------------------------------------------------
    # Stale penalty
    # ------------------------------------------------------------------

    def _stale_penalty_step(self, frame: np.ndarray) -> float:
        """Apply a negative reward when the screen is static too long.

        The "staleness" counter increments when the per-pixel MSE between
        consecutive frames falls below *stale_diff_threshold*, and resets
        when it rises above.
        """
        if self._stale_penalty == 0.0:
            return 0.0

        if self._prev_frame is not None:
            diff = float(np.mean(np.abs(frame - self._prev_frame)))
            if diff < self._stale_diff_threshold:
                self._stale_steps += 1
            else:
                self._stale_steps = 0

        # Serve penalty only after the threshold is exceeded
        if self._stale_steps > self._stale_threshold:
            return self._stale_penalty

        return 0.0

    # ------------------------------------------------------------------
    # RAM scanners
    # ------------------------------------------------------------------

    def _ram_scanner_step(self) -> float:
        """Evaluate all YAML-defined RAM scanners and return total bonus."""
        if not self._scanners:
            return 0.0

        ram = self._env.raw_env.get_ram()
        total = 0.0

        for scanner in self._scanners:
            stype = scanner.get("type", "discrete")
            name = scanner.get("name", "")
            addr = scanner.get("address", 0)
            reward_val = float(scanner.get("reward", 1.0))

            if stype == "discrete":
                total += self._scan_discrete(name, ram, addr, scanner, reward_val)
            elif stype == "delta":
                total += self._scan_delta(name, ram, addr, scanner, reward_val)
            elif stype == "multi_byte":
                total += self._scan_multi_byte(name, ram, addr, scanner, reward_val)
            elif stype == "delta_multi":
                total += self._scan_delta_multi(name, ram, addr, scanner, reward_val)

        return total

    # ------------------------------------------------------------------
    # discrete scanner
    # ------------------------------------------------------------------

    def _scan_discrete(
        self,
        name: str,
        ram: np.ndarray,
        addr: int,
        scanner: dict[str, Any],
        reward_val: float,
    ) -> float:
        """Reward when RAM[addr] matches *target_value* (once per hit)."""
        target = int(scanner.get("target_value", 0))

        try:
            current = int(ram[addr])
        except IndexError:
            return 0.0

        state = self._scanner_state.setdefault(name, {_ST_PREV_VALUE: -1, _ST_LAST_REWARDED: -1})

        if current == target and current != state[_ST_LAST_REWARDED]:
            state[_ST_LAST_REWARDED] = current
            state[_ST_PREV_VALUE] = current
            return reward_val

        if current != state[_ST_LAST_REWARDED]:
            state[_ST_LAST_REWARDED] = -1  # allow re-reward if we come back

        state[_ST_PREV_VALUE] = current
        return 0.0

    # ------------------------------------------------------------------
    # delta scanner
    # ------------------------------------------------------------------

    def _scan_delta(
        self,
        name: str,
        ram: np.ndarray,
        addr: int,
        scanner: dict[str, Any],
        reward_val: float,
    ) -> float:
        """Reward when the value at *addr* changes in a given direction.

        *delta_sign* can be ``"positive"``, ``"negative"``, or ``"any"``.
        """
        direction = scanner.get("delta_sign", "positive")

        try:
            current = int(ram[addr])
        except IndexError:
            return 0.0

        state = self._scanner_state.setdefault(name, {_ST_PREV_VALUE: -1})

        if state[_ST_PREV_VALUE] < 0:
            state[_ST_PREV_VALUE] = current
            return 0.0

        delta = current - state[_ST_PREV_VALUE]
        state[_ST_PREV_VALUE] = current

        if direction == "positive" and delta > 0:
            return reward_val
        if direction == "negative" and delta < 0:
            return reward_val
        if direction == "any" and delta != 0:
            return reward_val
        return 0.0

    # ------------------------------------------------------------------
    # multi_byte scanner
    # ------------------------------------------------------------------

    def _scan_multi_byte(
        self,
        name: str,
        ram: np.ndarray,
        addr: int,
        scanner: dict[str, Any],
        reward_val: float,
    ) -> float:
        """Decode *n_bytes* at *addr* and reward on target match (once)."""
        n_bytes = int(scanner.get("n_bytes", 2))
        byte_order = scanner.get("byte_order", "little")
        target = int(scanner.get("target_value", 0))

        try:
            raw = ram[addr : addr + n_bytes]
            if len(raw) < n_bytes:
                return 0.0
            current = int.from_bytes(bytes(raw), byteorder=byte_order, signed=False)
        except (IndexError, OverflowError, ValueError):
            return 0.0

        state = self._scanner_state.setdefault(name, {_ST_PREV_VALUE: -1, _ST_LAST_REWARDED: -1})

        if current == target and current != state[_ST_LAST_REWARDED]:
            state[_ST_LAST_REWARDED] = current
            state[_ST_PREV_VALUE] = current
            return reward_val

        if current != state[_ST_LAST_REWARDED]:
            state[_ST_LAST_REWARDED] = -1  # re-allow if we leave target
            state[_ST_PREV_VALUE] = current

        return 0.0

    # ------------------------------------------------------------------
    # Diagnostics / query helpers
    # ------------------------------------------------------------------

    @property
    def scanner_states(self) -> dict[str, Any]:
        """Return current scanner state dict (read-only)."""
        return dict(self._scanner_state)

    @property
    def stale_steps(self) -> int:
        """Current consecutive-stale count."""
        return self._stale_steps

    @property
    def is_stale(self) -> bool:
        """Whether the stale penalty is currently active."""
        return self._stale_steps > self._stale_threshold


    # ------------------------------------------------------------------
    # delta_multi scanner
    # ------------------------------------------------------------------

    def _scan_delta_multi(
        self,
        name: str,
        ram: np.ndarray,
        addr: int,
        scanner: dict[str, Any],
        reward_val: float,
    ) -> float:
        """Reward proportional to the absolute change of a multi-byte value.

        *n_bytes* consecutive bytes at *addr* are decoded as an unsigned
        integer (little- or big-endian) and the absolute difference from
        the previous step is multiplied by *reward_per_unit* (default 1.0).
        """
        n_bytes = int(scanner.get("n_bytes", 2))
        byte_order = scanner.get("byte_order", "little")
        reward_per_unit = float(scanner.get("reward_per_unit", reward_val))

        try:
            raw = ram[addr : addr + n_bytes]
            if len(raw) < n_bytes:
                return 0.0
            current = int.from_bytes(bytes(raw), byteorder=byte_order, signed=False)
        except (IndexError, OverflowError, ValueError):
            return 0.0

        state = self._scanner_state.setdefault(name, {_ST_PREV_VALUE: -1})

        if state[_ST_PREV_VALUE] < 0:
            state[_ST_PREV_VALUE] = current
            return 0.0

        delta = abs(current - state[_ST_PREV_VALUE])
        state[_ST_PREV_VALUE] = current
        return reward_per_unit * delta


__all__ = ["RewardSystem"]
