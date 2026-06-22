"""GAE rollout buffer for PPO with factorised (dpad, btn) actions.

Stores one rollout of *n_steps* × *num_envs* transitions and computes
Generalized Advantage Estimation (GAE) once the rollout is complete.

Usage
-----
    buffer = RolloutBuffer(num_envs=8, n_steps=128, obs_shape=(84, 84, 4))

    for step in range(n_steps):
        buffer.store(obs, dpad_a, btn_a, reward, done, log_prob, value)

    buffer.compute_gae(last_value, gamma=0.99, gae_lambda=0.95)

    for batch in buffer.get_batches(batch_size=256):
        obs_b, dpad_b, btn_b, ret_b, adv_b, old_lp_b = batch
        # … PPO update …
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class RolloutBuffer:
    """Fixed-size rollout buffer with GAE.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments.
    n_steps : int
        Number of environment steps per rollout.
    obs_shape : tuple[int, int, int]
        Observation shape excluding batch dimension, e.g. ``(84, 84, 4)``.
    """

    def __init__(self, num_envs: int, n_steps: int, obs_shape: tuple[int, int, int]):
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.obs_shape = obs_shape
        self.clear()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Zero out all storage arrays and reset the step counter."""
        shape = (self.n_steps, self.num_envs)
        self.obs: np.ndarray = np.zeros(
            (self.n_steps, self.num_envs, *self.obs_shape), dtype=np.float32
        )
        self.dpad_actions: np.ndarray = np.zeros(shape, dtype=np.int32)
        self.btn_actions: np.ndarray = np.zeros(shape, dtype=np.int32)
        self.rewards: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.dones: np.ndarray = np.zeros(shape, dtype=bool)
        self.log_probs: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.values: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.advantages: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.returns: np.ndarray = np.zeros(shape, dtype=np.float32)
        self._step = 0

    def store(
        self,
        obs: np.ndarray,           # (num_envs, 84, 84, 4)
        dpad_action: np.ndarray,    # (num_envs,)  int
        btn_action: np.ndarray,     # (num_envs,)  int
        reward: np.ndarray,         # (num_envs,)  float
        done: np.ndarray,           # (num_envs,)  bool
        log_prob: np.ndarray,       # (num_envs,)  float
        value: np.ndarray,          # (num_envs,)  float
    ) -> None:
        """Store one step of data from *num_envs* parallel environments."""
        idx = self._step
        self.obs[idx] = obs
        self.dpad_actions[idx] = dpad_action
        self.btn_actions[idx] = btn_action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self._step += 1

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_gae(self, last_value: np.ndarray, gamma: float,
                    gae_lambda: float) -> None:
        """Compute advantages via Generalized Advantage Estimation.

        Populates ``self.advantages`` and ``self.returns`` in-place.

        Parameters
        ----------
        last_value : (num_envs,) array
            Value estimate for the observation *after* the last stored step.
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE trace-decay parameter (λ).
        """
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]
            nonterminal = 1.0 - self.dones[t].astype(np.float32)

            delta = (
                self.rewards[t]
                + gamma * next_value * nonterminal
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * nonterminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def get_batches(
        self, batch_size: int
    ) -> Iterator[tuple[np.ndarray, ...]]:
        """Yield shuffled mini-batches for PPO update.

        Each batch is a tuple of:
            (obs, dpad_actions, btn_actions, returns, advantages, log_probs)
        """
        total = self.n_steps * self.num_envs
        indices = np.random.permutation(total)

        # Flatten all arrays from (T, N, …) → (T*N, …)
        def _flat(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(total, *arr.shape[2:])

        obs_flat = _flat(self.obs)
        dpad_flat = _flat(self.dpad_actions)
        btn_flat = _flat(self.btn_actions)
        ret_flat = _flat(self.returns)
        adv_flat = _flat(self.advantages)
        lp_flat = _flat(self.log_probs)

        # Normalise advantages per batch (standard trick for stability)
        adv_mean = adv_flat.mean()
        adv_std = adv_flat.std() + 1e-8
        adv_flat = (adv_flat - adv_mean) / adv_std

        start = 0
        while start < total:
            end = start + batch_size
            idx = indices[start:end]
            yield (
                obs_flat[idx],
                dpad_flat[idx],
                btn_flat[idx],
                ret_flat[idx],
                adv_flat[idx],
                lp_flat[idx],
            )
            start = end

    def __len__(self) -> int:
        return self.n_steps

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}"
            f"(n_steps={self.n_steps}, num_envs={self.num_envs}, "
            f"obs_shape={self.obs_shape}, step={self._step})"
        )
