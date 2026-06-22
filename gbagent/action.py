"""Action sampling utilities for factorised (dpad, btn) actions.

Supports both Game Boy (6 btn actions) and GBA (8 btn actions with L/R).

Flow
----
1. Model returns ``dpad_logits`` (B,5) and ``btn_logits`` (B,N) where N
   depends on the console (6 for GB, 8 for GBA).
2. ``sample_actions`` → ``(dpad_action, btn_action, log_prob)``.
3. ``combine_actions`` maps (dpad, btn) → single discrete action index for
   the env, using the correct lookup for the btn head size.
4. During PPO update, ``compute_log_probs`` recomputes log-probs from the
   current model weights for the stored actions.

All functions are NumPy-based and designed for eager-mode rollout loops.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from keras import ops

# ---------------------------------------------------------------------------
# Button head sizes
# ---------------------------------------------------------------------------

BTN_GB = 6    # NOOP, B, A, SELECT, START, A+B
BTN_GBA = 8   # NOOP, B, A, SELECT, START, A+B, L, R


def get_btn_size(gba_mode: bool = False) -> int:
    """Return the btn head size for the given console mode."""
    return BTN_GBA if gba_mode else BTN_GB


# ---------------------------------------------------------------------------
# Factorised (dpad, btn) → env discrete action mapping
# ---------------------------------------------------------------------------
#
# GB canonical action table (env.py):
#   0 NOOP   1 B    2 A    3 SELECT  4 START
#   5 UP     6 DOWN 7 LEFT 8 RIGHT
#   9 A+B   10 LR+ 11 UD+
#
# GBA canonical action table (env.py):
#   0 NOOP   1 B    2 A    3 SELECT  4 START
#   5 UP     6 DOWN 7 LEFT 8 RIGHT
#   9 L     10 R   11 A+B 12 LR+   13 UD+
#
# Factorised dpad (0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT)
# Factorised btn:
#   GB:  (0=NOOP, 1=B, 2=A, 3=SELECT, 4=START, 5=A+B)
#   GBA: (0=NOOP, 1=B, 2=A, 3=SELECT, 4=START, 5=A+B, 6=L, 7=R)

# GB mapping: btn_size=6 → actions 0-11
_LOOKUP_GB: dict[tuple[int, int], int] = {
    # btn only (dpad=NOOP)
    (0, 0): 0,   # NOOP
    (0, 1): 1,   # B
    (0, 2): 2,   # A
    (0, 3): 3,   # SELECT
    (0, 4): 4,   # START
    (0, 5): 9,   # A+B
    # dpad only (btn=NOOP)
    (1, 0): 5,   # UP
    (2, 0): 6,   # DOWN
    (3, 0): 7,   # LEFT
    (4, 0): 8,   # RIGHT
}

# GBA mapping: btn_size=8 → actions 0-13
_LOOKUP_GBA: dict[tuple[int, int], int] = {
    # btn only (dpad=NOOP)
    (0, 0): 0,   # NOOP
    (0, 1): 1,   # B
    (0, 2): 2,   # A
    (0, 3): 3,   # SELECT
    (0, 4): 4,   # START
    (0, 5): 11,  # A+B
    (0, 6): 9,   # L
    (0, 7): 10,  # R
    # dpad only (btn=NOOP)
    (1, 0): 5,   # UP
    (2, 0): 6,   # DOWN
    (3, 0): 7,   # LEFT
    (4, 0): 8,   # RIGHT
}

# Fallback when both dpad and btn are non-NOOP (dpad priority)
_DPAD_ONLY: dict[int, int] = {1: 5, 2: 6, 3: 7, 4: 8}


def _get_lookup(btn_size: int) -> dict[tuple[int, int], int]:
    """Return the correct combine lookup for the given btn head size."""
    if btn_size >= BTN_GBA:
        return _LOOKUP_GBA
    return _LOOKUP_GB


def combine_actions(
    dpad_action: np.ndarray,  # (B,) int in [0, 4]
    btn_action: np.ndarray,   # (B,) int in [0, 5] or [0, 7]
    btn_size: int = BTN_GB,
) -> np.ndarray:              # (B,) int in [0, 11] or [0, 13]
    """Map factorised (dpad, btn) actions to the env's discrete action index.

    When both dpad and btn are non-NOOP, dpad takes priority.

    Parameters
    ----------
    dpad_action : (B,) int
        Dpad action indices 0-4 (NOOP, UP, DOWN, LEFT, RIGHT).
    btn_action : (B,) int
        Button action indices; 0-5 for GB, 0-7 for GBA.
    btn_size : int
        Size of the btn head (6 for GB, 8 for GBA).
        Determines which action lookup table to use.

    Returns
    -------
    (B,) int
        Combined discrete action index.
    """
    lookup = _get_lookup(btn_size)
    result = np.empty_like(dpad_action)
    for i in range(len(dpad_action)):
        key = (int(dpad_action[i]), int(btn_action[i]))
        env_action = lookup.get(key)
        if env_action is not None:
            result[i] = env_action
        else:
            # Both non-NOOP → dpad priority
            result[i] = _DPAD_ONLY.get(int(dpad_action[i]), 0)
    return result


# ---------------------------------------------------------------------------
# NumPy softmax helpers
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------


def sample_actions(
    dpad_logits: Any,  # (B, 5)
    btn_logits: Any,   # (B, N)  N=6 (GB) or 8 (GBA)
    training: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample discrete actions from categorical (dpad, btn) logits.

    Parameters
    ----------
    dpad_logits : (B, 5) array
        Logits for the dpad head (NOOP, UP, DOWN, LEFT, RIGHT).
    btn_logits : (B, N) array, N=6 or 8
        Logits for the btn head.
        GB: NOOP, B, A, SELECT, START, A+B.
        GBA: NOOP, B, A, SELECT, START, A+B, L, R.
    training : bool
        *True* → stochastic sampling.
        *False* → greedy (argmax).

    Returns
    -------
    dpad_action : (B,) int
        Sampled dpad action indices.
    btn_action : (B,) int
        Sampled btn action indices.
    log_prob : (B,) float
        Joint log-probability ``log P(dpad) + log P(btn)``.
    """
    batch = dpad_logits.shape[0]

    # Convert to numpy if needed (handles Keras tensors / EagerTensors)
    if hasattr(dpad_logits, "numpy"):
        dpad_logits = dpad_logits.numpy()
    if hasattr(btn_logits, "numpy"):
        btn_logits = btn_logits.numpy()

    dpad_logits = np.asarray(dpad_logits, dtype=np.float64)
    btn_logits = np.asarray(btn_logits, dtype=np.float64)

    if training:
        dpad_probs = _softmax(dpad_logits, axis=-1)
        btn_probs = _softmax(btn_logits, axis=-1)
        dpad_action = np.array([
            np.random.choice(dpad_probs.shape[1], p=dpad_probs[i])
            for i in range(batch)
        ])
        btn_action = np.array([
            np.random.choice(btn_probs.shape[1], p=btn_probs[i])
            for i in range(batch)
        ])
    else:
        dpad_action = np.argmax(dpad_logits, axis=-1).astype(np.int32)
        btn_action = np.argmax(btn_logits, axis=-1).astype(np.int32)

    # Joint log-probability
    dpad_lp = _log_softmax(dpad_logits, axis=-1)
    btn_lp = _log_softmax(btn_logits, axis=-1)
    log_prob = (
        dpad_lp[np.arange(batch), dpad_action]
        + btn_lp[np.arange(batch), btn_action]
    )

    return dpad_action, btn_action, log_prob


# ---------------------------------------------------------------------------
# Log-probability computation (Keras ops for use in the PPO loss)
# ---------------------------------------------------------------------------


def compute_log_probs(
    dpad_logits,   # (B, 5)  Keras tensor
    btn_logits,    # (B, N)  Keras tensor
    dpad_actions,  # (B,)    Keras tensor
    btn_actions,   # (B,)    Keras tensor
):
    """Compute joint log-probability of stored actions under current policy.

    Suitable for use inside the PPO surrogate loss (differentiable through the
    model weights).
    """
    dpad_lp = ops.log_softmax(dpad_logits, axis=-1)
    btn_lp = ops.log_softmax(btn_logits, axis=-1)

    dpad_lp = ops.take_along_axis(
        dpad_lp, dpad_actions[:, None], axis=-1
    )[:, 0]
    btn_lp = ops.take_along_axis(
        btn_lp, btn_actions[:, None], axis=-1
    )[:, 0]

    return dpad_lp + btn_lp


# ---------------------------------------------------------------------------
# Entropy bonus (Keras ops)
# ---------------------------------------------------------------------------


def policy_entropy(
    dpad_logits,  # (B, 5)  Keras tensor
    btn_logits,   # (B, N)  Keras tensor
):
    """Compute total policy entropy ``H[π(dpad)] + H[π(btn)]``.

    Used as an entropy bonus in the PPO loss for exploration.
    """
    def _cat_entropy(logits):
        p = ops.softmax(logits, axis=-1)
        logp = ops.log_softmax(logits, axis=-1)
        return -ops.sum(p * logp, axis=-1)

    return _cat_entropy(dpad_logits) + _cat_entropy(btn_logits)
