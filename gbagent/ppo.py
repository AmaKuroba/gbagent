"""PPO loss functions and update step for gbagent.

Provides the clipped surrogate objective, entropy bonus, value loss, gradient
clipping, and AdamW optimiser wiring.  Designed to work with the existing
ActorCritic model, RolloutBuffer, and action utilities.
"""

from __future__ import annotations

import keras
import tensorflow as tf
from keras import ops

from gbagent.action import compute_log_probs, policy_entropy

# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------


def ppo_loss(
    dpad_logits,  # (B, 5)
    btn_logits,  # (B, 6)
    values,  # (B, 1)
    dpad_actions,  # (B,)
    btn_actions,  # (B,)
    returns,  # (B,)
    advantages,  # (B,)
    old_log_probs,  # (B,)
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> dict[str, keras.KerasTensor | float]:
    """Compute PPO loss components.

    Parameters
    ----------
    dpad_logits, btn_logits, values :
        Current-policy outputs from the ActorCritic model.
    dpad_actions, btn_actions :
        Actions stored during rollout.
    returns, advantages :
        GAE returns and (normalised) advantages from the buffer.
    old_log_probs :
        Log-probabilities of the stored actions under the old policy.
    clip_epsilon, value_coef, entropy_coef :
        PPO hyper-parameters.

    Returns
    -------
    dict with keys ``policy_loss``, ``value_loss``, ``entropy_loss``,
    ``total_loss``, ``approx_kl``, ``clip_frac``.
    """
    # ── Probability ratio ────────────────────────────────────────────
    new_log_probs = compute_log_probs(dpad_logits, btn_logits, dpad_actions, btn_actions)  # (B,)
    ratio = ops.exp(new_log_probs - old_log_probs)  # (B,)

    # ── Clipped surrogate objective ──────────────────────────────────
    surr1 = ratio * advantages
    surr2 = ops.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -ops.mean(ops.minimum(surr1, surr2))

    # ── Value loss (Huber-like MSE) ──────────────────────────────────
    values_sq = ops.squeeze(values, axis=-1)  # (B,)  — same shape as returns
    value_loss = ops.mean(ops.square(returns - values_sq))

    # ── Entropy bonus (encourage exploration) ────────────────────────
    entropy = policy_entropy(dpad_logits, btn_logits)  # (B,)
    entropy_loss = -ops.mean(entropy)  # negative because we *maximise* entropy

    # ── Total loss ───────────────────────────────────────────────────
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    # ── Diagnostics ──────────────────────────────────────────────────
    approx_kl = ops.mean(ops.square(new_log_probs - old_log_probs))
    clip_frac = ops.mean(ops.cast(ops.abs(ratio - 1.0) > clip_epsilon, dtype="float32"))

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "total_loss": total_loss,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }


# ---------------------------------------------------------------------------
# PPO update step  (one gradient step on one mini-batch)
# ---------------------------------------------------------------------------


def ppo_update_step(
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
    obs_batch,
    dpad_batch,
    btn_batch,
    return_batch,
    adv_batch,
    old_lp_batch,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> dict[str, float]:
    """Run one PPO gradient update on a single mini-batch.

    Parameters
    ----------
    model :
        The ActorCritic Keras model.
    optimizer :
        AdamW (or Adam) optimiser instance.
    obs_batch … old_lp_batch :
        Mini-batch tensors (already batched — **not** the flat concatenation
        from the buffer; that must be handled outside).
    clip_epsilon … max_grad_norm :
        PPO hyper-parameters.

    Returns
    -------
    dict of scalar loss components for logging.
    """
    with tf.GradientTape() as tape:
        dpad_logits, btn_logits, values = model(obs_batch, training=True)
        losses = ppo_loss(
            dpad_logits=dpad_logits,
            btn_logits=btn_logits,
            values=values,
            dpad_actions=dpad_batch,
            btn_actions=btn_batch,
            returns=return_batch,
            advantages=adv_batch,
            old_log_probs=old_lp_batch,
            clip_epsilon=clip_epsilon,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )
        total_loss = losses["total_loss"]

    # Compute gradients
    grads = tape.gradient(total_loss, model.trainable_variables)

    # Gradient clipping (global norm)
    if max_grad_norm > 0.0:
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

    # Apply gradients
    optimizer.apply_gradients(zip(grads, model.trainable_variables, strict=False))

    return {k: float(ops.convert_to_numpy(v)) for k, v in losses.items()}
