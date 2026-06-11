"""PPO with ViT encoder for retro-driver.

Architecture: ViT → LSTM → Actor-Critic (dual heads: dpad + buttons)
Training: PPO with GAE, clipped surrogate loss, entropy bonus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---- device -------------------------------------------------------


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
logger.info("PPO device: %s", DEVICE)


# ---- ViT encoder --------------------------------------------------


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""

    def __init__(self, img_size: int = 144, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 384):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for Game Boy frames.

    Input: (B, C, H, W) with C=1 (grayscale), H=144, W=160
    Output: (B, embed_dim) feature vector
    """

    def __init__(
        self,
        img_size: int = 144,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embedding.num_patches

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Resize to 144x144 if needed
        if x.shape[-2] != 144 or x.shape[-1] != 144:
            x = F.adaptive_avg_pool2d(x, (144, 144))

        # Patch embedding
        x = self.patch_embedding(x)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm and CLS token output
        x = self.norm(x)
        return x[:, 0]  # CLS token


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm MLP
        return x + self.mlp(self.norm2(x))


# ---- Actor-Critic network ----------------------------------------


class ActorCritic(nn.Module):
    """PPO actor-critic with ViT encoder and LSTM.

    Architecture:
        ViT → LSTM → Policy heads (dpad + buttons) + Value head
    """

    def __init__(
        self,
        embed_dim: int = 384,
        lstm_hidden: int = 256,
        dpad_actions: int = 5,
        btn_actions: int = 4,
    ):
        super().__init__()
        self.encoder = ViTEncoder(embed_dim=embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True)

        # Policy heads (separate for dpad and buttons)
        self.dpad_policy = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, dpad_actions),
        )
        self.btn_policy = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, btn_actions),
        )

        # Value head
        self.value = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._lstm_hidden = lstm_hidden
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_hidden(self, batch_size: int = 1) -> None:
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self._lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self._lstm_hidden, device=device)
        self._hidden = (h, c)

    def get_hidden(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._hidden

    def set_hidden(self, hidden: tuple[torch.Tensor, torch.Tensor]) -> None:
        self._hidden = hidden

    def forward(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            obs: (B, C, H, W) or (B, T, C, H, W) for sequences
            hidden: Optional LSTM hidden state

        Returns:
            dpad_logits: (B, dpad_actions) or (B, T, dpad_actions)
            btn_logits: (B, btn_actions) or (B, T, btn_actions)
            values: (B,) or (B, T)
            new_hidden: Updated LSTM hidden state
        """
        if obs.dim() == 5:
            # Sequence input: (B, T, C, H, W)
            B, T, C, H, W = obs.shape
            obs = obs.reshape(B * T, C, H, W)
            features = self.encoder(obs)
            features = features.reshape(B, T, -1)

            # LSTM
            if hidden is None:
                self.reset_hidden(B)
                hidden = self._hidden
            features, new_hidden = self.lstm(features, hidden)
            self._hidden = new_hidden

            # Policy and value
            dpad_logits = self.dpad_policy(features)
            btn_logits = self.btn_policy(features)
            values = self.value(features).squeeze(-1)
        else:
            # Single input: (B, C, H, W)
            features = self.encoder(obs)

            # LSTM
            if hidden is None:
                self.reset_hidden(obs.shape[0])
                hidden = self._hidden
            features, new_hidden = self.lstm(features.unsqueeze(1), hidden)
            features = features.squeeze(1)
            self._hidden = new_hidden

            # Policy and value
            dpad_logits = self.dpad_policy(features)
            btn_logits = self.btn_policy(features)
            values = self.value(features).squeeze(-1)

        return dpad_logits, btn_logits, values, new_hidden

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        dpad_action: int | None = None,
        btn_action: int | None = None,
    ) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Get action, log_prob, and value for PPO.

        Args:
            obs: Current observation
            hidden: LSTM hidden state
            dpad_action: If provided, use this action (for rollout collection)
            btn_action: If provided, use this action (for rollout collection)

        Returns:
            dpad_action: Selected dpad action
            btn_action: Selected button action
            dpad_log_prob: Log probability of dpad action
            btn_log_prob: Log probability of button action
            value: State value estimate
            new_hidden: Updated LSTM hidden state
        """
        dpad_logits, btn_logits, value, new_hidden = self.forward(obs, hidden)

        # Sample from policy distributions
        dpad_dist = torch.distributions.Categorical(logits=dpad_logits)
        btn_dist = torch.distributions.Categorical(logits=btn_logits)

        if dpad_action is None:
            dpad_action = dpad_dist.sample()
        if btn_action is None:
            btn_action = btn_dist.sample()

        dpad_log_prob = dpad_dist.log_prob(dpad_action)
        btn_log_prob = btn_dist.log_prob(btn_action)

        return (
            dpad_action.item(),
            btn_action.item(),
            dpad_log_prob,
            btn_log_prob,
            value,
            new_hidden,
        )

    def get_value(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Get value estimate only (for GAE computation)."""
        _, _, value, _ = self.forward(obs, hidden)
        return value


# ---- Rollout buffer ------------------------------------------------


@dataclass
class RolloutStep:
    """Single step in a rollout buffer."""

    obs: np.ndarray  # (C, H, W) uint8
    dpad_action: int
    btn_action: int
    dpad_log_prob: float
    btn_log_prob: float
    reward: float
    done: bool
    value: float


class RolloutBuffer:
    """Rollout buffer for PPO training."""

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.buffer: list[RolloutStep] = []

    def add(self, step: RolloutStep) -> None:
        self.buffer.append(step)

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns.

        Args:
            last_value: Value estimate for the state after the last step
            gamma: Discount factor
            lam: GAE lambda

        Returns:
            advantages: (T,) advantage estimates
            returns: (T,) discounted returns
        """
        T = len(self.buffer)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        # Compute GAE backwards
        gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else self.buffer[t + 1].value

            reward = self.buffer[t].reward
            done = self.buffer[t].done
            value = self.buffer[t].value

            delta = reward + gamma * next_value * (1.0 - done) - value
            gae = delta + gamma * lam * (1.0 - done) * gae
            advantages[t] = gae
            returns[t] = gae + value

        return advantages, returns

    def get_batches(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
        batch_size: int = 32,
    ) -> list[dict[str, torch.Tensor]]:
        """Get shuffled minibatches for PPO update.

        Returns:
            List of dicts with keys: obs, dpad_actions, btn_actions,
            old_dpad_log_probs, old_btn_log_probs, returns, advantages
        """
        T = len(self.buffer)
        indices = np.random.permutation(T)

        batches = []
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_indices = indices[start:end]

            batch = {
                "obs": torch.stack([
                    torch.from_numpy(self.buffer[i].obs).float() / 255.0
                    for i in batch_indices
                ]),
                "dpad_actions": torch.tensor(
                    [self.buffer[i].dpad_action for i in batch_indices],
                    dtype=torch.long,
                ),
                "btn_actions": torch.tensor(
                    [self.buffer[i].btn_action for i in batch_indices],
                    dtype=torch.long,
                ),
                "old_dpad_log_probs": torch.tensor(
                    [self.buffer[i].dpad_log_prob for i in batch_indices],
                    dtype=torch.float32,
                ),
                "old_btn_log_probs": torch.tensor(
                    [self.buffer[i].btn_log_prob for i in batch_indices],
                    dtype=torch.float32,
                ),
                "returns": torch.tensor(
                    [returns[i] for i in batch_indices],
                    dtype=torch.float32,
                ),
                "advantages": torch.tensor(
                    [advantages[i] for i in batch_indices],
                    dtype=torch.float32,
                ),
            }
            batches.append(batch)

        return batches


# ---- PPO config ---------------------------------------------------


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    rollout_len: int = 128
    batch_size: int = 32
    total_steps: int = 6_000_000

    # ViT
    embed_dim: int = 384
    lstm_hidden: int = 256


# ---- PPO agent ----------------------------------------------------


class PPOAgent:
    """PPO agent with ViT encoder for Pokemon."""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.network = ActorCritic(
            embed_dim=config.embed_dim,
            lstm_hidden=config.lstm_hidden,
        ).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.buffer = RolloutBuffer(capacity=config.rollout_len)

        self.step_count = 0
        self.update_count = 0

    def act(
        self,
        obs: np.ndarray,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        training: bool = True,
    ) -> tuple[int, int, float, float, float, tuple[torch.Tensor, torch.Tensor]]:
        """Select action and collect rollout data.

        Args:
            obs: (C, H, W) uint8 observation
            hidden: LSTM hidden state
            training: Whether to sample (True) or use greedy (False)

        Returns:
            dpad_action: Selected dpad action
            btn_action: Selected button action
            dpad_log_prob: Log probability of dpad action
            btn_log_prob: Log probability of button action
            value: State value estimate
            new_hidden: Updated LSTM hidden state
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            if training:
                dpad_action, btn_action, dpad_log_prob, btn_log_prob, value, new_hidden = \
                    self.network.get_action_and_value(obs_tensor, hidden)
            else:
                # Greedy action selection
                dpad_logits, btn_logits, value, new_hidden = self.network.forward(obs_tensor, hidden)
                dpad_action = dpad_logits.argmax(dim=-1).item()
                btn_action = btn_logits.argmax(dim=-1).item()
                dpad_log_prob = 0.0
                btn_log_prob = 0.0

        return dpad_action, btn_action, dpad_log_prob, btn_log_prob, value.item(), new_hidden

    def add_to_buffer(
        self,
        obs: np.ndarray,
        dpad_action: int,
        btn_action: int,
        dpad_log_prob: float,
        btn_log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Add a step to the rollout buffer."""
        step = RolloutStep(
            obs=obs,
            dpad_action=dpad_action,
            btn_action=btn_action,
            dpad_log_prob=dpad_log_prob,
            btn_log_prob=btn_log_prob,
            reward=reward,
            done=done,
            value=value,
        )
        self.buffer.add(step)

    def update(self) -> dict[str, float]:
        """Run PPO update on collected rollout.

        Returns:
            Dictionary of training statistics.
        """
        if len(self.buffer) == 0:
            return {}

        # Get last value for GAE
        last_obs = self.buffer.buffer[-1].obs
        last_obs_tensor = torch.from_numpy(last_obs).float().unsqueeze(0).to(DEVICE) / 255.0
        with torch.no_grad():
            last_value = self.network.get_value(last_obs_tensor).item()

        # Compute GAE
        advantages, returns = self.buffer.compute_gae(
            last_value,
            gamma=self.config.gamma,
            lam=self.config.lam,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        for _ in range(self.config.ppo_epochs):
            batches = self.buffer.get_batches(
                advantages, returns, self.config.batch_size
            )

            for batch in batches:
                obs = batch["obs"].to(DEVICE)
                dpad_actions = batch["dpad_actions"].to(DEVICE)
                btn_actions = batch["btn_actions"].to(DEVICE)
                old_dpad_log_probs = batch["old_dpad_log_probs"].to(DEVICE)
                old_btn_log_probs = batch["old_btn_log_probs"].to(DEVICE)
                returns_batch = batch["returns"].to(DEVICE)
                advantages_batch = batch["advantages"].to(DEVICE)

                # Forward pass
                dpad_logits, btn_logits, values, _ = self.network.forward(obs)

                # Compute new log probs
                dpad_dist = torch.distributions.Categorical(logits=dpad_logits)
                btn_dist = torch.distributions.Categorical(logits=btn_logits)

                new_dpad_log_probs = dpad_dist.log_prob(dpad_actions)
                new_btn_log_probs = btn_dist.log_prob(btn_actions)

                # Compute ratios
                dpad_ratio = torch.exp(new_dpad_log_probs - old_dpad_log_probs)
                btn_ratio = torch.exp(new_btn_log_probs - old_btn_log_probs)

                # PPO clipped loss
                dpad_surr1 = dpad_ratio * advantages_batch
                dpad_surr2 = torch.clamp(
                    dpad_ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                ) * advantages_batch
                dpad_policy_loss = -torch.min(dpad_surr1, dpad_surr2).mean()

                btn_surr1 = btn_ratio * advantages_batch
                btn_surr2 = torch.clamp(
                    btn_ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                ) * advantages_batch
                btn_policy_loss = -torch.min(btn_surr1, btn_surr2).mean()

                policy_loss = dpad_policy_loss + btn_policy_loss

                # Value loss
                value_loss = F.mse_loss(values, returns_batch)

                # Entropy bonus
                entropy_loss = -(dpad_dist.entropy().mean() + btn_dist.entropy().mean())

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1

        # Clear buffer
        self.buffer.clear()

        # Increment counters
        self.step_count += self.config.rollout_len
        self.update_count += 1

        # Return stats
        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy_loss": total_entropy_loss / max(num_updates, 1),
            "step_count": self.step_count,
            "update_count": self.update_count,
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "update_count": self.update_count,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint.get("step_count", 0)
        self.update_count = checkpoint.get("update_count", 0)
