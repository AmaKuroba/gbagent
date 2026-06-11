"""Double DQN with LSTM for retro-driver.

Network: CNN (4-stack frames) -> LSTM -> Dual Q-heads (dpad + buttons)
Training: Double DQN with experience replay, epsilon decay, target network.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

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
logger.info("DQN device: %s", DEVICE)


# ---- frame store (individual frames, not stacks) ------------------


class FrameStore:
    """Ring buffer of individual grayscale frames (144x160 uint8).

    Frames are stored once and referenced by global ID from transitions,
    eliminating the ~8x memory redundancy of storing full stacks per transition.
    """

    def __init__(self, max_frames: int = 200_000) -> None:
        self.max_frames = max_frames
        self._frames: dict[int, np.ndarray] = {}
        self._next_gid = 0
        self._oldest_gid = 0

    def add(self, frame: np.ndarray) -> int:
        gid = self._next_gid
        self._frames[gid] = frame
        self._next_gid += 1
        if len(self._frames) > self.max_frames:
            del self._frames[self._oldest_gid]
            self._oldest_gid += 1
        return gid

    def get(self, gid: int) -> np.ndarray:
        return self._frames[gid]

    def has(self, gid: int) -> bool:
        return gid in self._frames

    def get_stack(self, start_gid: int, n: int) -> np.ndarray:
        frames = [self._frames[start_gid + i] for i in range(n)]
        return np.stack(frames, axis=0)

    @property
    def next_gid(self) -> int:
        """Next global ID that will be assigned."""
        return self._next_gid

    def is_valid_stack(self, start_gid: int, n: int) -> bool:
        """Check if n consecutive frames starting at start_gid are available."""
        return all(start_gid + i in self._frames for i in range(n))


# ---- network ------------------------------------------------------


class GBFeatureExtractor(nn.Module):
    """CNN feature extractor for Game Boy frames.

    Input: (B, C, H, W) with C=4, H=144, W=160
    Output: feature vector (B, 512)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self._flatten_size = self._compute_flatten()

    def _compute_flatten(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 144, 160)
            return self.conv(dummy).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return self.conv(x)


class GBDQN(nn.Module):
    """Double DQN with LSTM temporal memory.

    Architecture:
        CNN feature extractor -> LSTM -> Dueling Q-heads
          -> Dpad head (5 actions)
          -> Button head (4 actions)

    The LSTM carries hidden state across episodes. Call reset_hidden()
    at episode boundaries.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        lstm_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.feature_extractor = GBFeatureExtractor()
        self.lstm = nn.LSTM(
            input_size=self.feature_extractor._flatten_size,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        # Dueling: separate value + advantage streams for each head
        self.value_fc = nn.Linear(lstm_hidden, 128)

        self.dpad_value_out = nn.Linear(128, 1)
        self.dpad_adv_fc = nn.Linear(lstm_hidden, 128)
        self.dpad_adv_out = nn.Linear(128, 5)

        self.btn_value_out = nn.Linear(128, 1)
        self.btn_adv_fc = nn.Linear(lstm_hidden, 128)
        self.btn_adv_out = nn.Linear(128, 4)

        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._lstm_hidden = lstm_hidden

    def reset_hidden(self, batch_size: int = 1) -> None:
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self._lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self._lstm_hidden, device=device)
        self._hidden = (h, c)

    def forward(
        self, frames: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            frames: (B, seq_len, 4, 144, 160) uint8 tensor for sequence training,
                    or (B, 4, 144, 160) for single-step inference.
            hidden: Optional (h, c) tuple. If None, uses stored _hidden.

        Returns:
            dpad_q: (B, 5) Q-values for dpad actions
            btn_q: (B, 4) Q-values for button actions
            hidden: (h, c) LSTM hidden state after processing
        """
        # Handle both single-step and sequence inputs
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)  # (B, 1, 4, 144, 160)

        batch_size, seq_len = frames.shape[0], frames.shape[1]

        # Reshape to (B * seq_len, 4, 144, 160) for CNN, then back
        flat = frames.view(batch_size * seq_len, 4, 144, 160)
        features = self.feature_extractor(flat)  # (B*seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)  # (B, seq_len, feature_dim)

        # LSTM
        if hidden is None:
            if self._hidden is None or self._hidden[0].shape[1] != batch_size:
                self.reset_hidden(batch_size)
            hidden = self._hidden

        lstm_out, new_hidden = self.lstm(features, hidden)
        lstm_out = lstm_out[:, -1, :] if seq_len > 1 else lstm_out.squeeze(1)

        self._hidden = new_hidden

        # Dueling DQN — separate V(s) per head
        hidden_relu = F.relu(self.value_fc(lstm_out))
        dpad_val = self.dpad_value_out(hidden_relu)
        dpad_adv = self.dpad_adv_out(F.relu(self.dpad_adv_fc(lstm_out)))
        btn_val = self.btn_value_out(hidden_relu)
        btn_adv = self.btn_adv_out(F.relu(self.btn_adv_fc(lstm_out)))

        dpad_q = dpad_val + (dpad_adv - dpad_adv.mean(dim=1, keepdim=True))
        btn_q = btn_val + (btn_adv - btn_adv.mean(dim=1, keepdim=True))

        return dpad_q, btn_q, new_hidden

    @torch.no_grad()
    def act(self, frames: np.ndarray, epsilon: float = 0.0) -> tuple[int, int]:
        if random.random() < epsilon:
            return random.randint(0, 4), random.randint(0, 3)

        tensor = torch.from_numpy(frames).unsqueeze(0).to(DEVICE)
        dpad_q, btn_q, _ = self.forward(tensor)
        dpad_a = int(dpad_q.argmax(dim=1).item())
        btn_a = int(btn_q.argmax(dim=1).item())
        return dpad_a, btn_a


# ---- replay buffer ------------------------------------------------


@dataclass
class Transition:
    """Lightweight transition referencing frames by global ID in FrameStore."""

    frame_gid: int  # global ID of the first frame in the current stack
    dpad_action: int
    btn_action: int
    reward: float
    terminated: bool
    truncated: bool


class ReplayBuffer:
    """Circular experience replay buffer with frame-efficient storage
    and sequence sampling for LSTM training."""

    def __init__(
        self,
        capacity: int = 100_000,
        frame_stack: int = 4,
        frame_store: FrameStore | None = None,
    ) -> None:
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.frame_store = frame_store or FrameStore(max_frames=capacity + 1000)
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self._valid_indices: set[int] = set()  # indices with valid frame stacks

    def push(self, t: Transition) -> None:
        idx = len(self.buffer)
        self.buffer.append(t)
        # Old entry at idx was evicted if buffer was full; remove from valid set
        if idx in self._valid_indices:
            self._valid_indices.discard(idx)
        # Check if new entry is valid
        if self.frame_store.is_valid_stack(t.frame_gid, self.frame_stack):
            self._valid_indices.add(idx)

    def _rebuild_valid_indices(self) -> None:
        """Rebuild the valid indices set from scratch."""
        self._valid_indices = {
            i for i, t in enumerate(self.buffer)
            if self.frame_store.is_valid_stack(t.frame_gid, self.frame_stack)
        }

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample random individual transitions with valid frames."""
        if not self._valid_indices:
            return []
        if len(self._valid_indices) <= batch_size:
            return [self.buffer[i] for i in random.sample(list(self._valid_indices), len(self._valid_indices))]
        return [self.buffer[i] for i in random.sample(list(self._valid_indices), batch_size)]

    def sample_sequences(self, batch_size: int, seq_len: int) -> list[list[Transition]]:
        """Sample sequences of consecutive transitions for LSTM training.

        Each sequence is seq_len consecutive transitions from the same
        episode with valid frame data.
        """
        buf = list(self.buffer)
        n = len(buf)
        if n < seq_len:
            return []

        # Precompute episode boundaries and valid frame positions
        is_boundary = [t.terminated or t.truncated for t in buf]
        has_valid_start = [
            self.frame_store.is_valid_stack(t.frame_gid, self.frame_stack) for t in buf
        ]

        # Find valid sequence start positions in one pass
        valid_starts: list[int] = []
        for i in range(n - seq_len + 1):
            if not has_valid_start[i]:
                continue
            # No episode boundary within the sequence (except possibly at the end)
            if any(is_boundary[i + j] for j in range(seq_len - 1)):
                continue
            # Frames must be consecutive
            base_gid = buf[i].frame_gid
            consecutive = all(
                buf[i + j].frame_gid == base_gid + j for j in range(1, seq_len)
            )
            if consecutive:
                valid_starts.append(i)

        if not valid_starts:
            return []

        # Sample from valid starts
        picks = random.sample(valid_starts, min(batch_size, len(valid_starts)))
        return [list(buf[s : s + seq_len]) for s in picks]

    def __len__(self) -> int:
        return len(self.buffer)


# ---- DQN agent ----------------------------------------------------


@dataclass
class DQNConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50_000
    buffer_capacity: int = 10_000
    batch_size: int = 32
    target_update_interval: int = 1_000
    train_start: int = 2000
    seq_len: int = 32  # LSTM sequence length for training (set to 1 to disable temporal learning)


class DQNAgent:
    """Double DQN agent with LSTM memory.

    Manages online and target networks, replay buffer, epsilon schedule,
    and training step logic.
    """

    def __init__(
        self,
        config: DQNConfig | None = None,
        frame_store: FrameStore | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.online = GBDQN().to(DEVICE)
        self.target = GBDQN().to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=self.config.learning_rate,
        )

        if frame_store is None:
            frame_store = FrameStore(
                max_frames=self.config.buffer_capacity + self.config.seq_len + 100
            )
        self.buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            frame_store=frame_store,
        )

        self.step_count = 0
        self.episode_count = 0

    @property
    def epsilon(self) -> float:
        ratio = min(1.0, self.step_count / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + ratio * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def act(self, frames: np.ndarray, *, training: bool = True) -> tuple[int, int]:
        eps = self.epsilon if training else 0.0
        return self.online.act(frames, eps)

    def learn(self) -> dict[str, float]:
        if len(self.buffer) < self.config.train_start:
            return {"loss": 0.0, "q_dpad": 0.0, "q_btn": 0.0, "epsilon": self.epsilon}

        seq_len = self.config.seq_len

        if seq_len > 1:
            return self._learn_sequences(seq_len)
        return self._learn_single()

    def _learn_single(self) -> dict[str, float]:
        """Single-step training (no LSTM temporal learning)."""
        batch = self.buffer.sample(self.config.batch_size)
        if not batch:
            return {"loss": 0.0, "q_dpad": 0.0, "q_btn": 0.0}

        fs = self.buffer.frame_store
        fstack = self.buffer.frame_stack

        frames = torch.from_numpy(np.stack([fs.get_stack(t.frame_gid, fstack) for t in batch])).to(
            DEVICE
        )
        next_frames = torch.from_numpy(
            np.stack([fs.get_stack(t.frame_gid + 1, fstack) for t in batch])
        ).to(DEVICE)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(
            [t.terminated or t.truncated for t in batch],
            dtype=torch.float32,
            device=DEVICE,
        )
        dpad_actions = torch.tensor([t.dpad_action for t in batch], device=DEVICE)
        btn_actions = torch.tensor([t.btn_action for t in batch], device=DEVICE)

        self.online.reset_hidden(len(batch))
        dpad_q, btn_q, _ = self.online(frames)
        dpad_q_selected = dpad_q.gather(1, dpad_actions.unsqueeze(1)).squeeze(1)
        btn_q_selected = btn_q.gather(1, btn_actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            self.online.reset_hidden(len(batch))
            next_dpad_q_online, next_btn_q_online, _ = self.online(next_frames)

            self.target.reset_hidden(len(batch))
            next_dpad_q_target, next_btn_q_target, _ = self.target(next_frames)

            dpad_next = next_dpad_q_target.gather(
                1, next_dpad_q_online.argmax(dim=1, keepdim=True)
            ).squeeze(1)
            btn_next = next_btn_q_target.gather(
                1, next_btn_q_online.argmax(dim=1, keepdim=True)
            ).squeeze(1)

            target = rewards + self.config.gamma * (1 - dones) * (dpad_next + btn_next) / 2

        loss = F.mse_loss(dpad_q_selected + btn_q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        # Hard target update
        if self.step_count % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.step_count += 1

        return {
            "loss": loss.item(),
            "q_dpad": dpad_q.mean().item(),
            "q_btn": btn_q.mean().item(),
            "epsilon": self.epsilon,
        }

    def _process_sequence(
        self,
        network: GBDQN,
        frames_seq: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a (B, seq_len, 4, 144, 160) sequence through CNN + LSTM
        and return per-timestep Q-values.

        Returns:
            dpad_q: (B, seq_len, 5)
            btn_q: (B, seq_len, 4)
        """
        B, S = frames_seq.shape[0], frames_seq.shape[1]

        flat = frames_seq.view(B * S, 4, 144, 160)
        features = network.feature_extractor(flat)
        features = features.view(B, S, -1)

        if hidden is None:
            h = torch.zeros(1, B, network._lstm_hidden, device=DEVICE)
            c = torch.zeros(1, B, network._lstm_hidden, device=DEVICE)
            hidden = (h, c)

        lstm_out, _ = network.lstm(features, hidden)  # (B, S, hidden)

        flat_out = lstm_out.reshape(B * S, -1)
        hidden_relu = F.relu(network.value_fc(flat_out))
        dpad_val = network.dpad_value_out(hidden_relu)
        dpad_adv = network.dpad_adv_out(F.relu(network.dpad_adv_fc(flat_out)))
        btn_val = network.btn_value_out(hidden_relu)
        btn_adv = network.btn_adv_out(F.relu(network.btn_adv_fc(flat_out)))

        dpad_q = dpad_val + (dpad_adv - dpad_adv.mean(dim=1, keepdim=True))
        btn_q = btn_val + (btn_adv - btn_adv.mean(dim=1, keepdim=True))

        return dpad_q.view(B, S, 5), btn_q.view(B, S, 4)

    def _learn_sequences(self, seq_len: int) -> dict[str, float]:
        sequences = self.buffer.sample_sequences(self.config.batch_size, seq_len)
        if len(sequences) == 0:
            return {"loss": 0.0, "q_dpad": 0.0, "q_btn": 0.0}

        fs = self.buffer.frame_store
        fstack = self.buffer.frame_stack
        # Ensure we have enough frames for at least one stack before training
        if fs.next_gid <= fstack * 2:
            return {"loss": 0.0, "q_dpad": 0.0, "q_btn": 0.0, "epsilon": self.epsilon}
        max_gid = fs.next_gid - fstack

        frames_list = []
        next_frames_list = []
        rewards_list = []
        dones_list = []
        dpad_actions_list = []
        btn_actions_list = []

        for seq in sequences:
            f = np.stack([
                fs.get_stack(min(t.frame_gid, max_gid), fstack) for t in seq
            ], axis=0)
            nf = np.stack([
                fs.get_stack(min(t.frame_gid + 1, max_gid), fstack) for t in seq
            ], axis=0)
            frames_list.append(f)
            next_frames_list.append(nf)
            rewards_list.append([t.reward for t in seq])
            dones_list.append([float(t.terminated or t.truncated) for t in seq])
            dpad_actions_list.append([t.dpad_action for t in seq])
            btn_actions_list.append([t.btn_action for t in seq])

        frames = torch.from_numpy(np.stack(frames_list)).to(DEVICE)
        next_frames = torch.from_numpy(np.stack(next_frames_list)).to(DEVICE)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones_list, dtype=torch.float32, device=DEVICE)
        dpad_actions = torch.tensor(dpad_actions_list, device=DEVICE)
        btn_actions = torch.tensor(btn_actions_list, device=DEVICE)

        # Online Q-values for all timesteps
        dpad_q_all, btn_q_all = self._process_sequence(self.online, frames)

        q_dpad_selected = dpad_q_all.gather(2, dpad_actions.unsqueeze(-1)).squeeze(-1)
        q_btn_selected = btn_q_all.gather(2, btn_actions.unsqueeze(-1)).squeeze(-1)

        # Double DQN target
        with torch.no_grad():
            dpad_online, btn_online = self._process_sequence(self.online, next_frames)
            dpad_target, btn_target = self._process_sequence(self.target, next_frames)

            dpad_next = dpad_target.gather(2, dpad_online.argmax(dim=2, keepdim=True)).squeeze(-1)
            btn_next = btn_target.gather(2, btn_online.argmax(dim=2, keepdim=True)).squeeze(-1)

            target = rewards + self.config.gamma * (1 - dones) * (dpad_next + btn_next) / 2

        loss = F.mse_loss(q_dpad_selected + q_btn_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        if self.step_count % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        self.step_count += 1

        return {
            "loss": loss.item(),
            "q_dpad": dpad_q_all[:, -1, :].mean().item(),
            "q_btn": btn_q_all[:, -1, :].mean().item(),
            "epsilon": self.epsilon,
        }

    def reset_episode(self) -> None:
        self.online.reset_hidden(1)
        self.target.reset_hidden(1)
        self.episode_count += 1

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state": self.online.state_dict(),
                "target_state": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "episode_count": self.episode_count,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=DEVICE, weights_only=True)
        self.online.load_state_dict(data["online_state"])
        self.target.load_state_dict(data["target_state"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.step_count = data["step_count"]
        self.episode_count = data["episode_count"]
