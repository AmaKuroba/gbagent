# retro-driver — Planning Doc

## The Goal

A reinforcement learning agent that learns to play Game Boy games by
looking at raw pixels and pressing buttons. No LLM. No game internals
(other than what we choose to scan from RAM for rewards).

## Milestone 1: Get Out of Pallet Town (Pokemon Red)

Agent spawns in Professor Oak's lab (or right outside it) and learns to
navigate south + east to reach Route 1. Simple on paper, but requires:
- Sustained directional movement (not just mashing random buttons)
- Map awareness — knowing when it's no longer in Pallet Town
- Escape loops (walking in circles around the lab)

## Architecture

### Observation

- 4 most recent grayscale frames, stacked
- 160 x 144 resolution, single-channel (8-bit grayscale)
- CNN input: (4, 160, 144) — uint8, normalized to [0,1] in network

### Actions (Split)

- **D-pad** (5): None, Up, Down, Left, Right
- **Buttons** (4): None, A, B, Start, Select

Combined via MultiDiscrete: 5 x 4 = 20 possible combos.

### Network

```
Input (4, 144, 160)
  ↓ Conv5x5/2 → ReLU
  ↓ Conv3x3/2 → ReLU
  ↓ Conv3x3/2 → ReLU
  ↓ Conv3x3/1 → ReLU → Flatten (512)
  ↓ LSTM (256 hidden, carries across episode)
  ↓ Dueling Q-heads: shared V(s) + separate A(s,a) for dpad + buttons
```

### Algorithm

PPO (Proximal Policy Optimization) with:
- ViT-Small encoder (patch-based, resize to 144×144)
- LSTM temporal memory (256 hidden, carries across episode)
- Dual-head policy (dpad: 5 actions, buttons: 4 actions) + value head
- GAE (λ=0.95) for advantage estimation
- Clip-based surrogate loss (ε=0.2)
- Entropy bonus for exploration

## Reward Design (Hybrid)

### Screen Novelty Baseline (always active)

- Pixel-diff between current frame and previous frame
- +small reward when frame is meaningfully different
- Staleness penalty if low novelty persists

### Per-Game RAM Scanner (configurable)

Configurable via YAML:

```yaml
game: "pokemon-red"
scanners:
  - name: "map_change"
    ram_addr: 0xD35E         # Map ID
    type: "discrete"
    reward_on_change: 1.0
  - name: "party_levels"
    ram_addr: 0xD16B
    type: "delta"
    length: 4
    reward_per_unit: 5.0
```

Reward components summed per step, then clipped to [-10, +10].

### Anti-grind

- Tile-based movement detection (bottom-center hash of frame)
- If 100 consecutive steps on same tile -> penalty floor
- Movement to novel tiles rewarded

## Training Loop

### Save-State Curriculum (planned)

1. **Phase 1:** Spawn outside Oak's lab. Train until agent reaches
   Route 1 reliably (map ID changes to Route 1).
2. **Phase 2:** Spawn at Route 1 entrance. Train until it navigates
   to Viridian City.
3. **Phase 3+:** Add checkpoints for each major progression point.

Checkpoints via gbagent save/load state over MCP.

### Episode Management

- Episode ends on: game crash (white screen), freeze (identical frames),
  or timeout (10k steps)
- On game over: reload save-state
- Stats logged per phase (steps to clear, reward accumulated)

## Tech Stack

- **Python 3.12+** — RL glue, training loop
- **PyTorch 2.3+** — neural networks (MPS on Apple Silicon)
- **Gymnasium 1.0+** — RL environment interface
- **gbagent MCP** — subprocess stdio, JSON-RPC
- **Pillow** — frame decoding
- **TensorBoard** — training curves

## Frame Storage

Transitions reference individual frames via global IDs in a shared
ring buffer (FrameStore), avoiding the ~8x memory overhead of storing
full 4-frame stacks per transition. Replay buffer stores lightweight
( ~40 byte) transition records; frames (~23 KB each) stored once.

## Open Questions

1. Frame rate: how many steps per second can gbagent-mcp handle over stdio?
   Need to benchmark before designing batch sizes.
2. Should we pre-record some demonstration data (human playing) for
   imitation learning to kickstart the agent, or pure RL from scratch?
