"""Masked Autoencoder (MAE) pre-training for ViT encoder.

Trains the ViT encoder on unlabeled Pokemon gameplay frames.
The encoder learns visual features (sprites, menus, text, backgrounds)
without any labels or rewards.

Usage:
    uv run python -m retro_driver.pretrain --data pokemon_dataset.npz
    uv run python -m retro_driver.pretrain --data pokemon_dataset.npz --epochs 200
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from retro_driver.ppo import DEVICE, TransformerBlock, ViTEncoder


# ---- MAE model ----------------------------------------------------


class MAE(nn.Module):
    """Masked Autoencoder for self-supervised pre-training.

    Architecture:
        Encoder: ViT (processes visible patches)
        Decoder: Lightweight MLP (reconstructs masked patches)
    """

    def __init__(
        self,
        img_size: int = 144,
        patch_size: int = 16,
        in_channels: int = 1,
        encoder_embed_dim: int = 384,
        encoder_depth: int = 6,
        encoder_heads: int = 6,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        mask_ratio: float = 0.5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

        # Encoder (ViT)
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )

        # Decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim) * 0.02)
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, decoder_embed_dim) * 0.02
        )
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_heads, 4.0, 0.1)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_channels)

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random masking of patches.

        Args:
            x: (B, num_patches, embed_dim) patch embeddings
            mask_ratio: Fraction of patches to mask

        Returns:
            x_masked: (B, num_visible, embed_dim) visible patches
            mask: (B, num_patches) binary mask (1 = masked)
            ids_restore: (B, num_patches) indices to restore original order
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        # Random permutation
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first num_keep patches
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Binary mask: 0 = keep, 1 = masked
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode with masking.

        Args:
            x: (B, C, H, W) input images

        Returns:
            latent: (B, num_visible, embed_dim) visible patch embeddings
            mask: (B, num_patches) binary mask
            ids_restore: (B, num_patches) restoration indices
        """
        # Patch embedding (without CLS token for MAE)
        x = self.encoder.patch_embedding(x)  # (B, num_patches, embed_dim)

        # Add positional embedding (without CLS)
        x = x + self.encoder.pos_embed[:, 1:, :]

        # Random masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # Prepend CLS token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer blocks
        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.norm(x)

        # Remove CLS token
        latent = x[:, 1:, :]

        return latent, mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Decode and reconstruct patches.

        Args:
            latent: (B, num_visible, embed_dim) encoder output
            ids_restore: (B, num_patches) restoration indices

        Returns:
            recon: (B, num_patches, patch_size^2 * channels) reconstructed patches
        """
        # Project to decoder dimension
        x = self.decoder_embed(latent)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x = torch.cat([x, mask_tokens], dim=1)

        # Unshuffle to original order
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )

        # Add positional embedding
        x = x + self.decoder_pos_embed

        # Transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # Predict patches
        x = self.decoder_pred(x)

        return x

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patch sequences.

        Args:
            imgs: (B, C, H, W) images

        Returns:
            patches: (B, num_patches, patch_size^2 * C)
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        h = w = H // p

        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
        x = x.reshape(B, h * w, p * p * C)

        return x

    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with masking and reconstruction.

        Args:
            imgs: (B, C, H, W) input images

        Returns:
            loss: Reconstruction loss (MSE)
            pred: Reconstructed patches
        """
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)

        # Target: original patches
        target = self.patchify(imgs)

        # Reconstruction loss (MSE on masked patches only)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, num_patches)
        loss = (loss * mask).sum() / mask.sum()  # Mean over masked patches

        return loss, pred


# ---- Dataset ------------------------------------------------------


class FrameDataset(Dataset):
    """Dataset of collected gameplay frames."""

    def __init__(self, data_path: str):
        data = np.load(data_path)
        self.frames = data["frames"]  # (N, H, W) uint8
        print(f"Loaded {len(self.frames)} frames from {data_path}")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.frames[idx]
        # Add channel dimension and normalize to [0, 1]
        return torch.from_numpy(frame).float().unsqueeze(0) / 255.0


# ---- Training loop ------------------------------------------------


def train_mae(
    data_path: str,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1.5e-4,
    mask_ratio: float = 0.5,
    save_path: str = "pretrained_vit.pt",
) -> None:
    """Train MAE on collected frames.

    Args:
        data_path: Path to .npz dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        mask_ratio: Fraction of patches to mask
        save_path: Path to save pre-trained encoder
    """
    # Load dataset
    dataset = FrameDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Create MAE model
    model = MAE(mask_ratio=mask_ratio).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

    # Learning rate schedule (cosine decay)
    def lr_lambda(step: int) -> float:
        min_lr = learning_rate * 0.1
        return max(min_lr / learning_rate, 1.0 - step / (epochs * len(dataloader)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Training MAE for {epochs} epochs")
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Device: {DEVICE}")
    print(f"  Mask ratio: {mask_ratio}")
    print()

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch in dataloader:
            batch = batch.to(DEVICE)

            # Forward pass
            loss, _ = model(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        # Epoch stats
        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save only the encoder weights
            torch.save(model.encoder.state_dict(), save_path)
            print(f"  Saved encoder to {save_path} (best loss: {best_loss:.6f})")

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Encoder saved to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MAE pre-training for ViT encoder")
    parser.add_argument("--data", required=True, help="Path to .npz dataset")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--mask-ratio", type=float, default=0.5, help="Mask ratio")
    parser.add_argument("--save-path", default="pretrained_vit.pt", help="Output path")
    args = parser.parse_args()

    train_mae(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mask_ratio=args.mask_ratio,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
