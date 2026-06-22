"""ViT-Small encoder + ActorCritic network with Keras 3.

Architecture
------------
    obs ⟶ [Pad ⟶ Conv2D patch embed] ⟶ [+CLS + pos embed] ⟶
    6× TransformerBlock (pre-LN) ⟶ LayerNorm ⟶
    LSTM(hidden=512) ⟶ dpad head (5) / btn head (6) / value head (1)
"""

from __future__ import annotations

import keras
from keras import layers, ops

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAD_TARGET = 96      # smallest multiple of 16 ≥ 84
_N_PATCHES = (PAD_TARGET // 16) ** 2  # 36
_N_TOKENS = _N_PATCHES + 1             # 36 patches + 1 CLS = 37


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="gbagent")
class PatchEmbed(layers.Layer):
    """Convert a frame-stacked observation into a sequence of patch tokens.

    Input:  ``(B, 84, 84, C)`` — 84×84 grayscale, C = frame_stack (default 4).
    Output: ``(B, n_patches, embed_dim)`` where n_patches = 36.
    """

    def __init__(self, patch_size: int = 16, embed_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.padder = layers.ZeroPadding2D(
            padding=6,  # pad 84→96 (6 on each side)
            name="vit_padder",
        )
        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            use_bias=False,
            name="patch_embed",
        )

    def call(self, x, training=False):
        x = self.padder(x)                    # (B, 96, 96, C)
        x = self.proj(x)                      # (B, 6, 6, D)
        batch = ops.shape(x)[0]
        x = ops.reshape(x, (batch, _N_PATCHES, self.embed_dim))  # (B, 36, D)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], _N_PATCHES, self.embed_dim)


# ---------------------------------------------------------------------------
# Transformer block (pre-LayerNorm)
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="gbagent")
class TransformerBlock(layers.Layer):
    """Pre-LN Transformer block: LN → MSA → + → LN → MLP → +.

    Parameters
    ----------
    embed_dim : int
        Token / model dimension.
    num_heads : int
        Number of attention heads (must divide *embed_dim*).
    mlp_ratio : int
        Ratio for MLP hidden dimension (default 4 → hidden=embed_dim*4).
    dropout : float
        Dropout rate applied inside attention and MLP.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        # Pre-LN → MSA
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
            name="mha",
        )
        # Pre-LN → MLP
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name="ln2")
        self.mlp = keras.Sequential(
            [
                layers.Dense(embed_dim * mlp_ratio, activation="gelu", name="fc1"),
                layers.Dropout(dropout, name="mlp_drop"),
                layers.Dense(embed_dim, name="fc2"),
            ],
            name="mlp",
        )

    def call(self, x, training=False):
        # Attention
        residual = x
        x = self.ln1(x)
        x = self.attn(x, x, training=training)
        x = layers.add([residual, x])

        # MLP
        residual = x
        x = self.ln2(x)
        x = self.mlp(x, training=training)
        x = layers.add([residual, x])
        return x


# ---------------------------------------------------------------------------
# ViT-Small encoder
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="gbagent")
class ViTEncoder(layers.Layer):
    """ViT-Small encoder.

    Stack order:
        PatchEmbed → +CLS → +learned pos embed → N×TransformerBlock → LN.

    Input:  ``(B, 84, 84, C)``
    Output: ``(B, n_tokens, embed_dim)``  (n_tokens = 37)
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.patch_embed = PatchEmbed(patch_size, embed_dim, name="patch_embed")

        # Learnable [CLS] token
        self.cls_token = self.add_weight(
            shape=(1, 1, embed_dim),
            initializer="truncated_normal",
            trainable=True,
            name="cls_token",
        )
        # Learnable position embeddings
        self.pos_embed = self.add_weight(
            shape=(1, _N_TOKENS, embed_dim),
            initializer="truncated_normal",
            trainable=True,
            name="pos_embed",
        )
        self.pos_drop = layers.Dropout(dropout, name="pos_drop")

        self.blocks = [
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout,
                name=f"block_{i}",
            )
            for i in range(num_layers)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def call(self, x, training=False):
        # Patch embed → (B, 36, D)
        x = self.patch_embed(x, training=training)
        batch = ops.shape(x)[0]

        # Prepend [CLS] token
        cls = ops.tile(self.cls_token, (batch, 1, 1))  # (B, 1, D)
        x = ops.concatenate([cls, x], axis=1)       # (B, 37, D)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x, training=training)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Final layer norm
        x = self.norm(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], _N_TOKENS, self.embed_dim)


# ---------------------------------------------------------------------------
# ActorCritic (Keras Model)
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="gbagent")
class ActorCritic(keras.Model):
    """ViT → LSTM → (dpad_logits, btn_logits, value).

    Supports both Game Boy (6 btn actions) and GBA (8 btn actions with L/R).

    Parameters
    ----------
    hidden_dim : int
        LSTM hidden size (default 512).
    btn_size : int
        Size of the btn head.  6 for Game Boy, 8 for GBA (default 6).
    **vit_kwargs
        Forwarded to :class:`ViTEncoder`.  If *config* is given, *vit_kwargs*
        and *hidden_dim* are pulled from that object instead.

    Input
    -----
    ``obs`` : ``(B, 84, 84, 4)`` tensor
        Frame-stacked, preprocessed observations from :class:`GBAGEnv`.

    Output
    ------
    - ``dpad_logits`` : ``(B, 5)`` — NOOP / UP / DOWN / LEFT / RIGHT
    - ``btn_logits``  : ``(B, N)`` where N=6 (GB) or 8 (GBA)
    - ``value``       : ``(B, 1)`` — state-value estimate
    """

    def __init__(
        self,
        config=None,
        hidden_dim: int = 512,
        btn_size: int = 6,
        **vit_kwargs,
    ):
        if btn_size not in (6, 8):
            raise ValueError(f"btn_size must be 6 (GB) or 8 (GBA), got {btn_size}")
        self._btn_size = btn_size

        super().__init__()

        if config is not None:
            hidden_dim = config.hidden_dim
            btn_size = getattr(config, "btn_size", btn_size)
            vit_kwargs = {
                "patch_size": config.patch_size,
                "embed_dim": config.embed_dim,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "mlp_ratio": config.mlp_ratio,
                "dropout": config.dropout,
            }

        vit_kwargs.setdefault("patch_size", 16)
        vit_kwargs.setdefault("embed_dim", 256)
        vit_kwargs.setdefault("num_layers", 6)
        vit_kwargs.setdefault("num_heads", 8)
        vit_kwargs.setdefault("mlp_ratio", 4)
        vit_kwargs.setdefault("dropout", 0.1)

        self.encoder = ViTEncoder(**vit_kwargs, name="vit")
        self.lstm = layers.LSTM(
            hidden_dim,
            return_sequences=False,
            return_state=True,
            name="ac_lstm",
        )
        self.dpad_head = layers.Dense(5, name="dpad_head")
        self.btn_head = layers.Dense(btn_size, name="btn_head")
        self.value_head = layers.Dense(1, name="value_head")

    def call(self, obs, state=None, training=False):
        # ViT encoder → (B, 37, D)
        x = self.encoder(obs, training=training)

        # Mean-pool patch tokens to a single embedding per observation
        x = ops.mean(x, axis=1, keepdims=True)      # (B, 1, D)

        # LSTM over the single embedding with state carry-over
        x, h, c = self.lstm(x, initial_state=state)  # (B, hidden_dim)

        # Heads
        dpad_logits = self.dpad_head(x)
        btn_logits = self.btn_head(x)
        value = self.value_head(x)

        return dpad_logits, btn_logits, value, (h, c)
