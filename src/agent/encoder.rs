use candle_core::{Device, Tensor, Module};
use candle_nn::{layer_norm, linear, Linear, VarBuilder, VarMap};

/// ViT-Small encoder: PatchEmbed → 6x Transformer → LayerNorm
pub struct ViTEncoder {
    patch_embed: PatchEmbedding,
    transformer: Vec<TransformerBlock>,
    norm: candle_nn::LayerNorm,
    var_map: VarMap,
    device: Device,
}

pub struct ViTConfig {
    pub input_channels: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub image_h: usize,
    pub image_w: usize,
}

impl Default for ViTConfig {
    fn default() -> Self {
        Self {
            input_channels: 4,
            patch_size: 16,
            embed_dim: 384,
            num_layers: 6,
            num_heads: 6,
            mlp_ratio: 4.0,
            image_h: 144,
            image_w: 160,
        }
    }
}

impl ViTEncoder {
    pub fn new(cfg: &ViTConfig, device: &Device) -> anyhow::Result<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, device);

        let patch_embed = PatchEmbedding::new(cfg, vb.pp("patch_embed"))?;
        let mut transformer = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let block = TransformerBlock::new(cfg, vb.pp(&format!("transformer_{i}")))?;
            transformer.push(block);
        }
        let norm = layer_norm(cfg.embed_dim, cfg.embed_dim as f64, vb.pp("norm"))?;

        Ok(Self { patch_embed, transformer, norm, var_map, device: device.clone() })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.patch_embed.forward(x)?;
        let mut x = x;
        for block in &self.transformer {
            x = block.forward(&x)?;
        }
        x = self.norm.forward(&x)?;
        // Mean pool over patch tokens
        // (B, N, D) → (B, D)
        x.mean(1).map_err(Into::into)
    }

    pub fn var_map(&self) -> &VarMap {
        &self.var_map
    }
}

/// Patch embedding: Conv2d to split image into patches + linear projection + pos embed
struct PatchEmbedding {
    proj: candle_nn::Conv2d,
    pos_embed: Tensor,
    num_patches: usize,
    embed_dim: usize,
}

impl PatchEmbedding {
    fn new(cfg: &ViTConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        let num_patches_h = cfg.image_h / cfg.patch_size;
        let num_patches_w = cfg.image_w / cfg.patch_size;
        let num_patches = num_patches_h * num_patches_w;

        let proj = candle_nn::conv2d(
            cfg.input_channels,
            cfg.embed_dim,
            cfg.patch_size,
            candle_nn::Conv2dConfig {
                stride: cfg.patch_size,
                padding: 0,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;

        // Learnable position embedding
        let pos_embed = vb.get(
            (1, num_patches, cfg.embed_dim),
            "pos_embed",
        )?;

        Ok(Self { proj, pos_embed, num_patches, embed_dim: cfg.embed_dim })
    }

    fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        // x: (B, C, H, W) → (B, D, H', W')
        let x = self.proj.forward(x)?;
        // (B, D, H', W') → (B, H'*W', D)
        let (b, _d, h, w) = x.dims4()?;
        let x = x.reshape((b, self.embed_dim, h * w))?.transpose(1, 2)?;
        // Add position embedding
        Ok(x.broadcast_add(&self.pos_embed)?)
    }
}

/// Single Transformer block: MSA + MLP with residual connections
struct TransformerBlock {
    ln1: candle_nn::LayerNorm,
    attention: MultiHeadAttention,
    ln2: candle_nn::LayerNorm,
    mlp: Mlp,
}

impl TransformerBlock {
    fn new(cfg: &ViTConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        let ln1 = layer_norm(cfg.embed_dim, 1e-5, vb.pp("ln1"))?;
        let attention = MultiHeadAttention::new(cfg, vb.pp("attn"))?;
        let ln2 = layer_norm(cfg.embed_dim, 1e-5, vb.pp("ln2"))?;
        let mlp = Mlp::new(cfg.embed_dim, (cfg.embed_dim as f64 * cfg.mlp_ratio) as usize, vb.pp("mlp"))?;
        Ok(Self { ln1, attention, ln2, mlp })
    }

    fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let residual = x.clone();
        let x = self.ln1.forward(x)?;
        let x = self.attention.forward(&x)?;
        let x = (residual + x)?;
        let residual = x.clone();
        let x = self.ln2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((residual + x)?)
    }
}

/// Multi-Head Self Attention
struct MultiHeadAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    fn new(cfg: &ViTConfig, vb: VarBuilder) -> anyhow::Result<Self> {
        let num_heads = cfg.num_heads;
        let head_dim = cfg.embed_dim / num_heads;
        let qkv = linear(cfg.embed_dim, cfg.embed_dim * 3, vb.pp("qkv"))?;
        let proj = linear(cfg.embed_dim, cfg.embed_dim, vb.pp("proj"))?;
        Ok(Self { qkv, proj, num_heads, head_dim })
    }

    fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let (b, n, d) = x.dims3()?;
        let qkv = self.qkv.forward(x)?;
        let qkv = qkv.reshape((b, n, 3, self.num_heads, self.head_dim))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        // Transpose to (B, H, N, D_head)
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        // Attention scores
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = (attn / (self.head_dim as f64).sqrt())?;
        let attn = candle_nn::ops::softmax(&attn, 3)?;
        // Weighted sum
        let out = attn.matmul(&v)?;
        // Transpose back: (B, H, N, D_head) → (B, N, D)
        let out = out.transpose(1, 2)?.reshape((b, n, d))?;
        Ok(self.proj.forward(&out)?)
    }
}

/// MLP: Linear → GELU → Linear
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(dim: usize, hidden: usize, vb: VarBuilder) -> anyhow::Result<Self> {
        let fc1 = linear(dim, hidden, vb.pp("fc1"))?;
        let fc2 = linear(hidden, dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.fc1.forward(x)?;
        // GELU activation
        let x = x.gelu()?;
        Ok(self.fc2.forward(&x)?)
    }
}
