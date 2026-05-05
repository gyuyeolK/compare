"""
Small GPT-style decoder-only Transformer.

Following the architectural choices used in the Dion / modded-nanoGPT papers:
  - RoPE positional embeddings
  - Non-parametric RMSNorm
  - No biases in linear layers
  - Squared-ReLU activation in MLP

Tunable via a Config dataclass for easy scaling experiments.
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    seq_len: int = 512
    mlp_mult: int = 4
    rope_base: float = 10000.0


# --------------------------- RMSNorm (non-parametric) -----------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms


# --------------------------- Rotary positional embedding -------------------

def precompute_rope(head_dim, seq_len, base=10000.0, device=None, dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)  with D even
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :x.size(-2), :]
    sin = sin[None, None, :x.size(-2), :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


# --------------------------- Causal multi-head attention -------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(
                f"d_model ({cfg.d_model}) must be divisible by n_heads "
                f"({cfg.n_heads}). Try --n_heads {cfg.d_model // 64} "
                f"or pick a d_model that divides evenly."
            )
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x)                              # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape -> (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # SDPA with causal mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# --------------------------- MLP block --------------------------------------

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.d_model
        self.fc = nn.Linear(cfg.d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        # squared ReLU
        return self.proj(F.relu(self.fc(x)).pow(2))


# --------------------------- Transformer block ------------------------------

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------- Full GPT ---------------------------------------

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # init
        for p in self.parameters():
            if p.ndim == 2:
                nn.init.normal_(p, mean=0.0, std=0.02)

        # cached RoPE buffers (filled lazily on first forward)
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

    def _ensure_rope(self, T, device, dtype):
        if self.rope_cos.numel() < T * (self.cfg.d_model // self.cfg.n_heads // 2):
            cos, sin = precompute_rope(
                self.cfg.d_model // self.cfg.n_heads,
                seq_len=max(T, self.cfg.seq_len),
                base=self.cfg.rope_base,
                device=device,
                dtype=dtype,
            )
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        x = self.embed(idx)
        self._ensure_rope(T, device, x.dtype)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n
