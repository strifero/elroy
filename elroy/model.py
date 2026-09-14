"""The Elroy transformer. Chapter 3.

A Llama-style decoder-only transformer:

    tokens -> embedding -> [Block x n_layer] -> RMSNorm -> logits (tied to embedding)

    Block(x):  x = x + Attention(RMSNorm(x))
               x = x + SwiGLU(RMSNorm(x))

Everything is here, in the order a forward pass uses it: RMSNorm, rotary
position embeddings, causal self-attention, the SwiGLU feed-forward, the block,
and the full model with its loss. There are no biases anywhere; every current
small model dropped them and nothing was lost.

Attention has two implementations. `naive_attention` is the textbook math, kept
so you can read it and so the test can check the fast path against it.
`F.scaled_dot_product_attention` is what training uses; on Ampere it dispatches
to a fused flash-attention kernel that never materializes the T x T score matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    d_ff: int = 1408
    seq_len: int = 1024
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


class RMSNorm(nn.Module):
    """Root-mean-square normalization: rescale x to unit RMS, then a learned gain.

    LayerNorm without the mean subtraction and without the bias. Cheaper, and in
    practice just as good.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute in fp32 for stability, cast back to the input dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).type_as(x)


# ------------------------------------------------------------------ rotary embeddings
def rope_cos_sin(seq_len: int, head_dim: int, theta: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables of shape (seq_len, head_dim // 2).

    Pair i of the head vector rotates at frequency theta ** (-2i / head_dim), so
    the first pairs spin fast (local position) and the last spin slowly (global).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)               # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (x[2i], x[2i+1]) pair of x by position-dependent angles.

    x: (B, H, T, D). cos/sin: (T, D/2). Rotating the query and key by the same
    per-position angle makes their dot product depend only on the distance
    between them, which is the whole point.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :].type_as(x)
    sin = sin[None, None, :, :].type_as(x)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# --------------------------------------------------------------------- attention
def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal softmax attention written out. q, k, v: (B, H, T, D)."""
    T = q.size(-2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))       # (B, H, T, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~mask, float("-inf"))              # no peeking at the future
    return scores.softmax(dim=-1) @ v                              # (B, H, T, D)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.use_sdpa = True

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, D)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.use_sdpa:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.dropout if self.training else 0.0)
        else:
            y = naive_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)           # merge heads
        return self.proj(y)


class SwiGLU(nn.Module):
    """Feed-forward with a gated activation: down(silu(gate(x)) * up(x)).

    Three matrices instead of GELU's two, which is why d_ff is about 2.7x d_model
    rather than 4x: same parameter count, better results.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class Elroy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_cos_sin(cfg.seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # Residual-branch output projections get a smaller init so the residual
        # stream does not grow with depth (the GPT-2 trick).
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None):
        """idx: (B, T) token ids. Returns (logits, loss); loss is None without targets.

        loss_mask (B, T) of 0/1 lets the SFT chapter skip the loss on prompt tokens.
        """
        B, T = idx.shape
        assert T <= self.cfg.seq_len, f"sequence length {T} > {self.cfg.seq_len}"
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            per_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1),
                                      ignore_index=-1, reduction="none")
            if loss_mask is not None:
                m = loss_mask.reshape(-1).float()
                loss = (per_tok * m).sum() / m.sum().clamp(min=1)
            else:
                loss = per_tok.mean()
        return logits, loss

    # ------------------------------------------------------------- bookkeeping
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def flops_per_token(self) -> float:
        """Training FLOPs per token: 6 x params (fwd+bwd matmuls) plus attention scores.

        The attention term is 12 x n_layer x d_model x seq_len: the QK^T and AV
        products cost 2 x T x d per token each, forward and backward. Used for MFU.
        """
        c = self.cfg
        return 6 * self.num_params() + 12 * c.n_layer * c.d_model * c.seq_len

    def param_groups(self, weight_decay: float):
        """Decay the matrices, not the norms or the embedding gains."""
        decay = [p for n, p in self.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in self.named_parameters() if p.dim() < 2]
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]


def count_params_by_formula(cfg: ModelConfig) -> int:
    """The arithmetic the chapter does on paper; the test checks the model matches."""
    per_block = (4 * cfg.d_model * cfg.d_model          # q, k, v, out projections
                 + 3 * cfg.d_model * cfg.d_ff           # gate, up, down
                 + 2 * cfg.d_model)                      # two RMSNorm gains
    embed = cfg.vocab_size * cfg.d_model
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model
    return embed + cfg.n_layer * per_block + cfg.d_model + head
