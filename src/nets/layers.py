"""Attention, Learn to Solve Routing Problems! — shared attention layers (App. A)

Section references:
  App. A (eqs 10–14) — multi-head attention as weighted message passing.
  §3.1 / App. A — encoder layer: MHA + node-wise FF, each with skip + batch norm.

These layers are problem-agnostic: they operate on node embeddings only.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets.config import ModelConfig


class MultiHeadAttention(nn.Module):
    """App. A (eqs 10–14) — multi-head self-attention as weighted message passing.

    "we compute the value ... M = 8 times with different parameters, using
     d_k = d_v = d_h/M = 16 ... projected back ... using (d_h × d_v) matrices W^O_m."
    """

    def __init__(self, d_h: int, n_heads: int, bias: bool = False):
        super().__init__()
        assert d_h % n_heads == 0, "d_h must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_h // n_heads          # App. A — d_k = d_v = d_h/M = 16
        self.W_q = nn.Linear(d_h, d_h, bias=bias)  # all heads stacked (App. A eq 10)
        self.W_k = nn.Linear(d_h, d_h, bias=bias)
        self.W_v = nn.Linear(d_h, d_h, bias=bias)
        self.W_o = nn.Linear(d_h, d_h, bias=bias)  # App. A — W^O_m, heads concatenated

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Self-attention over all nodes.

        Args:
            h: node embeddings — shape: (batch, n, d_h)
        Returns:
            updated node embeddings h' — shape: (batch, n, d_h)
        """
        batch, n, _ = h.shape
        # Project to queries/keys/values and split into heads.
        q = self.W_q(h).view(batch, n, self.n_heads, self.d_k).transpose(1, 2)  # (batch, M, n, d_k)
        k = self.W_k(h).view(batch, n, self.n_heads, self.d_k).transpose(1, 2)  # (batch, M, n, d_k)
        v = self.W_v(h).view(batch, n, self.n_heads, self.d_k).transpose(1, 2)  # (batch, M, n, d_k)

        # App. A eq 11 — scaled dot-product compatibilities u_ij = q_i^T k_j / sqrt(d_k).
        u = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, M, n, n)
        a = F.softmax(u, dim=-1)                                        # App. A eq 12 — attention weights
        h_heads = torch.matmul(a, v)                                   # App. A eq 13 — convex combo of values: (batch, M, n, d_k)

        # App. A eq 14 — concatenate heads and project back to d_h via W^O.
        h_cat = h_heads.transpose(1, 2).contiguous().view(batch, n, -1)  # (batch, n, d_h)
        return self.W_o(h_cat)                                           # (batch, n, d_h)


class AttentionLayer(nn.Module):
    """§3.1 — one encoder layer: MHA sublayer + node-wise FF sublayer.

    Each sublayer has a skip-connection and batch normalization (add-then-BN):
      ĥ = BN(h + MHA(h));  h_out = BN(ĥ + FF(ĥ)).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mha = MultiHeadAttention(cfg.d_h, cfg.n_heads, bias=cfg.qkv_bias)
        # App. A — FF: hidden dim d_ff=512 with ReLU, then back to d_h.
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_h, cfg.d_ff),  # (batch, n, d_h) -> (batch, n, d_ff)
            nn.ReLU(),                      # §3.1 — "ReLu activation"
            nn.Linear(cfg.d_ff, cfg.d_h),  # (batch, n, d_ff) -> (batch, n, d_h)
        )
        # §3.1 / App. A eq 16 — batch norm with learnable affine params (per d_h feature).
        self.bn1 = nn.BatchNorm1d(cfg.d_h)
        self.bn2 = nn.BatchNorm1d(cfg.d_h)

    def _bn(self, bn: nn.BatchNorm1d, h: torch.Tensor) -> torch.Tensor:
        """Apply BatchNorm1d over the d_h feature dim, flattening (batch, n)."""
        batch, n, d_h = h.shape
        return bn(h.reshape(batch * n, d_h)).view(batch, n, d_h)  # normalize over batch*n

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (batch, n, d_h) -> (batch, n, d_h)."""
        h = self._bn(self.bn1, h + self.mha(h))   # §3.1 — ĥ = BN(h + MHA(h))
        h = self._bn(self.bn2, h + self.ff(h))    # §3.1 — h = BN(ĥ + FF(ĥ))
        return h
