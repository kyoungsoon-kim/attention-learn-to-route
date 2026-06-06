"""Attention, Learn to Solve Routing Problems! — generic decoder (§3.2)

Section references:
  §3.2 — autoregressive decoder: context node, M-head glimpse, single-head pointer
         with tanh logit clipping (C=10) and a visited/forbidden-node mask.

This decoder is problem-agnostic. The glimpse keys/values, the pointer keys, and
the (context -> query) projection are shared mechanisms; everything that differs
between routing problems — what the context vector is, which nodes are masked,
and when the rollout terminates — is delegated to a `DecodeState` object (see
`src/problems/state.py`). The state's context width is fixed per problem and
passed in as `context_dim`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets.config import ModelConfig


class AttentionDecoder(nn.Module):
    """§3.2 — M-head glimpse followed by a single-head pointer, driven by a state."""

    def __init__(self, cfg: ModelConfig, context_dim: int):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_h = cfg.d_h
        self.d_k = cfg.d_h // cfg.n_heads  # glimpse head dim
        self.tanh_clip = cfg.tanh_clip

        # §3.2 — W^Q maps the (problem-specific) context vector to a d_h query, then
        # split into M glimpse heads. `context_dim` = width of the state's context.
        self.W_q_glimpse = nn.Linear(context_dim, cfg.d_h, bias=cfg.qkv_bias)
        # Glimpse keys/values from node embeddings (App. A eq 10).
        self.W_k_glimpse = nn.Linear(cfg.d_h, cfg.d_h, bias=cfg.qkv_bias)
        self.W_v_glimpse = nn.Linear(cfg.d_h, cfg.d_h, bias=cfg.qkv_bias)
        self.W_o_glimpse = nn.Linear(cfg.d_h, cfg.d_h, bias=cfg.qkv_bias)
        # §3.2 — final single-head pointer layer (M=1, d_k=d_h); query from glimpse output.
        self.W_q_logit = nn.Linear(cfg.d_h, cfg.d_h, bias=cfg.qkv_bias)
        self.W_k_logit = nn.Linear(cfg.d_h, cfg.d_h, bias=cfg.qkv_bias)

    def _glimpse(self, query: torch.Tensor, node_emb: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        """§3.2 — M-head attention from the single context query over all nodes.

        Args:
            query:    projected context query (W^Q·context) — shape: (batch, d_h)
            node_emb: node embeddings — shape: (batch, n, d_h)
            mask:     True where a node is forbidden — shape: (batch, n)
        Returns:
            new context embedding h_c^{+1} — shape: (batch, d_h)
        """
        batch, n, _ = node_emb.shape
        q = query.view(batch, self.n_heads, 1, self.d_k)          # (batch, M, 1, d_k)
        k = self.W_k_glimpse(node_emb).view(batch, n, self.n_heads, self.d_k).transpose(1, 2)  # (batch, M, n, d_k)
        v = self.W_v_glimpse(node_emb).view(batch, n, self.n_heads, self.d_k).transpose(1, 2)  # (batch, M, n, d_k)

        u = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, M, 1, n)
        # §3.2 — mask forbidden nodes: u_(c)j = −∞.
        u = u.masked_fill(mask[:, None, None, :], float("-inf"))
        a = F.softmax(u, dim=-1)                                   # (batch, M, 1, n)
        h_c = torch.matmul(a, v)                                  # (batch, M, 1, d_k)
        h_c = h_c.transpose(1, 2).contiguous().view(batch, 1, self.d_h).squeeze(1)  # (batch, d_h)
        return self.W_o_glimpse(h_c)                              # (batch, d_h)

    def forward(self, node_emb: torch.Tensor, state, decode_type: str = "sampling"):
        """§3.2 — roll out a full solution autoregressively, driven by `state`.

        Args:
            node_emb:  encoder node embeddings — shape: (batch, n, d_h)
            state:     a DecodeState providing context / mask / update / all_done.
            decode_type: "sampling" (REINFORCE) or "greedy" (baseline / eval).
        Returns:
            tour:    sequence of selected node indices — shape: (batch, T)
            log_lik: sum_t log p_θ(π_t | s, π_<t) — shape: (batch,)
        """
        batch, n, d_h = node_emb.shape
        device = node_emb.device

        # Precompute logit keys once (App. A — keys depend only on node embeddings).
        k_logit = self.W_k_logit(node_emb)            # (batch, n, d_h)
        arange = torch.arange(batch, device=device)

        tours = []
        log_probs = []
        while not state.all_done():
            mask = state.get_mask()                              # (batch, n) — True = forbidden
            ctx = state.get_context()                            # (batch, context_dim)

            # Glimpse: project context to query heads, attend over nodes.
            q_glimpse = self.W_q_glimpse(ctx)                    # (batch, d_h)
            glimpse = self._glimpse(q_glimpse, node_emb, mask)   # (batch, d_h)

            # §3.2 — single-head pointer logits with tanh clipping then masking.
            q_logit = self.W_q_logit(glimpse)                    # (batch, d_h)
            logits = torch.matmul(q_logit.unsqueeze(1), k_logit.transpose(-2, -1)).squeeze(1)
            logits = logits / math.sqrt(d_h)                     # (batch, n); d_k = d_h (M=1)
            logits = self.tanh_clip * torch.tanh(logits)         # §3.2 — clip within [−C, C]
            logits = logits.masked_fill(mask, float("-inf"))     # §3.2 — mask forbidden (after clip)

            log_p = F.log_softmax(logits, dim=-1)                # (batch, n)

            if decode_type == "greedy":
                node = log_p.argmax(dim=-1)                      # §5 — select best action
            elif decode_type == "sampling":
                node = torch.distributions.Categorical(logits=log_p).sample()  # §5 — sample
            else:
                raise ValueError(f"unknown decode_type: {decode_type}")

            log_probs.append(log_p[arange, node])                # (batch,)
            tours.append(node)
            state.update(node)                                   # advance problem state

        tour = torch.stack(tours, dim=1)                         # (batch, T)
        log_lik = torch.stack(log_probs, dim=1).sum(dim=1)       # (batch,)
        return tour, log_lik
