"""
Attention, Learn to Solve Routing Problems! — Attention Model assembly (TSP)

Paper: https://arxiv.org/abs/1803.08475

This module assembles the full Attention Model from the shared network components
(`src/nets/`) and a problem definition (`src/problems/`). It also re-exports the
encoder/decoder building blocks so existing imports (`from src.model import ...`)
keep working after the refactor that split the code into `nets/` and `problems/`.

Section references:
  §3.1 — Encoder: input projection + N attention layers (MHA + FF, skip + BN).
  §3.2 — Decoder: context node, M-head glimpse, single-head pointer (tanh clip, mask).
  App. A (eqs 10–16) — attention as message passing, multi-head, FF, batch norm.
"""

import math

import torch
import torch.nn as nn

# Re-exported building blocks (kept importable from src.model for back-compat).
from src.nets.config import ModelConfig
from src.nets.decoder import AttentionDecoder
from src.nets.encoder import Encoder
from src.nets.layers import AttentionLayer, MultiHeadAttention
from src.problems.cvrp import Decoder as CVRPDecoder, cvrp_embedder
from src.problems.tsp import Decoder as TSPDecoder, tsp_embedder

# Problem registry: name -> (embedder factory, decoder class). New routing
# problems register here; the encoder/decoder mechanics are shared.
_PROBLEMS = {
    "tsp": (tsp_embedder, TSPDecoder),
    "cvrp": (cvrp_embedder, CVRPDecoder),
}

# Back-compat alias: `from src.model import Decoder` historically meant the TSP decoder.
Decoder = TSPDecoder


class AttentionModel(nn.Module):
    """§3 — full Attention Model: encoder + autoregressive attention decoder."""

    def __init__(self, cfg: ModelConfig, problem: str = "tsp"):
        super().__init__()
        self.cfg = cfg
        if problem not in _PROBLEMS:
            raise ValueError(
                f"unknown problem {problem!r}; available: {sorted(_PROBLEMS)}")
        embedder_factory, decoder_cls = _PROBLEMS[problem]
        self.problem = problem
        self.encoder = Encoder(cfg, embedder=embedder_factory(cfg))
        self.decoder = decoder_cls(cfg)
        self._init_parameters()

    def _init_parameters(self) -> None:
        """§5 — initialize parameters Uniform(−1/√d, 1/√d), d = input (fan-in) dim."""
        for p in self.parameters():
            if p.dim() >= 2:
                fan_in = p.shape[-1]
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(p, -bound, bound)
            # 1-D params (BN affine, placeholders) keep their module-level init.

    def forward(self, inputs: torch.Tensor, decode_type: str = "sampling"):
        """
        Args:
            inputs: raw problem inputs (for TSP: node coordinates (batch, n, 2)).
            decode_type: "sampling" or "greedy".
        Returns:
            tour:    sequence of selected node indices — shape: (batch, T)
            log_lik: log-likelihood of the produced solution — shape: (batch,)
        """
        node_emb, graph_emb = self.encoder(inputs)
        return self.decoder(node_emb, graph_emb, decode_type, inputs)


__all__ = [
    "ModelConfig",
    "MultiHeadAttention",
    "AttentionLayer",
    "Encoder",
    "AttentionDecoder",
    "Decoder",
    "AttentionModel",
]
