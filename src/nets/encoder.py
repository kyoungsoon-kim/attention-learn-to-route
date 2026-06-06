"""Attention, Learn to Solve Routing Problems! — Encoder (§3.1)

Section references:
  §3.1 — input projection + N attention layers; graph embedding = node mean.

The encoder is problem-agnostic except for the *embedder*: the module that maps
raw problem inputs to the initial d_h-dimensional node embeddings h^(0). For TSP
this is a single linear layer over the 2-D coordinates; for problems with a depot
or extra node features (e.g. CVRP) the problem supplies a different embedder.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.nets.config import ModelConfig
from src.nets.layers import AttentionLayer


class Encoder(nn.Module):
    """§3.1 — input projection (embedder) + N attention layers; graph emb = node mean."""

    def __init__(self, cfg: ModelConfig, embedder: Optional[nn.Module] = None):
        super().__init__()
        # §3.1 — h^(0)_i = W^x x_i + b^x (learned linear projection, has bias).
        # Default embedder = TSP coordinate projection; problems may inject their own.
        self.embedder = embedder if embedder is not None else nn.Linear(cfg.d_x, cfg.d_h)
        self.layers = nn.ModuleList(AttentionLayer(cfg) for _ in range(cfg.n_layers))

    def forward(self, inputs):
        """
        Args:
            inputs: raw problem inputs consumed by the embedder. For TSP this is
                    node coordinates — shape: (batch, n, d_x).
        Returns:
            node_emb:  final node embeddings h^(N) — shape: (batch, n, d_h)
            graph_emb: graph embedding (mean of node emb) — shape: (batch, d_h)
        """
        h = self.embedder(inputs)         # raw inputs -> (batch, n, d_h)
        for layer in self.layers:         # §3.1 — N layers, NOT sharing parameters
            h = layer(h)                  # (batch, n, d_h)
        graph_emb = h.mean(dim=1)         # §3.1 — "mean of the final node embeddings"
        return h, graph_emb
