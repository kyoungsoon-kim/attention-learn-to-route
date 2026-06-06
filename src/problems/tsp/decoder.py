"""TSP decoder wrapper (§3.2).

Owns the TSP-specific learned placeholders v^l, v^f used as the last/first node at
t=1, and assembles a `TSPState` for the shared `AttentionDecoder` to roll out.
The shared glimpse/pointer mechanics live in `src/nets/decoder.py`.
"""

import torch
import torch.nn as nn

from src.nets.config import ModelConfig
from src.nets.decoder import AttentionDecoder
from src.problems.tsp.state import TSPState


class Decoder(nn.Module):
    """§3.2 — TSP decoder: context [graph, last, first], glimpse + pointer rollout."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Shared glimpse/pointer; TSP context width = 3·d_h.
        self.core = AttentionDecoder(cfg, context_dim=TSPState.CONTEXT_MULT * cfg.d_h)
        # §3.2 — learned placeholders v^l, v^f used for the first/last node at t=1.
        self.placeholder_first = nn.Parameter(torch.empty(cfg.d_h))
        self.placeholder_last = nn.Parameter(torch.empty(cfg.d_h))
        nn.init.uniform_(self.placeholder_first, -1.0, 1.0)
        nn.init.uniform_(self.placeholder_last, -1.0, 1.0)

    def forward(self, node_emb: torch.Tensor, graph_emb: torch.Tensor,
                decode_type: str = "sampling", inputs=None):
        """§3.2 — roll out a full TSP tour.

        Args:
            node_emb:  encoder node embeddings — shape: (batch, n, d_h)
            graph_emb: graph embedding — shape: (batch, d_h)
            decode_type: "sampling" (REINFORCE) or "greedy" (baseline / eval).
            inputs:    unused for TSP (decoding needs only embeddings); kept for a
                       uniform decoder signature across problems.
        Returns:
            tour:    visiting order π — shape: (batch, n)
            log_lik: log-likelihood of the produced tour — shape: (batch,)
        """
        state = TSPState(node_emb, graph_emb,
                         self.placeholder_first, self.placeholder_last)
        return self.core(node_emb, state, decode_type)
