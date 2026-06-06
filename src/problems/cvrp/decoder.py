"""CVRP decoder wrapper (Kool et al. 2019, Appendix — VRP).

Reuses the shared glimpse/pointer `AttentionDecoder`. The only CVRP specifics are
the context width (2·d_h + 1, because the context carries the remaining-capacity
scalar) and the `CVRPState`, which needs the demands to build the capacity mask.
No learned t=1 placeholder is required (the rollout starts at the real depot node).
"""

import torch
import torch.nn as nn

from src.nets.config import ModelConfig
from src.nets.decoder import AttentionDecoder
from src.problems.cvrp.state import CVRPState


class Decoder(nn.Module):
    """App. (VRP) — context [graph, last, remaining capacity], glimpse + pointer rollout."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        context_dim = 2 * cfg.d_h + CVRPState.CONTEXT_DIM_EXTRA
        self.core = AttentionDecoder(cfg, context_dim=context_dim)

    def forward(self, node_emb: torch.Tensor, graph_emb: torch.Tensor,
                decode_type: str = "sampling", inputs: dict = None):
        """
        Args:
            node_emb:  encoder node embeddings incl. depot — shape: (batch, n+1, d_h)
            graph_emb: graph embedding — shape: (batch, d_h)
            inputs:    instance dict; `demand` (batch, n) is required for masking.
        Returns:
            tour:    node-index sequence (0 = depot, may repeat) — shape: (batch, T)
            log_lik: log-likelihood of the produced route — shape: (batch,)
        """
        if inputs is None:
            raise ValueError("CVRP decoder requires `inputs` (needs demands).")
        state = CVRPState(node_emb, graph_emb, inputs["demand"])
        return self.core(node_emb, state, decode_type)
