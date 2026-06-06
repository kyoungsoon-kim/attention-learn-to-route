"""CVRP node embedder (Kool et al. 2019, Appendix — VRP).

The depot and the customers carry different input features, so — unlike TSP — the
encoder uses *two* input projections (App.): the depot is embedded from its 2-D
coordinates, each customer from its 2-D coordinates plus its (normalized) demand.
The depot is node 0; customers follow.
"""

import torch
import torch.nn as nn

from src.nets.config import ModelConfig


class CVRPEmbedder(nn.Module):
    """App. — separate linear projections for the depot and the customers."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.depot_proj = nn.Linear(2, cfg.d_h)       # depot: coordinates only
        self.cust_proj = nn.Linear(2 + 1, cfg.d_h)    # customer: coordinates + demand

    def forward(self, inputs: dict) -> torch.Tensor:
        """
        Args:
            inputs: dict with depot (batch, 2), loc (batch, n, 2), demand (batch, n).
        Returns:
            node embeddings h^(0) — shape: (batch, n+1, d_h); index 0 = depot.
        """
        depot, loc, demand = inputs["depot"], inputs["loc"], inputs["demand"]
        depot_emb = self.depot_proj(depot).unsqueeze(1)             # (batch, 1, d_h)
        cust_feat = torch.cat([loc, demand.unsqueeze(-1)], dim=-1)  # (batch, n, 3)
        cust_emb = self.cust_proj(cust_feat)                        # (batch, n, d_h)
        return torch.cat([depot_emb, cust_emb], dim=1)             # (batch, n+1, d_h)
