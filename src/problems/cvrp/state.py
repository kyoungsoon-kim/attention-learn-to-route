"""CVRP decode state (Kool et al. 2019, Appendix — VRP).

Node 0 is the depot; nodes 1..n are customers. The vehicle starts at the depot
with a full (normalized) capacity of 1.

Context = [graph_emb, last_node_emb, remaining_capacity] (2·d_h + 1). No learned
t=1 placeholder is needed because the rollout always starts at the depot, whose
embedding is a real node embedding.

Masking (App.):
  - a customer is forbidden if already visited, or if its demand exceeds the
    remaining capacity;
  - the depot is forbidden only when the vehicle is currently at the depot AND at
    least one feasible customer remains (no two depot visits in a row while there
    is still something to serve). Once every customer is served the depot is
    always allowed, so finished instances simply park at the depot.

Termination: the rollout ends once every customer in the batch has been visited.
"""

import torch

from src.problems.state import DecodeState

# Normalized vehicle capacity (demands are pre-divided by D in the data).
VEHICLE_CAPACITY = 1.0
_EPS = 1e-6  # tolerance so a customer that exactly fits is still feasible


class CVRPState(DecodeState):
    """App. (VRP) — capacity-aware autoregressive state with depot returns."""

    #: context width = [graph_emb, last_node, remaining_capacity] = 2·d_h + 1
    CONTEXT_DIM_EXTRA = 1  # the scalar remaining-capacity feature

    def __init__(self, node_emb: torch.Tensor, graph_emb: torch.Tensor,
                 demand: torch.Tensor):
        """
        Args:
            node_emb:  encoder node embeddings incl. depot — shape: (batch, n+1, d_h)
            graph_emb: graph embedding — shape: (batch, d_h)
            demand:    normalized customer demands — shape: (batch, n)
        """
        self.node_emb = node_emb
        self.graph_emb = graph_emb
        self.demand = demand                       # (batch, n)

        batch, n_plus_1, d_h = node_emb.shape
        self.batch, self.n, self.d_h = batch, n_plus_1 - 1, d_h
        device = node_emb.device
        self.arange = torch.arange(batch, device=device)
        self.visited = torch.zeros(batch, self.n, dtype=torch.bool, device=device)  # customers
        self.used_capacity = torch.zeros(batch, device=device)                      # in [0, 1]
        self.prev_a = torch.zeros(batch, dtype=torch.long, device=device)           # start at depot (0)

    def _customer_mask(self) -> torch.Tensor:
        """Forbidden customers: visited OR demand exceeds remaining capacity — (batch, n)."""
        remaining = VEHICLE_CAPACITY - self.used_capacity                # (batch,)
        too_big = self.demand > remaining.unsqueeze(1) + _EPS            # (batch, n)
        return self.visited | too_big

    def get_mask(self) -> torch.Tensor:
        mask_cust = self._customer_mask()                               # (batch, n)
        feasible_exists = (~mask_cust).any(dim=1)                       # (batch,)
        at_depot = self.prev_a == 0                                     # (batch,)
        # Depot forbidden only while sitting at the depot with a servable customer left.
        mask_depot = at_depot & feasible_exists                        # (batch,)
        return torch.cat([mask_depot.unsqueeze(1), mask_cust], dim=1)  # (batch, n+1)

    def get_context(self) -> torch.Tensor:
        last_emb = self.node_emb[self.arange, self.prev_a]             # (batch, d_h)
        remaining = (VEHICLE_CAPACITY - self.used_capacity).unsqueeze(1)  # (batch, 1)
        return torch.cat([self.graph_emb, last_emb, remaining], dim=-1)   # (batch, 2·d_h+1)

    def update(self, selected: torch.Tensor) -> None:
        """Advance state given selected node indices in [0, n] (0 = depot)."""
        is_depot = selected == 0                                       # (batch,)
        cust_idx = (selected - 1).clamp(min=0)                         # depot rows -> 0 (ignored)

        # Mark the chosen customer visited (depot selection marks nothing).
        newly = torch.zeros_like(self.visited)
        newly[self.arange, cust_idx] = ~is_depot                       # only where a customer was chosen
        self.visited = self.visited | newly

        # Capacity: reset at the depot, otherwise add the served customer's demand.
        served = self.demand[self.arange, cust_idx]                    # (batch,)
        new_used = self.used_capacity + torch.where(
            is_depot, torch.zeros_like(served), served)
        self.used_capacity = torch.where(
            is_depot, torch.zeros_like(new_used), new_used)

        self.prev_a = selected

    def all_done(self) -> bool:
        return bool(self.visited.all())
