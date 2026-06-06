"""TSP decode state (§3.2).

Context = [graph_emb, last_node, first_node] (3·d_h). At t=1 the last/first nodes
are undefined, so learned placeholders v^l, v^f are used (supplied by the decoder).
Mask = already-visited nodes. The rollout terminates after exactly n steps (every
node visited once — a Hamiltonian tour).
"""

import torch

from src.problems.state import DecodeState


class TSPState(DecodeState):
    """§3.2 — autoregressive TSP state: visited mask + first/last node tracking."""

    #: context width = [graph_emb, last_node, first_node] = 3·d_h
    CONTEXT_MULT = 3

    def __init__(self, node_emb: torch.Tensor, graph_emb: torch.Tensor,
                 placeholder_first: torch.Tensor, placeholder_last: torch.Tensor):
        """
        Args:
            node_emb:  encoder node embeddings — shape: (batch, n, d_h)
            graph_emb: graph embedding — shape: (batch, d_h)
            placeholder_first / placeholder_last: learned v^f, v^l — shape: (d_h,)
        """
        self.node_emb = node_emb
        self.graph_emb = graph_emb
        self.ph_first = placeholder_first
        self.ph_last = placeholder_last

        batch, n, d_h = node_emb.shape
        self.batch, self.n, self.d_h = batch, n, d_h
        device = node_emb.device
        self.mask = torch.zeros(batch, n, dtype=torch.bool, device=device)  # True = visited
        self.arange = torch.arange(batch, device=device)
        self.first_idx = torch.zeros(batch, dtype=torch.long, device=device)
        self.last_idx = torch.zeros(batch, dtype=torch.long, device=device)
        self.step = 0

    def get_mask(self) -> torch.Tensor:
        return self.mask  # (batch, n)

    def get_context(self) -> torch.Tensor:
        # §3.2 — context = [graph_emb, last_node_emb, first_node_emb].
        if self.step == 0:
            first = self.ph_first.expand(self.batch, self.d_h)  # v^f
            last = self.ph_last.expand(self.batch, self.d_h)    # v^l
        else:
            first = self.node_emb[self.arange, self.first_idx]  # (batch, d_h)
            last = self.node_emb[self.arange, self.last_idx]    # (batch, d_h)
        return torch.cat([self.graph_emb, last, first], dim=-1)  # (batch, 3·d_h)

    def update(self, selected: torch.Tensor) -> None:
        self.mask = self.mask.scatter(1, selected.unsqueeze(1), True)  # mark visited
        if self.step == 0:
            self.first_idx = selected
        self.last_idx = selected
        self.step += 1

    def all_done(self) -> bool:
        return self.step >= self.n  # every node visited once
