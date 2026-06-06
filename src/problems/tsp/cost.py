"""TSP cost L(π): Euclidean length of the closed tour (§3, §4)."""

import torch


def tsp_tour_length(coords: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    """§3/§4 — L(π): Euclidean length of the closed tour π over the given nodes.

    The cost a TSP solution incurs is the sum of edge lengths along the visited
    permutation, returning to the start node (closed tour).

    Args:
        coords: node coordinates — shape: (batch, n, 2)
        tour:   visiting order π (node indices) — shape: (batch, n)

    Returns:
        tour length L(π) per instance — shape: (batch,)
    """
    batch, n = tour.shape
    # Gather coordinates in visiting order: (batch, n, 2)
    idx = tour.unsqueeze(-1).expand(batch, n, 2)  # (batch, n) -> (batch, n, 2)
    ordered = coords.gather(1, idx)               # (batch, n, 2)

    # Roll by one to get the "next" node along the tour (closing the loop).
    rolled = ordered.roll(shifts=-1, dims=1)      # (batch, n, 2)

    # Per-edge Euclidean distance, then sum over the n edges.
    seg = (ordered - rolled).norm(p=2, dim=-1)    # (batch, n, 2) -> (batch, n)
    return seg.sum(dim=1)                          # (batch, n) -> (batch,)
