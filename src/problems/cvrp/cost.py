"""CVRP route length (Kool et al. 2019, Appendix — VRP).

The route is the sequence of nodes the vehicle visits (0 = depot, possibly
repeated). Its length is the depot→first leg, plus every consecutive leg along the
sequence, plus the final leg back to the depot.
"""

import torch


def cvrp_route_length(inputs: dict, tour: torch.Tensor) -> torch.Tensor:
    """App. (VRP) — total Euclidean length of the CVRP route.

    Args:
        inputs: dict with depot (batch, 2) and loc (batch, n, 2).
        tour:   node-index sequence — shape: (batch, T); 0 = depot, 1..n = customers.
    Returns:
        route length per instance — shape: (batch,)
    """
    depot, loc = inputs["depot"], inputs["loc"]
    coords = torch.cat([depot.unsqueeze(1), loc], dim=1)     # (batch, n+1, 2); 0 = depot
    idx = tour.unsqueeze(-1).expand(-1, -1, 2)               # (batch, T, 2)
    d = coords.gather(1, idx)                                # (batch, T, 2) — visited coords in order

    internal = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(dim=1)  # consecutive legs
    to_first = (d[:, 0] - depot).norm(p=2, dim=-1)                  # depot -> first node
    from_last = (d[:, -1] - depot).norm(p=2, dim=-1)               # last node -> depot
    return internal + to_first + from_last                          # (batch,)
