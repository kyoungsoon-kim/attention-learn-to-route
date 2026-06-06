"""CVRP instance generation (Kool et al. 2019, Appendix — VRP).

Section references:
  App. (VRP) — n customers + 1 depot, coordinates uniform in [0,1]^2; customer
  demands δ_i ~ Uniform{1,...,9}; vehicle capacity D depends on n (20→30, 50→40,
  100→50). Demands are normalized by D so the normalized capacity is 1.
"""

import torch

# App. — capacity per graph size (normalized demands divide by this).
CAPACITIES = {10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}


def capacity_for(graph_size: int) -> float:
    """Vehicle capacity D for `graph_size` customers (App.)."""
    if graph_size in CAPACITIES:
        return CAPACITIES[graph_size]
    # Fallback for sizes not tabulated in the paper (keeps the demand scale sane).
    return 30.0 + (graph_size - 20) * (10.0 / 30.0)


def generate_cvrp_instances(batch_size: int, graph_size: int,
                            device: str = "cpu",
                            generator: torch.Generator | None = None) -> dict:
    """App. — sample CVRP instances.

    Args:
        batch_size: number of instances B.
        graph_size: number of customers n (depot is separate).
        generator:  optional RNG for a reproducible (e.g. held-out) set.
    Returns:
        dict with
          depot:  (batch, 2)        — depot coordinates in [0,1]^2
          loc:    (batch, n, 2)     — customer coordinates in [0,1]^2
          demand: (batch, n)        — normalized demands δ_i / D in (0, 1]
    """
    g = {"generator": generator} if generator is not None else {}
    depot = torch.rand(batch_size, 2, device=device, **g)
    loc = torch.rand(batch_size, graph_size, 2, device=device, **g)
    # App. — integer demands Uniform{1,...,9}; normalize by capacity D.
    cap = capacity_for(graph_size)
    demand_int = torch.randint(1, 10, (batch_size, graph_size), device=device, **g)
    demand = demand_int.to(torch.float) / cap
    return {"depot": depot, "loc": loc, "demand": demand}
