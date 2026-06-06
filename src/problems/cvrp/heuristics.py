"""Classic construction heuristics for CVRP — baselines for the Attention Model.

Two textbook heuristics, used as non-learned reference points:

  - **Nearest Neighbor (NN)** — greedily extend the current route to the closest
    feasible unvisited customer; return to the depot (resetting load) when nothing
    fits. Simple but weak.
  - **Clarke-Wright Savings (CW)** — start with one route per customer and greedily
    merge routes by descending "savings" s(i,j) = d(0,i) + d(0,j) − d(i,j) while the
    merged demand fits the vehicle. A strong classical baseline.

Both operate on a single instance with **normalized** demands (capacity = 1, as in
`src/problems/cvrp`). They return a tour as a list of node indices (0 = depot,
1..n = customers), matching `cvrp_route_length`'s convention.
"""

from typing import Callable, List

import numpy as np
import torch

from src.problems.cvrp.cost import cvrp_route_length

CAPACITY = 1.0  # normalized vehicle capacity
_EPS = 1e-6


def nearest_neighbor_tour(depot: np.ndarray, loc: np.ndarray,
                          demand: np.ndarray) -> List[int]:
    """Nearest-neighbor CVRP construction. Returns a node-index tour (0 = depot)."""
    n = len(loc)
    coords = np.vstack([depot[None, :], loc])  # (n+1, 2); row 0 = depot
    visited = np.zeros(n, dtype=bool)
    tour: List[int] = []
    cur = 0           # current node index (0 = depot)
    load = 0.0

    while not visited.all():
        feasible = np.where(~visited & (demand <= CAPACITY - load + _EPS))[0]
        if feasible.size == 0:
            tour.append(0)          # back to depot, reset capacity
            cur, load = 0, 0.0
            continue
        d = np.linalg.norm(loc[feasible] - coords[cur], axis=1)
        j = int(feasible[int(np.argmin(d))])
        visited[j] = True
        load += demand[j]
        tour.append(j + 1)          # customers are 1..n
        cur = j + 1
    return tour


def clarke_wright_tour(depot: np.ndarray, loc: np.ndarray,
                       demand: np.ndarray) -> List[int]:
    """Clarke-Wright savings CVRP construction. Returns a node-index tour (0 = depot)."""
    n = len(loc)
    d0 = np.linalg.norm(loc - depot[None, :], axis=1)  # depot <-> customer

    routes = {i: [i] for i in range(n)}    # route id -> ordered customer list (0-indexed)
    route_id = list(range(n))              # customer -> its route id
    load = {i: float(demand[i]) for i in range(n)}

    # Savings list, descending.
    savings = []
    for i in range(n):
        for j in range(i + 1, n):
            s = d0[i] + d0[j] - np.linalg.norm(loc[i] - loc[j])
            savings.append((s, i, j))
    savings.sort(reverse=True)

    for s, i, j in savings:
        if s <= 0:
            break
        ri, rj = route_id[i], route_id[j]
        if ri == rj:
            continue
        if load[ri] + load[rj] > CAPACITY + _EPS:
            continue
        Ri, Rj = routes[ri], routes[rj]
        # i and j must each be an endpoint; orient so i is the tail of Ri, j the head of Rj.
        if Ri[0] == i:
            Ri = Ri[::-1]
        elif Ri[-1] != i:
            continue
        if Rj[-1] == j:
            Rj = Rj[::-1]
        elif Rj[0] != j:
            continue
        merged = Ri + Rj
        routes[ri] = merged
        load[ri] += load[rj]
        for c in Rj:
            route_id[c] = ri
        del routes[rj]

    # Concatenate routes, separating them with a depot visit (0).
    tour: List[int] = []
    for r in routes.values():
        tour.extend(c + 1 for c in r)
        tour.append(0)
    return tour


@torch.no_grad()
def heuristic_mean_cost(inputs: dict, tour_fn: Callable) -> float:
    """Mean CVRP route length of `tour_fn` over a batch, using the repo cost metric.

    Args:
        inputs:  CVRP inputs dict (depot (B,2), loc (B,n,2), demand (B,n)) on CPU.
        tour_fn: depot, loc, demand (numpy) -> node-index tour list.
    Returns:
        mean route length (float).
    """
    depot, loc, demand = inputs["depot"], inputs["loc"], inputs["demand"]
    batch = loc.shape[0]
    costs = []
    for b in range(batch):
        dp, lc, dm = depot[b].cpu().numpy(), loc[b].cpu().numpy(), demand[b].cpu().numpy()
        tour = tour_fn(dp, lc, dm)
        # Reuse the repo's cost function so the metric is identical to the AM's.
        single = {"depot": depot[b:b + 1].cpu(),
                  "loc": loc[b:b + 1].cpu(),
                  "demand": demand[b:b + 1].cpu()}
        t = torch.tensor(tour, dtype=torch.long).unsqueeze(0)  # (1, T)
        costs.append(cvrp_route_length(single, t).item())
    return float(np.mean(costs))
