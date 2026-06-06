"""OR-Tools CVRP solver — a strong (near-optimal) baseline.

Google OR-Tools' constraint-programming routing solver with a guided-local-search
metaheuristic. Used as the strongest non-learned reference in the comparison; it
is far better than the construction heuristics but costs real solve time per
instance (it runs until the time limit).

The returned tour uses the repo convention (0 = depot, 1..n = customers, depot
repeated between routes), so `cvrp_route_length` scores it on the same metric as
everything else. Demands are normalized (capacity = 1).

Requires `ortools` (`pip install ortools`).
"""

from typing import List

import numpy as np

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

_DIST_SCALE = 100_000  # ints for the solver; reported cost uses real coords via cvrp_route_length
_DEM_SCALE = 10_000


def solve_ortools_tour(depot: np.ndarray, loc: np.ndarray, demand: np.ndarray,
                       time_limit_s: float = 0.5) -> List[int]:
    """Solve one CVRP instance with OR-Tools. Returns a node-index tour (0 = depot)."""
    n = len(loc)
    coords = np.vstack([depot[None, :], loc])        # (n+1, 2); 0 = depot
    demands = np.concatenate([[0.0], demand])        # depot demand 0
    num_vehicles = n                                 # enough vehicles to never block

    manager = pywrapcp.RoutingIndexManager(n + 1, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    dist = (np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            * _DIST_SCALE).astype(np.int64)

    def dist_cb(i, j):
        return int(dist[manager.IndexToNode(i)][manager.IndexToNode(j)])

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_cb))

    dem_int = (demands * _DEM_SCALE).astype(np.int64)

    def dem_cb(i):
        return int(dem_int[manager.IndexToNode(i)])

    dem_idx = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(
        dem_idx, 0, [_DEM_SCALE] * num_vehicles, True, "Capacity")  # cap = 1.0 * scale

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools found no solution")

    # Extract: walk each vehicle; emit visited customers then a depot separator.
    tour: List[int] = []
    for v in range(num_vehicles):
        idx = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            continue  # unused vehicle
        idx = solution.Value(routing.NextVar(idx))
        while not routing.IsEnd(idx):
            tour.append(manager.IndexToNode(idx))  # 1..n (depot 0 never appears mid-route)
            idx = solution.Value(routing.NextVar(idx))
        tour.append(0)  # return to depot after this vehicle's route
    return tour
