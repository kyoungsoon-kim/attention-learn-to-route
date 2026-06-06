"""
Attention, Learn to Solve Routing Problems! — Evaluation metrics

Paper: https://arxiv.org/abs/1803.08475
Implements: greedy and sampling decoding cost (§5 main results metric).

Section references:
  §5 — greedy decoding (best action each step); sampling 1280 solutions, report best.
  Table 1 — reported metric is mean tour length (and optimality gap vs a reference).
"""

from typing import Callable

import torch

from src.model import AttentionModel
from src.utils import tsp_tour_length


def _batch_size(coords) -> int:
    """Batch size for either a TSP tensor (batch, n, 2) or a CVRP inputs dict."""
    if isinstance(coords, dict):
        return coords["loc"].shape[0]
    return coords.shape[0]


def _device(coords):
    if isinstance(coords, dict):
        return coords["loc"].device
    return coords.device


@torch.no_grad()
def greedy_cost(model: AttentionModel, coords,
                cost_fn: Callable = tsp_tour_length) -> torch.Tensor:
    """§5 — greedy decoding: argmax action at each step.

    Args:
        coords:  instances (TSP tensor (batch, n, 2) or CVRP inputs dict).
        cost_fn: maps (coords, tour) -> per-instance cost; problem-specific.
    Returns:
        greedy cost — shape: (batch,)
    """
    model.eval()
    tour, _ = model(coords, decode_type="greedy")
    return cost_fn(coords, tour)  # (batch,)


@torch.no_grad()
def sampling_cost(model: AttentionModel, coords, n_samples: int = 1280,
                  cost_fn: Callable = tsp_tour_length) -> torch.Tensor:
    """§5 — sample `n_samples` solutions per instance and report the best (min cost).

    Args:
        coords:    instances (TSP tensor or CVRP inputs dict).
        n_samples: number of sampled solutions per instance (§5 uses 1280).
        cost_fn:   problem-specific cost function.
    Returns:
        best (minimum) sampled cost per instance — shape: (batch,)
    """
    model.eval()
    best = torch.full((_batch_size(coords),), float("inf"), device=_device(coords))
    for _ in range(n_samples):
        tour, _ = model(coords, decode_type="sampling")
        cost = cost_fn(coords, tour)           # (batch,)
        best = torch.minimum(best, cost)       # §5 — keep the best across samples
    return best


def optimality_gap(cost: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Table 1 — optimality gap (%) of `cost` w.r.t. a reference (e.g. optimal).

    Args:
        cost:      model cost — shape: (batch,)
        reference: reference/optimal cost — shape: (batch,)
    Returns:
        per-instance gap in percent — shape: (batch,)
    """
    return (cost / reference - 1.0) * 100.0
