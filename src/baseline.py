"""
Attention, Learn to Solve Routing Problems! — Rollout baseline

Paper: https://arxiv.org/abs/1803.08475
Implements: the greedy rollout baseline b(s) and its per-epoch t-test update,
plus the exponential warmup baseline used in the first epoch.

Section references:
  §4 — b(s) is the cost of a deterministic greedy rollout of the best model so far.
  §4 — replace baseline params only if improvement is significant (paired t-test, α=5%)
        on 10000 separate evaluation instances.
  §5 — exponential baseline (β=0.8) during the first epoch to stabilize learning.
"""

import copy
from typing import Callable, Optional

import torch
from scipy.stats import ttest_rel

from src.data import generate_tsp_instances
from src.model import AttentionModel
from src.utils import tsp_tour_length


@torch.no_grad()
def rollout_cost(model: AttentionModel, coords, cost_fn: Callable = tsp_tour_length) -> torch.Tensor:
    """§4 — deterministic greedy rollout cost of `model` on the given instances.

    Args:
        coords:  instances (TSP: (batch, n, 2); CVRP: an inputs dict).
        cost_fn: maps (coords, tour) -> per-instance cost; problem-specific.
    Returns:
        greedy tour cost L(π^BL) — shape: (batch,)
    """
    model.eval()
    tour, _ = model(coords, decode_type="greedy")  # §4 — greedy (deterministic) rollout
    return cost_fn(coords, tour)                    # (batch,)


class ExponentialBaseline:
    """§5 — exponential moving-average baseline b(s) = M, M ← βM + (1−β)L(π).

    Used only as the first-epoch warmup (§5). A scalar EMA shared across the batch.
    """

    def __init__(self, beta: float = 0.8):
        self.beta = beta
        self.M: Optional[torch.Tensor] = None

    def eval(self, cost: torch.Tensor) -> torch.Tensor:
        """Return the current baseline and update the EMA with this batch's cost."""
        if self.M is None:
            self.M = cost.mean().detach()  # §4 — M = L(π) in the first iteration
        else:
            self.M = self.beta * self.M + (1 - self.beta) * cost.mean().detach()
        return self.M.expand_as(cost)      # broadcast scalar baseline to the batch


class RolloutBaseline:
    """§4 — greedy rollout baseline from a frozen copy of the best model so far.

    The baseline policy θ^BL is a frozen snapshot; it is replaced by the current
    policy θ only when a paired t-test (α) over `bl_eval_size` instances says the
    current policy is significantly better (§4, Algorithm 1 lines 11–13).
    """

    def __init__(self, model: AttentionModel, graph_size: int,
                 eval_size: int = 10000, alpha: float = 0.05,
                 device: str = "cpu",
                 instance_fn: Callable = generate_tsp_instances,
                 cost_fn: Callable = tsp_tour_length):
        self.graph_size = graph_size
        self.eval_size = eval_size
        self.alpha = alpha
        self.device = device
        self.instance_fn = instance_fn  # (size, graph_size, device) -> instances
        self.cost_fn = cost_fn          # (instances, tour) -> per-instance cost
        self._update_model(model)

    def _update_model(self, model: AttentionModel) -> None:
        """Freeze a deep copy of `model` as the baseline policy θ^BL."""
        self.model = copy.deepcopy(model).to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        # §4 — fresh evaluation instances when the baseline is (re)set, to avoid overfitting.
        self.eval_instances = self.instance_fn(
            self.eval_size, self.graph_size, device=self.device)
        self.eval_baseline = rollout_cost(
            self.model, self.eval_instances, self.cost_fn)  # (eval_size,)

    def eval(self, coords) -> torch.Tensor:
        """§4 — baseline b(s): greedy rollout cost of the frozen baseline policy.

        Args:
            coords: training instances (TSP: (batch, n, 2); CVRP: inputs dict).
        Returns:
            baseline cost — shape: (batch,)
        """
        return rollout_cost(self.model, coords, self.cost_fn)

    def epoch_update(self, candidate: AttentionModel) -> bool:
        """§4 / Algorithm 1 lines 11–13 — replace baseline if t-test is significant.

        Compares the candidate (current) policy against the frozen baseline on the
        same fixed evaluation set with greedy decoding. Replaces the baseline only
        if the candidate is better AND a one-sided paired t-test gives p < α.

        Returns:
            True if the baseline was updated.
        """
        candidate_cost = rollout_cost(candidate, self.eval_instances, self.cost_fn)  # (eval_size,)
        cand_mean = candidate_cost.mean().item()
        base_mean = self.eval_baseline.mean().item()

        if cand_mean >= base_mean:
            return False  # candidate not better on average → no update

        # One-sided paired t-test: H1 = candidate cost < baseline cost.
        _, p_two_sided = ttest_rel(
            candidate_cost.cpu().numpy(), self.eval_baseline.cpu().numpy())
        p_one_sided = p_two_sided / 2.0
        if p_one_sided < self.alpha:
            self._update_model(candidate)  # §4 — θ^BL ← θ
            return True
        return False
