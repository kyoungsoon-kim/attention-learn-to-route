"""
Attention, Learn to Solve Routing Problems! — REINFORCE loss

Paper: https://arxiv.org/abs/1803.08475
Implements: the REINFORCE gradient estimator with a baseline b(s) (§4, Algorithm 1).

Section references:
  §4 — ∇L(θ|s) = E[(L(π) − b(s)) ∇ log p_θ(π|s)].
  Algorithm 1, line 8 — ∇L ← Σ_i (L(π_i) − L(π_i^BL)) ∇ log p_θ(π_i).
"""

import torch


def reinforce_loss(
    cost: torch.Tensor,
    baseline: torch.Tensor,
    log_likelihood: torch.Tensor,
) -> torch.Tensor:
    """§4 / Algorithm 1 line 8 — REINFORCE loss with baseline.

    The gradient estimator is (L(π) − b(s)) ∇ log p_θ(π). We return the scalar
    surrogate loss whose autograd gradient equals that estimator (averaged over
    the batch): mean[(L(π) − b(s)) · log p_θ(π)]. The advantage is detached so it
    acts as a constant multiplier and does not propagate gradients.

    Args:
        cost:           sampled solution cost L(π) — shape: (batch,)
        baseline:       baseline b(s) (greedy rollout cost) — shape: (batch,)
        log_likelihood: log p_θ(π|s) for the sampled tour — shape: (batch,)
    Returns:
        scalar loss to call .backward() on.
    """
    advantage = (cost - baseline).detach()          # (batch,) — L(π) − b(s), no grad
    return (advantage * log_likelihood).mean()      # scalar surrogate loss
