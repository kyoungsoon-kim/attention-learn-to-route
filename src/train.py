"""
Attention, Learn to Solve Routing Problems! — Training loop (Algorithm 1)

Paper: https://arxiv.org/abs/1803.08475
Implements: REINFORCE with greedy rollout baseline (§4, Algorithm 1) for TSP.

Run:
    python -m src.train --config configs/base.yaml
"""

import argparse
from pathlib import Path

import torch

from src.baseline import ExponentialBaseline, RolloutBaseline
from src.evaluate import greedy_cost
from src.loss import reinforce_loss
from src.model import AttentionModel, ModelConfig
from src.problems.cvrp import cvrp_route_length, generate_cvrp_instances
from src.problems.tsp import generate_tsp_instances, tsp_tour_length
from src.problems.tsp.data import TSPDataset
from src.utils import load_config


def _to_device(instances, device):
    """Move a problem's instances (TSP tensor or CVRP inputs dict) to `device`."""
    if isinstance(instances, dict):
        return {k: v.to(device) for k, v in instances.items()}
    return instances.to(device)


def _make_test_set(problem: str, size: int, graph_size: int, seed: int, device: str):
    """Build a fixed (seeded) held-out evaluation set for the greedy learning curve."""
    if problem == "cvrp":
        gen = torch.Generator().manual_seed(seed)  # CPU generator -> generate then move
        return _to_device(generate_cvrp_instances(size, graph_size, generator=gen), device)
    return TSPDataset(size, graph_size, seed=seed).data.to(device)


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    mcfg, tcfg = cfg["model"], cfg["training"]
    ecfg = cfg.get("eval", {})
    problem = cfg.get("problem", "tsp")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tcfg["seed"])  # [UNSPECIFIED] seed (§5 reports robustness)
    print(f"device: {device} | problem: {problem}")

    # Problem-specific instance generator and cost function.
    if problem == "cvrp":
        instance_fn, cost_fn = generate_cvrp_instances, cvrp_route_length
    elif problem == "tsp":
        instance_fn, cost_fn = generate_tsp_instances, tsp_tour_length
    else:
        raise ValueError(f"unknown problem: {problem}")

    # Fixed held-out test set (constant across epochs) for the greedy learning curve.
    test_size = ecfg.get("test_size", 1000)
    test_coords = _make_test_set(
        problem, test_size, tcfg["graph_size"], ecfg.get("test_seed", 4321), device)
    ckpt_dir = Path(ecfg.get("checkpoint_dir", "outputs"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build the Attention Model (§3).
    model = AttentionModel(ModelConfig(
        d_h=mcfg["d_h"], d_x=mcfg["d_x"], n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"], d_ff=mcfg["d_ff"], tanh_clip=mcfg["tanh_clip"],
        qkv_bias=mcfg["qkv_bias"],
    ), problem=problem).to(device)

    # §4 — Adam optimizer; §5 — constant learning rate η = 1e-4.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["lr"],
        betas=tuple(tcfg["betas"]), eps=tcfg["eps"],
        weight_decay=tcfg["weight_decay"],
    )

    # §5 — exponential baseline (β=0.8) warmup during the first epoch only.
    warmup_baseline = ExponentialBaseline(beta=tcfg["bl_warmup_beta"])
    # §4 — greedy rollout baseline (frozen best model) for all subsequent epochs.
    rollout_baseline = RolloutBaseline(
        model, graph_size=tcfg["graph_size"], eval_size=tcfg["bl_eval_size"],
        alpha=tcfg["bl_alpha"], device=device,
        instance_fn=instance_fn, cost_fn=cost_fn,
    )

    best_greedy = float("inf")  # track best held-out greedy tour length

    # Algorithm 1, line 3 — for epoch = 1, ..., E.
    for epoch in range(tcfg["n_epochs"]):
        model.train()
        use_warmup = epoch == 0  # §5 — exponential baseline only in the first epoch
        running = 0.0

        # Algorithm 1, line 4 — for step = 1, ..., T.
        for step in range(tcfg["steps_per_epoch"]):
            # Line 5 — s_i ← RandomInstance() (on-the-fly, §5).
            coords = instance_fn(tcfg["batch_size"], tcfg["graph_size"], device)

            # Line 6 — π_i ← SampleRollout(s_i, p_θ).
            tour, log_lik = model(coords, decode_type="sampling")
            cost = cost_fn(coords, tour)  # L(π) — (batch,)

            # Line 7 — baseline: greedy rollout cost (or warmup EMA in epoch 1).
            if use_warmup:
                baseline = warmup_baseline.eval(cost)        # §5 warmup
            else:
                baseline = rollout_baseline.eval(coords)     # §4 greedy rollout

            # Line 8 — REINFORCE gradient; Line 9 — Adam step.
            loss = reinforce_loss(cost, baseline, log_lik)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(  # [FROM_OFFICIAL_CODE] max_norm=1.0
                model.parameters(), max_norm=tcfg["grad_clip"])
            optimizer.step()

            running += cost.mean().item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} | mean cost {cost.mean().item():.4f}")

        avg_cost = running / tcfg["steps_per_epoch"]
        # Algorithm 1, lines 11–13 — replace baseline if t-test significant.
        updated = rollout_baseline.epoch_update(model)

        # §5 — held-out greedy length (the reported decoding for the main table).
        greedy_len = greedy_cost(model, test_coords, cost_fn).mean().item()
        if greedy_len < best_greedy:
            best_greedy = greedy_len
            torch.save(model.state_dict(), ckpt_dir / "best.pt")  # save best greedy model
        print(f"[epoch {epoch}] avg sampled cost {avg_cost:.4f} | "
              f"held-out greedy {greedy_len:.4f} (best {best_greedy:.4f}) | "
              f"baseline updated: {updated}")

    print(f"done. best held-out greedy {problem.upper()}-{tcfg['graph_size']} length: "
          f"{best_greedy:.4f} (checkpoint: {ckpt_dir / 'best.pt'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    train(parser.parse_args().config)
