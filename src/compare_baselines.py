"""
CVRP baselines vs the Attention Model — apples-to-apples comparison.

Evaluates, on the *same* fixed held-out set, four solvers:
  - Nearest Neighbor (construction heuristic)
  - Clarke-Wright Savings (construction heuristic)
  - Attention Model, greedy decoding
  - Attention Model, sampling decoding (best of N)

All share the repo cost metric (`cvrp_route_length`) and the seeded test set, so
the numbers are directly comparable. A reference length (LKH) gives the gap.

Add `--ortools` to include the OR-Tools solver (strong, near-optimal, slower).

Run:
    python -m src.compare_baselines --config configs/cvrp_reduced.yaml --checkpoint outputs_cvrp/best.pt
    python -m src.compare_baselines --config configs/cvrp_reduced.yaml --checkpoint outputs_cvrp/best.pt --ortools
"""

import argparse

import torch

from src.evaluate import greedy_cost, sampling_cost
from src.model import AttentionModel, ModelConfig
from src.problems.cvrp import cvrp_route_length
from src.problems.cvrp.heuristics import (
    clarke_wright_tour, heuristic_mean_cost, nearest_neighbor_tour)
from src.train import _make_test_set
from src.utils import load_config


def compare(config_path: str, checkpoint: str, n_sampling: int | None = None,
            test_size: int | None = None, ortools: bool = False,
            ortools_time: float = 0.5) -> None:
    cfg = load_config(config_path)
    mcfg, tcfg, ecfg = cfg["model"], cfg["training"], cfg.get("eval", {})
    assert cfg.get("problem") == "cvrp", "this comparison is for CVRP configs"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    size = test_size or ecfg.get("test_size", 1000)
    n_samples = n_sampling or ecfg.get("n_sampling", 1280)
    ref = ecfg.get("reference_length")
    graph_size = tcfg["graph_size"]

    # Fixed seeded held-out set (CPU for the heuristics; a device copy for the AM).
    test_cpu = _make_test_set("cvrp", size, graph_size, ecfg.get("test_seed", 4321), "cpu")
    test_dev = {k: v.to(device) for k, v in test_cpu.items()}

    # Heuristics.
    nn_cost = heuristic_mean_cost(test_cpu, nearest_neighbor_tour)
    cw_cost = heuristic_mean_cost(test_cpu, clarke_wright_tour)

    # OR-Tools (optional, strong, slower — runs to the time limit per instance).
    ortools_cost = None
    if ortools:
        from src.problems.cvrp.ortools_solver import solve_ortools_tour
        print(f">>> OR-Tools solving {size} instances @ {ortools_time}s each "
              f"(~{size * ortools_time / 60:.1f} min)...")
        ortools_cost = heuristic_mean_cost(
            test_cpu, lambda dp, lc, dm: solve_ortools_tour(dp, lc, dm, ortools_time))

    # Attention Model.
    model = AttentionModel(ModelConfig(
        d_h=mcfg["d_h"], d_x=mcfg["d_x"], n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"], d_ff=mcfg["d_ff"], tanh_clip=mcfg["tanh_clip"],
        qkv_bias=mcfg["qkv_bias"],
    ), problem="cvrp").to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    am_greedy = greedy_cost(model, test_dev, cvrp_route_length).mean().item()
    am_sample = sampling_cost(model, test_dev, n_samples=n_samples,
                              cost_fn=cvrp_route_length).mean().item()

    rows = [
        ("Nearest Neighbor (heuristic)", nn_cost),
        ("Clarke-Wright Savings (heuristic)", cw_cost),
        ("Attention Model — greedy", am_greedy),
        (f"Attention Model — sampling (N={n_samples})", am_sample),
    ]
    if ortools_cost is not None:
        rows.append((f"OR-Tools (GLS, {ortools_time}s/inst)", ortools_cost))
    rows.sort(key=lambda r: r[1])  # cheapest first

    print(f"== CVRP-{graph_size} | held-out {size} instances (seed {ecfg.get('test_seed', 4321)}) ==")
    header = f"{'method':<38} {'length':>9}"
    if ref:
        header += f" {'gap':>9}"
    print(header)
    print("-" * len(header))
    for name, cost in rows:
        line = f"{name:<38} {cost:>9.4f}"
        if ref:
            line += f" {100 * (cost / ref - 1):>+8.2f}%"
        print(line)
    if ref:
        print(f"{'reference (LKH3, Kool et al.)':<38} {ref:>9.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cvrp_reduced.yaml")
    parser.add_argument("--checkpoint", default="outputs_cvrp/best.pt")
    parser.add_argument("--n-sampling", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--ortools", action="store_true", help="include OR-Tools baseline")
    parser.add_argument("--ortools-time", type=float, default=0.5, help="OR-Tools s/instance")
    args = parser.parse_args()
    compare(args.config, args.checkpoint, args.n_sampling, args.test_size,
            args.ortools, args.ortools_time)
