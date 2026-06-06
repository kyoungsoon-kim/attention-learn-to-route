"""
Attention, Learn to Solve Routing Problems! — checkpoint evaluation (§5, Table 1)

Loads a trained checkpoint and reports the two decoding strategies from the paper
on a fixed (seeded) held-out set: greedy (argmax per step) and sampling (draw
`n_sampling` solutions, keep the best). If the config provides a reference length
(optimal / LKH), the optimality gap is reported too.

Run:
    python -m src.report --config configs/reduced.yaml      --checkpoint outputs/best.pt
    python -m src.report --config configs/cvrp_reduced.yaml --checkpoint outputs_cvrp/best.pt
"""

import argparse

import torch

from src.evaluate import greedy_cost, sampling_cost
from src.model import AttentionModel, ModelConfig
from src.problems.cvrp import cvrp_route_length
from src.problems.tsp import tsp_tour_length
from src.train import _make_test_set  # fixed seeded held-out set (TSP tensor / CVRP dict)
from src.utils import load_config


def report(config_path: str, checkpoint: str, n_sampling: int | None = None,
           test_size: int | None = None) -> None:
    cfg = load_config(config_path)
    mcfg, tcfg, ecfg = cfg["model"], cfg["training"], cfg.get("eval", {})
    problem = cfg.get("problem", "tsp")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cost_fn = cvrp_route_length if problem == "cvrp" else tsp_tour_length

    model = AttentionModel(ModelConfig(
        d_h=mcfg["d_h"], d_x=mcfg["d_x"], n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"], d_ff=mcfg["d_ff"], tanh_clip=mcfg["tanh_clip"],
        qkv_bias=mcfg["qkv_bias"],
    ), problem=problem).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    size = test_size or ecfg.get("test_size", 1000)
    n_samples = n_sampling or ecfg.get("n_sampling", 1280)
    test_set = _make_test_set(problem, size, tcfg["graph_size"], ecfg.get("test_seed", 4321), device)

    g = greedy_cost(model, test_set, cost_fn).mean().item()
    s = sampling_cost(model, test_set, n_samples=n_samples, cost_fn=cost_fn).mean().item()

    ref = ecfg.get("reference_length")
    name = f"{problem.upper()}-{tcfg['graph_size']}"
    print(f"== {name} | held-out {size} instances | checkpoint {checkpoint} ==")
    print(f"  greedy            : {g:.4f}" + (f"   (gap {100*(g/ref-1):+.2f}%)" if ref else ""))
    print(f"  sampling (N={n_samples}): {s:.4f}" + (f"   (gap {100*(s/ref-1):+.2f}%)" if ref else ""))
    if ref:
        print(f"  reference         : {ref:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-sampling", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    args = parser.parse_args()
    report(args.config, args.checkpoint, args.n_sampling, args.test_size)
