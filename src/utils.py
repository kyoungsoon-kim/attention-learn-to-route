"""
Attention, Learn to Solve Routing Problems! — Shared utilities

Config loading lives here. The TSP cost L(π) now lives in
`src/problems/tsp/cost.py` (per-problem modules) and is re-exported below so
existing imports (`from src.utils import tsp_tour_length`) keep working.

Section references:
  §3, §4 — cost L(π) is the Euclidean tour length of a permutation π.
"""

from pathlib import Path  # noqa: F401  (kept for compatibility)
from typing import Any, Dict

import yaml

from src.problems.tsp.cost import tsp_tour_length

__all__ = ["load_config", "tsp_tour_length"]


def load_config(path: str) -> Dict[str, Any]:
    """Load the YAML config (configs/base.yaml)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
