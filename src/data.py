"""
Attention, Learn to Solve Routing Problems! — TSP data (compatibility shim)

The TSP instance generation now lives in `src/problems/tsp/data.py` (the refactor
that introduced per-problem modules). This module re-exports it so existing
imports (`from src.data import generate_tsp_instances`) keep working.
"""

from src.problems.tsp.data import TSPDataset, generate_tsp_instances

__all__ = ["generate_tsp_instances", "TSPDataset"]
