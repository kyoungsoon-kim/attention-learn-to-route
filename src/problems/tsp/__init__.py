"""TSP problem: embedder, decode state, decoder, data, cost (Kool et al. §3, §5)."""

import torch.nn as nn

from src.nets.config import ModelConfig
from src.problems.tsp.cost import tsp_tour_length
from src.problems.tsp.data import TSPDataset, generate_tsp_instances
from src.problems.tsp.decoder import Decoder
from src.problems.tsp.state import TSPState


def tsp_embedder(cfg: ModelConfig) -> nn.Module:
    """§3.1 — TSP node embedder: linear projection of 2-D coordinates to d_h."""
    return nn.Linear(cfg.d_x, cfg.d_h)


__all__ = [
    "tsp_embedder",
    "Decoder",
    "TSPState",
    "generate_tsp_instances",
    "TSPDataset",
    "tsp_tour_length",
]
