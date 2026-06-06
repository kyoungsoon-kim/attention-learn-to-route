"""CVRP problem: embedder, decode state, decoder, data, cost (Kool et al. App. — VRP)."""

import torch.nn as nn

from src.nets.config import ModelConfig
from src.problems.cvrp.cost import cvrp_route_length
from src.problems.cvrp.data import capacity_for, generate_cvrp_instances
from src.problems.cvrp.decoder import Decoder
from src.problems.cvrp.embedder import CVRPEmbedder
from src.problems.cvrp.state import CVRPState


def cvrp_embedder(cfg: ModelConfig) -> nn.Module:
    """App. (VRP) — depot/customer split node embedder."""
    return CVRPEmbedder(cfg)


__all__ = [
    "cvrp_embedder",
    "Decoder",
    "CVRPState",
    "CVRPEmbedder",
    "generate_cvrp_instances",
    "capacity_for",
    "cvrp_route_length",
]
