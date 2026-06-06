"""TSP instance generation (§5, Appendix B.2).

Section references:
  §5  — "training data generated on the fly", 2500 batches × 512 instances / epoch.
  App. B.2 — TSP instances: node coordinates sampled uniformly in the unit square.
"""

import torch
from torch.utils.data import Dataset


def generate_tsp_instances(batch_size: int, graph_size: int,
                           device: str = "cpu") -> torch.Tensor:
    """App. B.2 — sample TSP instances: n nodes uniform in [0, 1]^2.

    Args:
        batch_size: number of instances B.
        graph_size: number of nodes n per instance.
    Returns:
        coordinates — shape: (batch_size, graph_size, 2)
    """
    return torch.rand(batch_size, graph_size, 2, device=device)  # U([0,1]^2)


class TSPDataset(Dataset):
    """A fixed set of random TSP instances (e.g. for the test set or t-test eval).

    Training uses `generate_tsp_instances` directly (§5 — data generated on the
    fly), so this dataset is only for reproducible held-out evaluation.
    """

    def __init__(self, size: int, graph_size: int, seed: int | None = None):
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        # App. B.2 — uniform nodes in the unit square.
        self.data = torch.rand(size, graph_size, 2, generator=gen)  # (size, n, 2)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]  # (n, 2)
