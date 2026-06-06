"""Attention Model configuration (Kool et al., ICLR 2019)."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """AM configuration. Defaults from Kool et al. (2019) unless marked [UNSPECIFIED].

    These fields describe the shared encoder/decoder dimensions; they are the same
    across routing problems. Problem-specific shapes (e.g. the decoder context
    width) are derived per problem, not stored here.
    """
    d_h: int = 128          # §3.1 — "d_h = 128"
    d_x: int = 2            # §3.1 — "for TSP d_x = 2" (raw input feature dim)
    n_layers: int = 3       # §5  — "N = 3 layers in the encoder"
    n_heads: int = 8        # §3.1 — "M = 8 heads"
    d_ff: int = 512         # §3.1 — FF hidden dim 512
    tanh_clip: float = 10.0 # §3.2 — "C = 10"
    qkv_bias: bool = False  # [UNSPECIFIED] — matrices only in paper; no bias
