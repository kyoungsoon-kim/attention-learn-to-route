"""Decode-state interface driven by the generic AttentionDecoder (§3.2).

The decoder is problem-agnostic; it asks the state, at every rollout step, for the
context vector and the forbidden-node mask, tells it which node was selected, and
checks whether the rollout is finished. Each routing problem implements this
interface (e.g. `TSPState`, and later `CVRPState`).

The context width returned by `get_context` must be constant across steps and
equal to the `context_dim` passed to `AttentionDecoder` for that problem.
"""

import torch


class DecodeState:
    """Interface the generic decoder drives during an autoregressive rollout."""

    def get_context(self) -> torch.Tensor:
        """Current decoder context — shape: (batch, context_dim)."""
        raise NotImplementedError

    def get_mask(self) -> torch.Tensor:
        """Forbidden-node mask, True where a node may not be selected — (batch, n)."""
        raise NotImplementedError

    def update(self, selected: torch.Tensor) -> None:
        """Advance the state given the selected node indices — (batch,)."""
        raise NotImplementedError

    def all_done(self) -> bool:
        """True once every instance in the batch has a complete solution."""
        raise NotImplementedError
