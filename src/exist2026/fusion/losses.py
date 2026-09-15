"""Losses that optimize the soft, disagreement-aware targets directly.

Under *Learning with Disagreements* the target is a distribution, so the loss
compares distributions:

* ``x.1`` and ``x.2`` are mono-label — KL divergence against the annotator
  distribution.
* ``x.3`` is multi-label, where the entries are independent probabilities that
  do not sum to one — binary cross-entropy per category.

Both heads keep an explicit ``No`` entry, which is what lets the soft metric,
rather than a post-hoc threshold, drive learning.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F

from exist2026.taxonomy import Subtask, subtask_of

LossFn = Callable[[Mapping[str, Any], Mapping[str, Any]], torch.Tensor]


def loss_for(key: str) -> LossFn:
    """The loss of one subtask, reading its target from ``batch["y_<key>"]``."""
    target_key = f"y_{key}"
    if subtask_of(key) is Subtask.CATEGORIZATION:

        def binary_cross_entropy(output, batch):
            return F.binary_cross_entropy_with_logits(output["logits"], batch[target_key])

        return binary_cross_entropy

    def kl_divergence(output, batch):
        return F.kl_div(F.log_softmax(output["logits"], dim=-1), batch[target_key], reduction="batchmean")

    return kl_divergence


def losses_for(keys) -> dict[str, LossFn]:
    return {key: loss_for(key) for key in keys}
