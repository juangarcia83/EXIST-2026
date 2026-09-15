"""ELiRF-UPV at EXIST 2026 — sexism detection in memes and TikTok videos.

Two system families share this package:

* **System A** (:mod:`exist2026.fusion`) — a trained cross-attention model that
  fuses an enriched textual view with physiological signals and optimizes the
  soft, disagreement-aware targets directly.
* **System B** (:mod:`exist2026.fewshot`) — a training-free hierarchical
  cascade that prompts an open-weight vision-language model.

Both run over memes (Task 2) and videos (Task 3) through the same code, with
the modality as a parameter rather than a fork.

Importing this package pulls in no heavy dependency: ``torch``, ``transformers``
and ``PyEvALL`` are imported by the modules that need them, when they need them.
"""

from exist2026.taxonomy import (
    SEXISM_CATEGORIES,
    Modality,
    Subtask,
    subtask_key,
    subtask_keys,
)

__all__ = ["SEXISM_CATEGORIES", "Modality", "Subtask", "subtask_key", "subtask_keys"]
__version__ = "0.1.0"
