"""Dataset and collation for the trained system.

One :class:`ExistDataset` serves both modalities: it tokenizes the enriched
text view and attaches the already-preprocessed physiology. The physiological
mask travels with the batch only for the ``matrix`` representation, where the
model needs to know which subject rows are real.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ExistDataset(Dataset):
    """Tokenized text + physiology + soft labels of one instance.

    ``physio_transform`` returns ``(values, mask)`` and is the *fitted*
    preprocessor, so normalization statistics never depend on the batch.
    """

    def __init__(
        self,
        instance_ids: Sequence[str],
        *,
        text_view: Callable[[str], str],
        physio_transform: Callable[[str], tuple[np.ndarray, np.ndarray | None]],
        tokenizer,
        max_len: int,
        soft_labels: Mapping[str, Mapping[str, np.ndarray]] | None = None,
        subtask_keys: Sequence[str] = (),
    ) -> None:
        self.instance_ids = list(instance_ids)
        self.text_view = text_view
        self.physio_transform = physio_transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.soft_labels = soft_labels
        self.subtask_keys = tuple(subtask_keys)

    def __len__(self) -> int:
        return len(self.instance_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        instance_id = self.instance_ids[index]
        encoded = self.tokenizer(
            self.text_view(instance_id),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        values, mask = self.physio_transform(instance_id)
        item: dict[str, Any] = {
            "id_EXIST": instance_id,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "physio": torch.from_numpy(np.ascontiguousarray(values)),
        }
        if mask is not None:
            item["physio_mask"] = torch.from_numpy(np.ascontiguousarray(mask))
        if self.soft_labels is not None:
            for key in self.subtask_keys:
                item[f"y_{key}"] = torch.from_numpy(self.soft_labels[instance_id][key])
        return item


def collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack every tensor field; ids stay a plain list."""
    out: dict[str, Any] = {"id_EXIST": [item["id_EXIST"] for item in batch]}
    for key in batch[0]:
        if key == "id_EXIST":
            continue
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def make_loader(dataset: ExistDataset, batch_size: int, *, shuffle: bool, device: str = "cpu"):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate,
        pin_memory=(device == "cuda"),
    )


def to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()
    }
