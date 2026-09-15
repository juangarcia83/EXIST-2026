"""Wiring for System A: everything between a config and a trainable loader.

The notebook and the CLI both build one :class:`FusionExperiment` and then only
call training functions. Keeping the assembly here means the order that matters
— fit the preprocessor *after* the split, never before — is written once and
cannot drift between the two entry points.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from exist2026.config import FusionConfig
from exist2026.evaluation.pyevall import soft_gold
from exist2026.fusion import physio as physio_module
from exist2026.fusion.data import ExistDataset, make_loader
from exist2026.fusion.splits import Split, build_split, yes_subset
from exist2026.fusion.text import TextView, load_vlm_enrichment
from exist2026.io_utils import read_json
from exist2026.labels.soft import soft_labels as build_soft_labels
from exist2026.taxonomy import Subtask, subtask_of


@dataclass
class FusionExperiment:
    """A loaded, split and preprocessed dataset ready for training."""

    config: FusionConfig
    records: dict[str, Mapping[str, Any]]
    soft: dict[str, dict[str, np.ndarray]]
    schema: physio_module.FeatureSchema
    preprocessor: physio_module.PhysioPreprocessor
    raw_physio: dict[str, np.ndarray]
    physio_masks: dict[str, np.ndarray]
    split: Split
    text_view: TextView
    tokenizer: Any
    physio_in: int
    dropped: int = 0
    _loaders: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- construction ------------------------------------------------------- #

    @classmethod
    def build(cls, config: FusionConfig, *, verbose: bool = True) -> FusionExperiment:
        from transformers import AutoTokenizer

        records = {str(k): v for k, v in read_json(config.data.train_json()).items()}
        enrichment = load_vlm_enrichment(config.data.vlm_json())
        if verbose:
            print(f"instances: {len(records):,} | VLM enrichment: {len(enrichment):,}")
            if not enrichment:
                print("[warn] no VLM enrichment found; the text view falls back to OCR only")

        soft: dict[str, dict[str, np.ndarray]] = {}
        dropped = 0
        for instance_id, record in records.items():
            labels = build_soft_labels(record, config.modality)
            if labels is None:
                dropped += 1
            else:
                soft[instance_id] = labels
        if verbose:
            print(f"usable soft labels: {len(soft):,} (dropped {dropped:,})")

        schema = physio_module.FeatureSchema.discover(records.values())
        instance_ids = list(soft)
        raw_physio, masks = physio_module.extract(
            records, instance_ids, schema, config.physio_kind, config.max_subjects
        )
        if verbose:
            print(
                f"physiology: {schema.dim} features "
                f"(ET={len(schema.eye_tracking)} HR={len(schema.heart_rate)} EEG={len(schema.eeg)})"
            )

        identification_key = config.subtask_keys[0]
        split = build_split(
            records,
            soft,
            identification_key,
            config.modality,
            val_fraction=config.val_fraction,
            seed=config.seed,
        )

        # Fitted on TRAIN only — this ordering is the whole point.
        preprocessor = physio_module.PhysioPreprocessor(config.physio_kind, schema).fit(
            [raw_physio[i] for i in split.train],
            [masks[i] for i in split.train] if config.physio_kind == "matrix" else None,
        )

        return cls(
            config=config,
            records=records,
            soft=soft,
            schema=schema,
            preprocessor=preprocessor,
            raw_physio=raw_physio,
            physio_masks=masks,
            split=split,
            text_view=TextView(records, enrichment, config.modality),
            tokenizer=AutoTokenizer.from_pretrained(config.text_model),
            physio_in=physio_module.input_dim(
                config.physio_kind, schema, raw_physio[instance_ids[0]] if instance_ids else None
            ),
            dropped=dropped,
        )

    # -- data --------------------------------------------------------------- #

    def _transform(self, preprocessor=None):
        preprocessor = preprocessor or self.preprocessor

        def transform(instance_id: str):
            return preprocessor.transform(self.raw_physio[instance_id]), self.physio_masks.get(instance_id)

        return transform

    def dataset(self, instance_ids, *, with_labels: bool = True, preprocessor=None) -> ExistDataset:
        return ExistDataset(
            instance_ids,
            text_view=self.text_view,
            physio_transform=self._transform(preprocessor),
            tokenizer=self.tokenizer,
            max_len=self.config.max_len,
            soft_labels=self.soft if with_labels else None,
            subtask_keys=self.config.subtask_keys,
        )

    def training_ids(self, key: str) -> list[str]:
        """x.1 trains on everything; x.2 / x.3 only on the sexist instances."""
        if subtask_of(key) is Subtask.IDENTIFICATION:
            return list(self.split.train)
        return yes_subset(self.split.train, self.soft, self.config.subtask_keys[0], self.config.yes_threshold)

    def train_loader(self, key: str, batch_size: int, device: str = "cpu"):
        return make_loader(self.dataset(self.training_ids(key)), batch_size, shuffle=True, device=device)

    def val_loader(self, batch_size: int = 16, device: str = "cpu"):
        """One shared validation loader: every subtask is scored on the full split."""
        cached = self._loaders.get("val")
        if cached is None:
            cached = make_loader(self.dataset(self.split.val), batch_size, shuffle=False, device=device)
            self._loaders["val"] = cached
        return cached

    def gold(self, key: str):
        return soft_gold(self.soft, self.split.val, key)

    # -- refit on the full corpus ------------------------------------------- #

    def full_preprocessor(self) -> physio_module.PhysioPreprocessor:
        """Preprocessor refitted on train+validation, for the final model.

        Safe here and only here: validation has already done its job, and the
        test split still contributes nothing to the statistics.
        """
        ids = list(self.split.train) + list(self.split.val)
        return physio_module.PhysioPreprocessor(self.config.physio_kind, self.schema).fit(
            [self.raw_physio[i] for i in ids],
            [self.physio_masks[i] for i in ids] if self.config.physio_kind == "matrix" else None,
        )

    def full_loader(self, key: str, batch_size: int, preprocessor, device: str = "cpu"):
        ids = list(self.split.train) + list(self.split.val)
        if subtask_of(key) is not Subtask.IDENTIFICATION:
            ids = yes_subset(ids, self.soft, self.config.subtask_keys[0], self.config.yes_threshold)
        return make_loader(
            self.dataset(ids, preprocessor=preprocessor), batch_size, shuffle=True, device=device
        )

    def test_loader(self, preprocessor, batch_size: int = 16, device: str = "cpu"):
        """Loader over the official test split, or ``None`` when it is absent."""
        test_json = self.config.data.test_json(required=False)
        if test_json is None:
            return None, []
        test_records = {str(k): v for k, v in read_json(test_json).items()}
        self.records.update(test_records)
        ids = list(test_records)
        values, masks = physio_module.extract(
            self.records, ids, self.schema, self.config.physio_kind, self.config.max_subjects
        )
        self.raw_physio.update(values)
        self.physio_masks.update(masks)
        dataset = self.dataset(ids, with_labels=False, preprocessor=preprocessor)
        return make_loader(dataset, batch_size, shuffle=False, device=device), ids
