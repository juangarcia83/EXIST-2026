"""Run configuration: paths, backends and hyperparameters, loaded from YAML.

Every notebook and script starts by building one of these objects. Nothing
below this module reads ``os.environ`` or hard-codes a filesystem layout, which
is what makes the same code run unchanged on a laptop, on Colab and in CI.

The dataset itself is distributed by the EXIST organizers and is *not* part of
this repository, so paths are declared as candidate lists and resolved lazily:
a config can be built (and unit-tested) on a machine that has no data at all.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from exist2026.io_utils import first_existing
from exist2026.labels.hard import HardLabelThresholds
from exist2026.taxonomy import Modality, Subtask, subtask_key

#: Environment variable that overrides ``project_root`` from the YAML file.
ROOT_ENV_VAR = "EXIST2026_ROOT"

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
PROMPT_DIR = REPO_ROOT / "prompts"


class ConfigError(ValueError):
    """Raised when a configuration file is inconsistent or incomplete."""


@dataclass(frozen=True)
class DataPaths:
    """Where the (externally distributed) corpus lives.

    ``*_candidates`` are ordered; the first existing entry wins. ``resolve_*``
    raises only when the data is actually needed, so a missing test split never
    blocks a training-only run.
    """

    train_json_candidates: tuple[Path, ...]
    train_media_candidates: tuple[Path, ...]
    test_json_candidates: tuple[Path, ...] = ()
    test_media_candidates: tuple[Path, ...] = ()
    vlm_json_candidates: tuple[Path, ...] = ()

    def _resolve(self, candidates: tuple[Path, ...], what: str, required: bool) -> Path | None:
        found = first_existing(candidates)
        if found is None and required:
            listed = "\n  ".join(str(c) for c in candidates)
            raise FileNotFoundError(
                f"could not locate the {what}. Checked:\n  {listed}\n"
                f"Set {ROOT_ENV_VAR} or edit the config so one of these paths exists."
            )
        return found

    def train_json(self, required: bool = True) -> Path | None:
        return self._resolve(self.train_json_candidates, "training JSON", required)

    def train_media_dir(self, required: bool = True) -> Path | None:
        return self._resolve(self.train_media_candidates, "training media directory", required)

    def test_json(self, required: bool = True) -> Path | None:
        return self._resolve(self.test_json_candidates, "test JSON", required)

    def test_media_dir(self, required: bool = True) -> Path | None:
        return self._resolve(self.test_media_candidates, "test media directory", required)

    def vlm_json(self, required: bool = False) -> Path | None:
        return self._resolve(self.vlm_json_candidates, "VLM enrichment JSON", required)


@dataclass(frozen=True)
class RunDirs:
    """Output tree of a single run, namespaced so runs never overwrite one another."""

    root: Path

    @property
    def pools(self) -> Path:
        return self.root / "fewshot_pools"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def pyevall_work(self) -> Path:
        return self.root / "pyevall_work"

    @property
    def submission(self) -> Path:
        return self.root / "submission"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    def create(self) -> RunDirs:
        for path in (
            self.pools,
            self.predictions,
            self.checkpoints,
            self.metrics,
            self.logs,
            self.pyevall_work,
            self.submission,
            self.figures,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class PoolSpec:
    """How to build one anti-leakage few-shot pool.

    The pool is always drawn from the *training* partition, which is what makes
    it leakage-free: no test instance can ever reach a prompt.
    """

    per_class: int
    #: Balance each class half ES / half EN (used for videos, which are bilingual
    #: enough per class to afford it).
    bilingual: bool = False
    #: Mix high-agreement and medium-agreement exemplars instead of taking the
    #: top-k, so the prompt also shows the model what a borderline case looks like.
    diversify_consensus: bool = False
    high_consensus_fraction: float = 0.7
    medium_consensus_range: tuple[float, float] = (0.5, 0.7)
    #: Extra genuinely multi-label exemplars appended to an x.3 pool.
    cooccurrence_extra: int = 0


@dataclass(frozen=True)
class GenerationSpec:
    """Decoding budget of the few-shot backend."""

    max_new_tokens: Mapping[str, int]
    #: Greedy decoding keeps runs reproducible; sampling would not be.
    do_sample: bool = False


@dataclass(frozen=True)
class BackendSpec:
    """A vision-language backend of System B."""

    name: str
    model_id: str
    model_classes: tuple[str, ...]
    needs_qwen_vl_utils: bool
    #: Load in 4-bit when the GPU has less VRAM than this (bytes).
    four_bit_below_vram: float = 48e9


@dataclass(frozen=True)
class FewShotConfig:
    """System B — training-free hierarchical cascade over a VLM."""

    modality: Modality
    stage: str
    backend: BackendSpec
    data: DataPaths
    dirs: RunDirs
    generation: GenerationSpec
    pools: Mapping[str, PoolSpec]
    thresholds: HardLabelThresholds
    team_name: str
    run_id: int
    seed: int = 42
    #: ``None`` evaluates the whole split; an int truncates it (sanity runs).
    max_eval_instances: int | None = None
    checkpoint_every: int = 100
    #: Append a one-sentence rationale to each few-shot answer.
    use_rationale: bool = False
    #: >1 rebuilds the pools with different tie-breaks to measure pool variance.
    pool_seeds: int = 1
    #: Frame sampling for video backends that expose it (Qwen-VL).
    video_fps: float = 0.5
    video_max_pixels: int = 480 * 640
    #: Label to emit when the parser finds nothing usable. ``None`` for x.3
    #: means "the most frequent training category", computed at load time.
    categorization_fallback: str | None = None

    @property
    def subtask_keys(self) -> tuple[str, str, str]:
        return tuple(subtask_key(self.modality, s) for s in Subtask)  # type: ignore[return-value]

    @property
    def run_mode(self) -> str:
        """``"train_eval"`` for sanity runs, ``"test"`` for submission runs."""
        return "test" if self.stage == "submission" else "train_eval"

    @property
    def checkpoint_suffix(self) -> str:
        if self.run_mode == "test":
            return "test"
        return f"sanity{self.max_eval_instances or 'all'}"


@dataclass(frozen=True)
class TrainingSpec:
    """Default optimizer / architecture hyperparameters of System A."""

    epochs: int = 5
    batch_size: int = 16
    lr_head: float = 5e-4
    lr_backbone: float = 1e-5
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    p_drop: float = 0.2
    sensor_hidden: int = 256
    sensor_out: int = 128
    fusion_hidden: int = 512
    num_heads: int = 8
    freeze_backbone: bool = True

    def merged(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """This spec as a dict, with ``overrides`` applied on top."""
        base = dict(self.__dict__)
        base.update(overrides or {})
        return base


@dataclass(frozen=True)
class HpoSpec:
    """Optuna search budget."""

    n_trials: int = 20
    epochs: int = 4
    startup_trials: int = 5
    warmup_steps: int = 1


@dataclass(frozen=True)
class FusionConfig:
    """System A — trained cross-attention fusion of text and physiology."""

    modality: Modality
    data: DataPaths
    dirs: RunDirs
    text_model: str
    max_len: int
    #: ``"flat"`` aggregates subjects into one vector (memes); ``"matrix"``
    #: keeps one row per subject and pools them with attention (videos).
    physio_kind: str
    training: TrainingSpec = field(default_factory=TrainingSpec)
    hpo: HpoSpec = field(default_factory=HpoSpec)
    team_name: str = "ELiRF_UPV"
    run_id: int = 2
    seed: int = 42
    val_fraction: float = 0.15
    max_subjects: int = 4
    final_epochs: int = 5
    #: Instances whose P(YES) reaches this threshold train the x.2 / x.3 heads.
    yes_threshold: float = 0.5

    @property
    def subtask_keys(self) -> tuple[str, str, str]:
        return tuple(subtask_key(self.modality, s) for s in Subtask)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #


def resolve_project_root(raw: str | None) -> Path:
    """Data root: ``$EXIST2026_ROOT`` if set, else the config value."""
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).expanduser()
    if not raw:
        raise ConfigError(
            f"no project_root in the config and {ROOT_ENV_VAR} is unset; "
            "point one of them at the directory holding the EXIST 2026 corpus."
        )
    return Path(os.path.expandvars(raw)).expanduser()


def _paths(root: Path, entries: Any) -> tuple[Path, ...]:
    if not entries:
        return ()
    if isinstance(entries, str):
        entries = [entries]
    return tuple((root / Path(os.path.expandvars(e))).resolve() for e in entries)


def _data_paths(root: Path, block: Mapping[str, Any]) -> DataPaths:
    return DataPaths(
        train_json_candidates=_paths(root, block.get("train_json")),
        train_media_candidates=_paths(root, block.get("train_media_dir")),
        test_json_candidates=_paths(root, block.get("test_json")),
        test_media_candidates=_paths(root, block.get("test_media_dir")),
        vlm_json_candidates=_paths(root, block.get("vlm_json")),
    )


def _pool_spec(block: Mapping[str, Any]) -> PoolSpec:
    medium = block.get("medium_consensus_range", (0.5, 0.7))
    return PoolSpec(
        per_class=int(block["per_class"]),
        bilingual=bool(block.get("bilingual", False)),
        diversify_consensus=bool(block.get("diversify_consensus", False)),
        high_consensus_fraction=float(block.get("high_consensus_fraction", 0.7)),
        medium_consensus_range=(float(medium[0]), float(medium[1])),
        cooccurrence_extra=int(block.get("cooccurrence_extra", 0)),
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML config, resolving it against ``configs/`` when relative."""
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        path = CONFIG_DIR / path
    if not path.exists() and path.suffix == "":
        path = path.with_suffix(".yaml")
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_fewshot_config(path: str | Path, **overrides: Any) -> FewShotConfig:
    """Build a :class:`FewShotConfig` from ``configs/fewshot_*.yaml``.

    ``overrides`` replace top-level fields after loading, which is how the CLI
    turns ``--stage sanity`` into a config change without a second YAML file.
    """
    raw = load_yaml(path)
    root = resolve_project_root(raw.get("project_root"))
    modality = Modality(raw["modality"])

    backend_name = overrides.pop("backend", None) or raw["backend"]
    backends = raw.get("backends", {})
    if backend_name not in backends:
        raise ConfigError(f"unknown backend {backend_name!r}; the config defines {sorted(backends)}")
    block = backends[backend_name]
    backend = BackendSpec(
        name=backend_name,
        model_id=block["model_id"],
        model_classes=tuple(block["model_classes"]),
        needs_qwen_vl_utils=bool(block.get("needs_qwen_vl_utils", False)),
        four_bit_below_vram=float(block.get("four_bit_below_vram", 48e9)),
    )

    stage = overrides.pop("stage", None) or raw.get("stage", "sanity")
    if stage not in ("sanity", "submission"):
        raise ConfigError(f"stage must be 'sanity' or 'submission', got {stage!r}")

    keys = subtask_keys_for(modality)
    generation = GenerationSpec(
        max_new_tokens={
            k: int(v) for k, v in zip(keys, _per_subtask(raw["max_new_tokens"], keys), strict=True)
        }
    )
    pools = {k: _pool_spec(p) for k, p in zip(keys, _per_subtask(raw["pools"], keys), strict=True)}

    max_eval = raw.get("max_eval_instances")
    if stage == "submission":
        max_eval = None
    elif max_eval is not None:
        max_eval = int(max_eval)

    dirs = RunDirs(root / raw.get("run_dir", f"runs/fewshot_{modality.value}_{backend_name}"))

    config = FewShotConfig(
        modality=modality,
        stage=stage,
        backend=backend,
        data=_data_paths(root, raw["data"]),
        dirs=dirs,
        generation=generation,
        pools=pools,
        thresholds=HardLabelThresholds.official(modality),
        team_name=raw.get("team_name", "ELiRF_UPV"),
        run_id=int(raw.get("run_id", 1)),
        seed=int(raw.get("seed", 42)),
        max_eval_instances=max_eval,
        checkpoint_every=int(raw.get("checkpoint_every", 100)),
        use_rationale=bool(raw.get("use_rationale", False)),
        pool_seeds=int(raw.get("pool_seeds", 1)),
        video_fps=float(raw.get("video_fps", 0.5)),
        video_max_pixels=int(raw.get("video_max_pixels", 480 * 640)),
        categorization_fallback=raw.get("categorization_fallback"),
    )
    return replace(config, **overrides) if overrides else config


def load_fusion_config(path: str | Path, **overrides: Any) -> FusionConfig:
    """Build a :class:`FusionConfig` from ``configs/fusion_*.yaml``."""
    raw = load_yaml(path)
    root = resolve_project_root(raw.get("project_root"))
    modality = Modality(raw["modality"])
    physio_kind = raw["physio_kind"]
    if physio_kind not in ("flat", "matrix"):
        raise ConfigError(f"physio_kind must be 'flat' or 'matrix', got {physio_kind!r}")

    config = FusionConfig(
        modality=modality,
        data=_data_paths(root, raw["data"]),
        dirs=RunDirs(root / raw.get("run_dir", f"runs/fusion_{modality.value}")),
        text_model=raw["text_model"],
        max_len=int(raw["max_len"]),
        physio_kind=physio_kind,
        training=TrainingSpec(**(raw.get("training") or {})),
        hpo=HpoSpec(**(raw.get("hpo") or {})),
        team_name=raw.get("team_name", "ELiRF_UPV"),
        run_id=int(raw.get("run_id", 2)),
        seed=int(raw.get("seed", 42)),
        val_fraction=float(raw.get("val_fraction", 0.15)),
        max_subjects=int(raw.get("max_subjects", 4)),
        final_epochs=int(raw.get("final_epochs", 5)),
        yes_threshold=float(raw.get("yes_threshold", 0.5)),
    )
    return replace(config, **overrides) if overrides else config


def subtask_keys_for(modality: Modality) -> tuple[str, str, str]:
    return tuple(subtask_key(modality, s) for s in Subtask)  # type: ignore[return-value]


def _per_subtask(block: Mapping[str, Any], keys: tuple[str, ...]) -> list[Any]:
    """Accept per-subtask blocks keyed by ``t21``/``t31`` or by ``1``/``2``/``3``."""
    out = []
    for i, key in enumerate(keys, start=1):
        if key in block:
            out.append(block[key])
        elif str(i) in block:
            out.append(block[str(i)])
        elif i in block:
            out.append(block[i])
        else:
            raise ConfigError(f"missing entry for subtask {key} in {sorted(block)}")
    return out
