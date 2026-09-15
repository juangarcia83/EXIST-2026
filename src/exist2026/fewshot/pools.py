"""Anti-leakage few-shot exemplar pools.

Three properties matter and are enforced here rather than by convention:

1. **No leakage.** Pools are built from the TRAIN partition only, and every
   instance that ends up in a pool is excluded from evaluation.
2. **Determinism.** With ``seed=None`` selection is a pure function of the
   corpus (ties broken by id), so the same data always yields the same prompt.
   An integer seed adds a random tie-break, which is how pool variance is
   measured without touching anything else.
3. **Hierarchy consistency.** Subtasks x.2 and x.3 only ever see instances the
   cascade already called sexist, so their pools contain no NO exemplar. A
   prompt that forbids NO while showing a NO exemplar would contradict itself.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from exist2026.config import PoolSpec
from exist2026.datasets import Corpus
from exist2026.io_utils import read_json, write_json
from exist2026.labels.hard import (
    active_categories,
    categorization_consensus,
    identification_consensus,
    intention_consensus,
)
from exist2026.taxonomy import (
    IDENTIFICATION_INT_TO_STR,
    INTENTION_INT_TO_STR,
    SEXISM_CATEGORIES,
    Subtask,
    subtask_key,
)


@dataclass(frozen=True)
class Candidate:
    """One TRAIN instance considered for a pool."""

    instance_id: str
    lang: str
    #: Annotator agreement, reported in the pool file for auditability.
    consensus: float
    #: Ranking key. Equal to ``consensus`` except in x.3, which also rewards
    #: single-label (prototypical) instances.
    sort_score: float
    label: str | None = None
    labels: tuple[str, ...] = ()

    def as_exemplar(self) -> dict[str, Any]:
        exemplar: dict[str, Any] = {"id": self.instance_id}
        if self.label is not None:
            exemplar["label"] = self.label
        else:
            exemplar["labels"] = list(self.labels)
        exemplar["consensus"] = round(float(self.consensus), 6)
        exemplar["lang"] = self.lang
        return exemplar


def _language(corpus: Corpus, instance_id: str) -> str:
    record = corpus.raw.get(str(instance_id), {})
    return (record.get("lang") or "en").lower()[:2]


def _sorted(candidates: Sequence[Candidate], seed: int | None, offset: int = 0) -> list[Candidate]:
    """Rank by score, descending. Ties break on id, or randomly with a seed."""
    if seed is None:
        return sorted(candidates, key=lambda c: (-c.sort_score, c.instance_id))
    rng = random.Random(int(seed) + offset)
    decorated = [(c, rng.random()) for c in candidates]
    decorated.sort(key=lambda pair: (-pair[0].sort_score, pair[1]))
    return [c for c, _ in decorated]


def _take_with_consensus_mix(ranked: Sequence[Candidate], n: int, spec: PoolSpec) -> list[Candidate]:
    """Mix high-agreement and borderline exemplars instead of taking the top-n.

    A pool made only of unanimous cases teaches the model what an easy instance
    looks like; adding a few medium-agreement ones shows it the boundary. Any
    shortfall in either band is back-filled from the ranking.
    """
    if n <= 0:
        return []
    low, high = spec.medium_consensus_range
    n_high = max(1, round(n * spec.high_consensus_fraction))
    picked = [c for c in ranked if c.consensus >= high][:n_high]
    picked += [c for c in ranked if low <= c.consensus < high][: max(0, n - n_high)]
    return _backfill(picked, ranked, n)


def _take_bilingual(ranked: Sequence[Candidate], n: int) -> list[Candidate]:
    """Half Spanish, half English; back-fill from the ranking if one side is short."""
    if n <= 0 or not ranked:
        return []
    by_lang: dict[str, list[Candidate]] = {}
    for candidate in ranked:
        by_lang.setdefault(candidate.lang, []).append(candidate)
    n_es = n // 2
    picked = by_lang.get("es", [])[:n_es] + by_lang.get("en", [])[: n - n_es]
    return _backfill(picked, ranked, n)


def _backfill(picked: Sequence[Candidate], ranked: Sequence[Candidate], n: int) -> list[Candidate]:
    out = list(picked)
    if len(out) >= n:
        return out[:n]
    used = {c.instance_id for c in out}
    for candidate in ranked:
        if candidate.instance_id in used:
            continue
        out.append(candidate)
        used.add(candidate.instance_id)
        if len(out) >= n:
            break
    return out[:n]


def _select(ranked: Sequence[Candidate], n: int, spec: PoolSpec) -> list[Candidate]:
    """Apply the pool's selection strategy to one already-ranked class."""
    if not spec.diversify_consensus and not spec.bilingual:
        return list(ranked[:n])

    pool = list(ranked)
    if spec.diversify_consensus:
        diversified = {c.instance_id for c in _take_with_consensus_mix(ranked, n, spec)}
        pool = [c for c in ranked if c.instance_id in diversified]

    if not spec.bilingual:
        return pool[:n]

    balanced = _take_bilingual(pool, n)
    if len(balanced) < n:
        balanced = _take_bilingual(ranked, n)
    return balanced


# --------------------------------------------------------------------------- #
# Per-subtask builders
# --------------------------------------------------------------------------- #


def _mono_label_candidates(
    corpus: Corpus,
    frame: pd.DataFrame,
    subtask: Subtask,
    consensus_of: Callable[[Sequence[str]], float],
    int_to_str: Mapping[int, str],
) -> list[Candidate]:
    column = corpus.hard_column(subtask)
    candidates = []
    for row in frame.itertuples(index=False):
        value = getattr(row, column)
        if pd.isna(value) or not row.media_ok:
            continue
        consensus = consensus_of(corpus.annotations(row.id_EXIST, subtask))
        candidates.append(
            Candidate(
                instance_id=row.id_EXIST,
                lang=_language(corpus, row.id_EXIST),
                consensus=consensus,
                sort_score=consensus,
                label=int_to_str[int(value)],
            )
        )
    return candidates


def build_identification_pool(
    corpus: Corpus, frame: pd.DataFrame, spec: PoolSpec, seed: int | None = None
) -> list[dict[str, Any]]:
    """``per_class`` YES exemplars plus ``per_class`` NO exemplars."""
    candidates = _mono_label_candidates(
        corpus, frame, Subtask.IDENTIFICATION, identification_consensus, IDENTIFICATION_INT_TO_STR
    )
    ranked = _sorted(candidates, seed)
    pool: list[Candidate] = []
    for label in ("YES", "NO"):
        pool += _select([c for c in ranked if c.label == label], spec.per_class, spec)
    return [c.as_exemplar() for c in pool]


def build_intention_pool(
    corpus: Corpus, frame: pd.DataFrame, spec: PoolSpec, seed: int | None = None
) -> list[dict[str, Any]]:
    """DIRECT and JUDGEMENTAL exemplars only — NO never reaches this subtask."""
    candidates = _mono_label_candidates(
        corpus, frame, Subtask.INTENTION, intention_consensus, INTENTION_INT_TO_STR
    )
    ranked = _sorted([c for c in candidates if c.label != "NO"], seed)
    pool: list[Candidate] = []
    for label in ("DIRECT", "JUDGEMENTAL"):
        pool += _select([c for c in ranked if c.label == label], spec.per_class, spec)
    return [c.as_exemplar() for c in pool]


def build_categorization_pool(
    corpus: Corpus, frame: pd.DataFrame, spec: PoolSpec, seed: int | None = None
) -> list[dict[str, Any]]:
    """``per_class`` exemplars per sexism category, prototypical ones first.

    Candidates are ranked by ``0.6 * single_label_rate + 0.4 * agreement``: a
    meme that most annotators tagged with *only* this category teaches the
    category more cleanly than one buried in a four-label annotation. Each
    instance is used for at most one category. When ``cooccurrence_extra`` is
    set, genuinely multi-label exemplars are appended so the prompt also shows
    that categories combine.
    """
    column = corpus.hard_column(Subtask.CATEGORIZATION)
    pool: list[Candidate] = []
    used: set[str] = set()

    for offset, category in enumerate(SEXISM_CATEGORIES):
        candidates = []
        for row in frame.itertuples(index=False):
            if row.id_EXIST in used or not row.media_ok:
                continue
            multi_hot = getattr(row, column)
            if multi_hot is None or multi_hot[offset] < 0.5:
                continue
            annotations = corpus.annotations(row.id_EXIST, Subtask.CATEGORIZATION)
            single_rate, agreement = categorization_consensus(annotations, category)
            if single_rate == 0.0 and agreement == 0.0:
                continue  # no usable annotation
            candidates.append(
                Candidate(
                    instance_id=row.id_EXIST,
                    lang=_language(corpus, row.id_EXIST),
                    consensus=agreement,
                    sort_score=single_rate * 0.6 + agreement * 0.4,
                    labels=tuple(active_categories(multi_hot)),
                )
            )
        ranked = _sorted(candidates, seed, offset=offset)
        if spec.bilingual:
            chosen = _take_bilingual(ranked[: max(spec.per_class * 4, spec.per_class)], spec.per_class)
        else:
            chosen = list(ranked[: spec.per_class])
        for candidate in chosen:
            pool.append(candidate)
            used.add(candidate.instance_id)

    pool += _cooccurrence_candidates(corpus, frame, spec, used, seed)
    return [c.as_exemplar() for c in pool]


def _cooccurrence_candidates(
    corpus: Corpus,
    frame: pd.DataFrame,
    spec: PoolSpec,
    used: set[str],
    seed: int | None,
) -> list[Candidate]:
    """High-agreement exemplars carrying two or more categories at once."""
    if spec.cooccurrence_extra <= 0:
        return []
    column = corpus.hard_column(Subtask.CATEGORIZATION)
    candidates = []
    for row in frame.itertuples(index=False):
        if row.id_EXIST in used or not row.media_ok:
            continue
        multi_hot = getattr(row, column)
        if multi_hot is None:
            continue
        categories = active_categories(multi_hot)
        if len(categories) < 2:
            continue
        agreements = [
            categorization_consensus(corpus.annotations(row.id_EXIST, Subtask.CATEGORIZATION), category)[1]
            for category in categories
        ]
        mean_agreement = sum(agreements) / len(agreements)
        if mean_agreement < 0.5:
            continue
        candidates.append(
            Candidate(
                instance_id=row.id_EXIST,
                lang=_language(corpus, row.id_EXIST),
                consensus=mean_agreement,
                sort_score=mean_agreement,
                labels=tuple(categories),
            )
        )
    if seed is None:
        candidates.sort(key=lambda c: (-len(c.labels), -c.sort_score, c.instance_id))
    else:
        rng = random.Random(int(seed) + 7)
        decorated = [(c, rng.random()) for c in candidates]
        decorated.sort(key=lambda pair: (-len(pair[0].labels), -pair[0].sort_score, pair[1]))
        candidates = [c for c, _ in decorated]
    return candidates[: spec.cooccurrence_extra]


BUILDERS: Mapping[Subtask, Callable[..., list[dict[str, Any]]]] = {
    Subtask.IDENTIFICATION: build_identification_pool,
    Subtask.INTENTION: build_intention_pool,
    Subtask.CATEGORIZATION: build_categorization_pool,
}


def build_pools(
    corpus: Corpus, specs: Mapping[str, PoolSpec], seed: int | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Build all three pools from TRAIN, keyed by subtask key."""
    train = corpus.train
    pools = {}
    for subtask in Subtask:
        key = subtask_key(corpus.modality, subtask)
        pools[key] = BUILDERS[subtask](corpus, train, specs[key], seed)
    return pools


def is_stale(key: str, cached: Sequence[Mapping[str, Any]], spec: PoolSpec) -> bool:
    """Whether a cached pool file predates the current pool design.

    Pools are cached to keep prompts identical across runs, but a cache written
    by an older design would silently contradict the current prompts, so it is
    detected and rebuilt rather than trusted.
    """
    if not cached:
        return True
    subtask_index = int(key[-1])
    if subtask_index == 2 and any(ex.get("label") == "NO" for ex in cached):
        return True  # x.2 pool from before the hierarchy-consistent design
    if subtask_index == 3:
        if any(list(ex.get("labels", [])) == ["NO"] for ex in cached):
            return True
        expected = len(SEXISM_CATEGORIES) * spec.per_class + spec.cooccurrence_extra
        if len(cached) != expected:
            return True
    return "lang" not in cached[0]


def load_or_build_pools(
    corpus: Corpus,
    specs: Mapping[str, PoolSpec],
    directory: Path,
    seed: int | None = None,
    *,
    force: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Read cached pools from ``directory``, rebuilding stale or missing ones."""
    directory.mkdir(parents=True, exist_ok=True)
    pools: dict[str, list[dict[str, Any]]] = {}
    for subtask in Subtask:
        key = subtask_key(corpus.modality, subtask)
        path = directory / f"few_shot_pool_{key}.json"
        if path.exists() and not force:
            cached = read_json(path)
            if not is_stale(key, cached, specs[key]):
                pools[key] = cached
                continue
        pools[key] = BUILDERS[subtask](corpus, corpus.train, specs[key], seed)
        write_json(path, pools[key])
    return pools


def pool_ids(pools: Mapping[str, Sequence[Mapping[str, Any]]]) -> set[str]:
    """Every id used as an exemplar — excluded from evaluation to avoid leakage."""
    return {str(ex["id"]) for pool in pools.values() for ex in pool}


def summarize(pools: Mapping[str, Sequence[Mapping[str, Any]]]) -> pd.DataFrame:
    """One row per exemplar, for eyeballing a pool in a notebook."""
    rows = []
    for key, pool in pools.items():
        for exemplar in pool:
            rows.append(
                {
                    "subtask": key,
                    "id": exemplar["id"],
                    "label": exemplar.get("label") or ", ".join(exemplar.get("labels", [])),
                    "consensus": round(float(exemplar.get("consensus", 0.0)), 3),
                    "lang": exemplar.get("lang", "?"),
                }
            )
    return pd.DataFrame(rows)


def rebuild_for_seed(
    corpus: Corpus, specs: Mapping[str, PoolSpec], seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Pools with a seeded tie-break, for the pool-variance experiment."""
    return build_pools(corpus, specs, seed=seed)


__all__ = [
    "Candidate",
    "build_categorization_pool",
    "build_identification_pool",
    "build_intention_pool",
    "build_pools",
    "is_stale",
    "load_or_build_pools",
    "pool_ids",
    "rebuild_for_seed",
    "summarize",
]
