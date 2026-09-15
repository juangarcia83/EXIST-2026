"""The hierarchical few-shot cascade.

Subtask x.1 runs over every instance; x.2 and x.3 only run on what x.1 called
sexist, and anything x.1 called NO receives ``NO`` / ``["NO"]`` automatically.
This mirrors the official hierarchy (guidelines p. 17), keeps the three
subtasks consistent with one another, and removes roughly half the x.2/x.3
generations — which for videos is most of the run time.

Every subtask checkpoints to disk. The pipeline is written to be interrupted:
re-running it skips instances already predicted, so a disconnected Colab
session or an OOM on instance 400 costs one instance, not the run.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from exist2026.config import FewShotConfig
from exist2026.datasets import Corpus, exclude
from exist2026.fewshot.parsing import ResponseParser
from exist2026.fewshot.prompts import PromptBuilder
from exist2026.io_utils import read_json, write_json
from exist2026.taxonomy import Subtask, subtask_of


class CheckpointStore:
    """Per-subtask prediction cache, namespaced by stage.

    Sanity and submission runs write to different files, so running a sanity
    check first never contaminates the official run that follows it.
    """

    def __init__(self, directory: Path, suffix: str) -> None:
        self.directory = directory
        self.suffix = suffix
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.directory / f"predictions_{key}_{self.suffix}_partial.json"

    def load(self, key: str) -> dict[str, Any]:
        path = self.path(key)
        return read_json(path) if path.exists() else {}

    def save(self, key: str, predictions: Mapping[str, Any]) -> Path:
        return write_json(self.path(key), dict(predictions), indent=None)


@dataclass
class CascadeResult:
    """Predictions of the three subtasks plus the gate they agreed on."""

    predictions: dict[str, dict[str, Any]]
    sexist_ids: set[str]
    evaluated_ids: dict[str, set[str]] = field(default_factory=dict)

    def __getitem__(self, key: str) -> dict[str, Any]:
        return self.predictions[key]


def evaluation_frame(config: FewShotConfig, corpus: Corpus, blocked_ids: set[str], key: str) -> pd.DataFrame:
    """Instances to predict for one subtask.

    On a submission run this is the whole official TEST split. On a sanity run
    it is one shared subset of TRAIN — shared so the hierarchy stays coherent
    across the three subtasks — filtered per subtask to the instances that have
    a usable hard label to score against.
    """
    if config.run_mode == "test":
        frame = corpus.test[corpus.test["media_ok"]].copy()
        if config.max_eval_instances is not None and len(frame) > config.max_eval_instances:
            frame = frame.sample(n=config.max_eval_instances, random_state=config.seed)
        return frame.reset_index(drop=True)

    sample_ids = sanity_sample_ids(config, corpus, blocked_ids)
    base = corpus.train[corpus.train["id_EXIST"].isin(sample_ids) & corpus.train["media_ok"]]
    subtask = subtask_of(key)
    if subtask is Subtask.CATEGORIZATION:
        return base.copy().reset_index(drop=True)
    column = corpus.hard_column(subtask)
    return base[base[column].notna()].copy().reset_index(drop=True)


def sanity_sample_ids(config: FewShotConfig, corpus: Corpus, blocked_ids: set[str]) -> set[str]:
    """A stratified TRAIN subset for self-evaluation, excluding pool members.

    Stratifying on the x.1 hard label matters: the corpus is unbalanced enough
    that a plain random draw of a few hundred instances can misrepresent the
    YES rate, and every downstream metric inherits that.
    """
    column = corpus.hard_column(Subtask.IDENTIFICATION)
    base = exclude(corpus.train[corpus.train["media_ok"]], blocked_ids)
    base = base[base[column].notna()]
    limit = config.max_eval_instances
    if limit is not None and len(base) > limit:
        try:
            from sklearn.model_selection import train_test_split

            base, _ = train_test_split(
                base,
                train_size=limit,
                stratify=base[column].astype(int),
                random_state=config.seed,
            )
        except (ImportError, ValueError):
            base = base.sample(n=limit, random_state=config.seed)
    return set(base["id_EXIST"].astype(str))


def propagated_value(key: str) -> Any:
    """What a NO from x.1 means downstream."""
    return ["NO"] if subtask_of(key) is Subtask.CATEGORIZATION else "NO"


def exception_fallback(key: str, parser: ResponseParser) -> Any:
    """Label assigned when generation itself fails (OOM, missing file, ...)."""
    subtask = subtask_of(key)
    if subtask is Subtask.CATEGORIZATION:
        return list(parser.policy.categorization_fallback)
    if subtask is Subtask.INTENTION:
        return parser.policy.intention_fallback
    return "NO"


def predict_one(
    backend,
    builder: PromptBuilder,
    parser: ResponseParser,
    pool: Sequence[Mapping[str, Any]],
    corpus: Corpus,
    config: FewShotConfig,
    instance_id: str,
    media_path: str,
    text: str,
    *,
    return_raw: bool = False,
):
    """Prompt the backend for one instance and parse the answer."""
    messages = builder.build(media_path, text, pool, corpus.media_of, corpus.text_of)
    raw = backend.generate(messages, config.generation.max_new_tokens[builder.key])
    parsed = parser.parse(raw, builder.subtask, builder.key, instance_id)
    return (parsed, raw) if return_raw else parsed


def run_subtask(
    key: str,
    *,
    backend,
    builders: Mapping[str, PromptBuilder],
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    parser: ResponseParser,
    corpus: Corpus,
    config: FewShotConfig,
    store: CheckpointStore,
    target_ids: set[str],
    cleanup: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Predict one subtask over ``target_ids``, resuming from the checkpoint.

    Generation failures never abort the run: the instance gets the subtask's
    fallback label and the exception is logged with its type, so the fallback
    report can separate "the model answered something unparseable" from "the
    GPU ran out of memory".
    """
    builder = builders[key]
    frame = corpus.frame[
        corpus.frame["id_EXIST"].astype(str).isin({str(i) for i in target_ids}) & corpus.frame["media_ok"]
    ]
    predictions = store.load(key)
    todo = frame[~frame["id_EXIST"].isin(predictions.keys())]

    print(f"[{key}] target={len(frame):,}  cached={len(predictions):,}  pending={len(todo):,}")
    if todo.empty:
        return predictions

    progress = tqdm(todo.itertuples(index=False), total=len(todo), desc=f"infer {key}")
    for done, row in enumerate(progress, start=1):
        try:
            prediction = predict_one(
                backend,
                builder,
                parser,
                pools[key],
                corpus,
                config,
                row.id_EXIST,
                row.media_path,
                row.text,
            )
        except Exception as error:
            prediction = exception_fallback(key, parser)
            parser.record_exception(key, row.id_EXIST, error, prediction)
            print(f"  [{type(error).__name__}] {row.id_EXIST}: {error}")
            if cleanup:
                cleanup()
        predictions[row.id_EXIST] = prediction
        if done % config.checkpoint_every == 0:
            store.save(key, predictions)
            if cleanup:
                cleanup()
            progress.set_postfix(fallbacks=f"{parser.fallbacks[key]}/{parser.total[key]}")

    store.save(key, predictions)
    if cleanup:
        cleanup()
    rate = parser.fallback_rate(key)
    print(f"[{key}] fallbacks: {parser.fallbacks[key]}/{parser.total[key]} ({100 * rate:.1f}%)")
    if rate > 0.05:
        print(f"[{key}] WARNING: over 5% fallbacks — inspect the prompt or the token budget.")
    return predictions


def cuda_cleanup() -> None:
    """Free cached VRAM between chunks; a no-op without CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def run_cascade(
    *,
    backend,
    builders: Mapping[str, PromptBuilder],
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    parser: ResponseParser,
    corpus: Corpus,
    config: FewShotConfig,
    store: CheckpointStore,
    blocked_ids: set[str] | None = None,
) -> CascadeResult:
    """Run x.1 over everything, then x.2/x.3 over the instances x.1 called sexist."""
    blocked_ids = blocked_ids or set()
    k1, k2, k3 = config.subtask_keys
    frames = {key: evaluation_frame(config, corpus, blocked_ids, key) for key in (k1, k2, k3)}
    evaluated = {key: set(frame["id_EXIST"]) for key, frame in frames.items()}
    union = set().union(*evaluated.values())
    print(f"evaluation set: {len(union):,} instances ({k1} runs over all of them)")

    predictions = {
        k1: run_subtask(
            k1,
            backend=backend,
            builders=builders,
            pools=pools,
            parser=parser,
            corpus=corpus,
            config=config,
            store=store,
            target_ids=union,
            cleanup=cuda_cleanup,
        )
    }
    sexist = {i for i, value in predictions[k1].items() if value == "YES"}
    share = 100 * len(sexist) / max(1, len(predictions[k1]))
    print(f"{k1} -> {len(sexist):,} sexist ({share:.1f}%)")

    for key in (k2, k3):
        predictions[key] = run_subtask(
            key,
            backend=backend,
            builders=builders,
            pools=pools,
            parser=parser,
            corpus=corpus,
            config=config,
            store=store,
            target_ids=sexist & evaluated[key],
            cleanup=cuda_cleanup,
        )
        # Enforce the hierarchy: x.1 has the last word, even over a cached value.
        for instance_id in evaluated[key]:
            if predictions[k1].get(instance_id) == "NO":
                predictions[key][instance_id] = propagated_value(key)
            elif instance_id not in predictions[key]:
                row = corpus.row(instance_id)
                predictions[key][instance_id] = predict_one(
                    backend,
                    builders[key],
                    parser,
                    pools[key],
                    corpus,
                    config,
                    instance_id,
                    row["media_path"],
                    row["text"],
                )
        store.save(key, predictions[key])

    return CascadeResult(predictions=predictions, sexist_ids=sexist, evaluated_ids=evaluated)


def majority_baseline(
    config: FewShotConfig, corpus: Corpus, blocked_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Trivial floor: the modal TRAIN class, propagated through the hierarchy.

    A system that does not beat this is not working, whatever its absolute
    numbers look like.
    """
    k1, k2, k3 = config.subtask_keys
    column = corpus.hard_column(Subtask.IDENTIFICATION)
    labels = corpus.train[column].dropna().astype(int)
    majority = "YES" if labels.mean() >= 0.5 else "NO"

    frames = {key: evaluation_frame(config, corpus, blocked_ids, key) for key in (k1, k2, k3)}
    union = set().union(*(set(frame["id_EXIST"]) for frame in frames.values()))

    if majority == "NO":
        downstream = {k2: "NO", k3: ["NO"]}
    else:
        downstream = {k2: "DIRECT", k3: [corpus.most_frequent_category()]}
    return {
        k1: dict.fromkeys(union, majority),
        k2: dict.fromkeys(frames[k2]["id_EXIST"], downstream[k2]),
        k3: {instance_id: list(downstream[k3]) for instance_id in frames[k3]["id_EXIST"]},
    }


def check_token_budget(
    backend,
    builders: Mapping[str, PromptBuilder],
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    corpus: Corpus,
    config: FewShotConfig,
    blocked_ids: set[str],
) -> pd.DataFrame:
    """Measure the real prompt length of each subtask before a long run.

    Video prompts carry several sampled frames per exemplar, so a pool that is
    one exemplar too large silently overflows the context. Checking costs
    seconds; finding out after six hours of inference does not.
    """
    split = corpus.test if config.run_mode == "test" else corpus.train
    query = exclude(split, blocked_ids).iloc[0]
    max_context = backend.max_context()
    rows = []
    for key, builder in builders.items():
        messages = builder.build(
            query["media_path"], query["text"], pools[key], corpus.media_of, corpus.text_of
        )
        n_tokens = backend.prompt_tokens(messages)
        ratio = n_tokens / max_context
        rows.append(
            {
                "subtask": key,
                "prompt_tokens": n_tokens,
                "max_context": max_context,
                "pct_used": 100 * ratio,
                "status": "OVERFLOW" if ratio > 1 else ("WARN" if ratio > 0.8 else "ok"),
            }
        )
    report = pd.DataFrame(rows)
    overflowing = report[report["status"] == "OVERFLOW"]
    if not overflowing.empty:
        raise ValueError(
            f"prompt exceeds the context window for {overflowing['subtask'].tolist()}; "
            "reduce video_fps / video_max_pixels or the number of exemplars per class."
        )
    return report
