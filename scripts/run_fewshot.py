#!/usr/bin/env python3
"""Run the few-shot cascade (System B) end to end.

    python scripts/run_fewshot.py --config fewshot_memes --stage sanity
    python scripts/run_fewshot.py --config fewshot_videos --stage submission --backend qwen35

A sanity run scores a stratified TRAIN subset with PyEvALL (hard-hard) so the
prompts can be checked before the expensive run. A submission run predicts the
whole official TEST split and packages the upload. Both resume from their
checkpoints, so re-running after an interruption picks up where it stopped.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from exist2026.config import load_fewshot_config
from exist2026.datasets import describe, load_corpus
from exist2026.evaluation import HARD_METRICS, evaluate, hard_gold
from exist2026.evaluation.reports import compare, prediction_distribution, save_errors
from exist2026.fewshot import pools as pool_module
from exist2026.fewshot.backends import load_backend
from exist2026.fewshot.parsing import ParserPolicy, ResponseParser
from exist2026.fewshot.pipeline import (
    CheckpointStore,
    check_token_budget,
    majority_baseline,
    run_cascade,
)
from exist2026.fewshot.prompts import builders_for
from exist2026.submission import build_records, package, write_predictions


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="config name or path, e.g. fewshot_memes")
    parser.add_argument("--stage", choices=("sanity", "submission"), help="override the config stage")
    parser.add_argument("--backend", help="override the config backend, e.g. qwen35")
    parser.add_argument("--team-name", help="override the team name used in the submission")
    parser.add_argument("--run-id", type=int, choices=(1, 2, 3), help="official run id")
    parser.add_argument("--rebuild-pools", action="store_true", help="ignore cached few-shot pools")
    parser.add_argument("--skip-baseline", action="store_true", help="skip the majority baseline")
    parser.add_argument(
        "--dry-run", action="store_true", help="build pools and prompts, then stop before loading the model"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = {}
    if args.stage:
        overrides["stage"] = args.stage
    if args.backend:
        overrides["backend"] = args.backend
    config = load_fewshot_config(args.config, **overrides)
    if args.team_name:
        config = replace(config, team_name=args.team_name)
    if args.run_id:
        config = replace(config, run_id=args.run_id)
    config.dirs.create()

    print(f"modality={config.modality.value} backend={config.backend.name} stage={config.stage}")

    needs_test = config.run_mode == "test"
    corpus = load_corpus(
        config.modality,
        config.thresholds,
        train_json=config.data.train_json(),
        train_media_dir=config.data.train_media_dir(),
        test_json=config.data.test_json() if needs_test else None,
        test_media_dir=config.data.test_media_dir() if needs_test else None,
    )
    print(describe(corpus).to_string(index=False))

    pools = pool_module.load_or_build_pools(corpus, config.pools, config.dirs.pools, force=args.rebuild_pools)
    blocked = pool_module.pool_ids(pools)
    print(f"\nfew-shot pools ({len(blocked)} instances held out of evaluation):")
    print(pool_module.summarize(pools).to_string(index=False))

    builders = builders_for(
        config.modality,
        use_rationale=config.use_rationale,
        fps=config.video_fps if config.modality.value == "videos" else None,
        max_pixels=config.video_max_pixels if config.modality.value == "videos" else None,
    )

    if args.dry_run:
        print("\n--dry-run: pools and prompts built, stopping before model load.")
        return 0

    parser_policy = ParserPolicy.for_modality(
        config.modality, config.categorization_fallback or corpus.most_frequent_category()
    )
    parser = ResponseParser(parser_policy, log_path=config.dirs.logs / "parse_fallbacks.log")

    backend = load_backend(config.backend, config.generation)
    print("\ntoken budget:")
    print(check_token_budget(backend, builders, pools, corpus, config, blocked).to_string(index=False))

    store = CheckpointStore(config.dirs.checkpoints, config.checkpoint_suffix)
    result = run_cascade(
        backend=backend,
        builders=builders,
        pools=pools,
        parser=parser,
        corpus=corpus,
        config=config,
        store=store,
        blocked_ids=blocked,
    )

    for key, predictions in result.predictions.items():
        path = config.dirs.predictions / f"pred_{key}_hard_{config.checkpoint_suffix}.json"
        write_predictions(path, predictions, key)
        print(f"[{key}] {len(predictions):,} predictions -> {path.name}")

    print("\nparser fallbacks:")
    print(parser.report(config.subtask_keys).to_string(index=False, float_format="%.2f"))

    if config.run_mode == "train_eval":
        score(config, corpus, result, args.skip_baseline, blocked)
    else:
        for key, predictions in result.predictions.items():
            print(f"\n[{key}] prediction distribution:")
            print(prediction_distribution(predictions).to_string(index=False, float_format="%.1f"))
        base = package(
            result.predictions,
            modality=config.modality,
            output_dir=config.dirs.submission,
            team_name=config.team_name,
            run_id=config.run_id,
        )
        print(f"\nsubmission ready: {base} (+ .zip)")
    return 0


def score(config, corpus, result, skip_baseline: bool, blocked: set[str]) -> None:
    """Hard-hard PyEvALL scoring plus per-subtask error analysis."""
    rows = []
    for key, predictions in result.predictions.items():
        records = build_records(predictions)
        gold = hard_gold(corpus, [r["id"] for r in records], key)
        metrics = evaluate(records, gold, key, HARD_METRICS, work_dir=config.dirs.pyevall_work)
        rows.append({"subtask": key, "run": "model", **metrics})

        comparison = compare(predictions, corpus, key)
        print(f"\n[{key}] {comparison.metrics}")
        if comparison.note:
            print(f"  note: {comparison.note}")
        saved = save_errors(comparison, corpus, config.dirs.metrics / "errors")
        if saved:
            print(f"  errors -> {saved}")

    if not skip_baseline:
        for key, predictions in majority_baseline(config, corpus, blocked).items():
            records = build_records(predictions)
            gold = hard_gold(corpus, [r["id"] for r in records], key)
            metrics = evaluate(records, gold, key, HARD_METRICS, work_dir=config.dirs.pyevall_work)
            rows.append({"subtask": key, "run": "majority", **metrics})

    import pandas as pd

    summary = pd.DataFrame(rows)
    path = config.dirs.metrics / "metrics_summary.csv"
    summary.to_csv(path, index=False)
    print(f"\n{summary.to_string(index=False)}\n\nsaved -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
