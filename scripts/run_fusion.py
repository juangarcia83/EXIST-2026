#!/usr/bin/env python3
"""Run the cross-attention fusion system (System A).

    python scripts/run_fusion.py --config fusion_memes  --steps train
    python scripts/run_fusion.py --config fusion_videos --steps hpo,train,submit

Steps, in the order they must run:

``hpo``
    Sequential Optuna search: x.1 first, then x.2 / x.3 through the gate that
    x.1's best configuration produces. Writes ``best_hyperparameters.json``.
``train``
    Final training of the three subtasks, keeping the best epoch by
    ICM-Soft-Norm on validation, plus the training curves.
``submit``
    Refit on train+validation and write the soft submission for the official
    test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from exist2026.config import load_fusion_config
from exist2026.submission import package
from exist2026.taxonomy import Subtask, subtask_of

# torch, transformers and optuna are imported inside main() so that --help and
# argument validation work on a machine without the training stack installed.

BEST_HP_FILE = "best_hyperparameters.json"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="config name or path, e.g. fusion_memes")
    parser.add_argument("--steps", default="train", help="comma-separated: hpo, train, submit")
    parser.add_argument("--trials", type=int, help="override the number of Optuna trials")
    parser.add_argument("--epochs", type=int, help="override the number of final training epochs")
    parser.add_argument("--device", help="force a device, e.g. cpu")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_fusion_config(args.config)
    if args.epochs:
        config = replace(config, final_epochs=args.epochs)
    config.dirs.create()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - {"hpo", "train", "submit"}
    if unknown:
        raise SystemExit(f"unknown step(s): {sorted(unknown)}")

    from exist2026.fusion.experiment import FusionExperiment
    from exist2026.fusion.training import device_of

    device = device_of(args.device)
    print(f"modality={config.modality.value} physio={config.physio_kind} device={device}")

    experiment = FusionExperiment.build(config)
    print(experiment.split.summary().to_string(index=False))

    best_path = config.dirs.checkpoints / BEST_HP_FILE
    best: dict[str, dict] = json.loads(best_path.read_text()) if best_path.exists() else {}

    if "hpo" in steps:
        best = run_hpo(experiment, config, device, args.trials)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(json.dumps(best, indent=2))
        print(f"best hyperparameters -> {best_path}")

    if "train" in steps:
        run_training(experiment, config, device, best)

    if "submit" in steps:
        run_submission(experiment, config, device, best)
    return 0


def _train(experiment, config, key, device, hyperparameters, gate, checkpoint, trial=None, verbose=True):
    from exist2026.fusion.training import train_task

    merged = config.training.merged(hyperparameters)
    return train_task(
        key,
        config=config,
        physio_in=experiment.physio_in,
        train_loader=experiment.train_loader(key, merged["batch_size"], device),
        val_loader=experiment.val_loader(device=device),
        gold=experiment.gold(key),
        checkpoint_name=checkpoint,
        hyperparameters=hyperparameters,
        gate=gate,
        trial=trial,
        device=device,
        verbose=verbose,
    )


def run_hpo(experiment, config, device, trials):
    """Tune x.1, freeze it as the gate, then tune x.2 and x.3 through it."""
    from exist2026.fusion.hpo import optimize
    from exist2026.fusion.training import load_checkpoint

    k1, k2, k3 = config.subtask_keys
    best: dict[str, dict] = {}

    def objective_for(key, gate):
        def train_fn(hyperparameters, trial):
            return _train(
                experiment,
                config,
                key,
                device,
                hyperparameters,
                gate,
                f"_optuna_{key}.pt",
                trial=trial,
                verbose=False,
            ).best_metric

        return train_fn

    best[k1] = optimize(k1, objective_for(k1, None), config=config, n_trials=trials).best_params
    print(f"\nretraining {k1} with its best configuration to use as the gate")
    _train(experiment, config, k1, device, best[k1], None, f"best_{k1}.pt")
    gate, _ = load_checkpoint(
        k1, config.dirs.checkpoints / f"best_{k1}.pt", config, experiment.physio_in, device
    )
    for key in (k2, k3):
        best[key] = optimize(key, objective_for(key, gate), config=config, n_trials=trials).best_params
    return best


def run_training(experiment, config, device, best):
    """Final training of the three subtasks with the tuned hyperparameters."""
    from exist2026.fusion.training import load_checkpoint, plot_curves

    histories, gate = {}, None
    for key in config.subtask_keys:
        hyperparameters = {**best.get(key, {}), "epochs": config.final_epochs}
        result = _train(
            experiment,
            config,
            key,
            device,
            hyperparameters,
            None if subtask_of(key) is Subtask.IDENTIFICATION else gate,
            f"best_{key}.pt",
        )
        histories[key] = result.history
        print(f"{key}: best ICM-Soft-Norm = {result.best_metric:.4f}")
        if subtask_of(key) is Subtask.IDENTIFICATION:
            gate, _ = load_checkpoint(
                key, config.dirs.checkpoints / f"best_{key}.pt", config, experiment.physio_in, device
            )
    plot_curves(
        histories,
        f"{config.modality.value} - cross-attention fusion",
        config.dirs.figures / "train_curves.png",
    )
    print(f"curves -> {config.dirs.figures / 'train_curves.png'}")


def run_submission(experiment, config, device, best):
    """Refit on train+validation and write the official soft submission."""
    from exist2026.fusion.training import predict_soft, refit_on_all

    k1, _, _ = config.subtask_keys
    preprocessor = experiment.full_preprocessor()
    loader, ids = experiment.test_loader(preprocessor, device=device)
    if loader is None:
        print("[!] no test split configured; skipping the submission step")
        return

    models = {}
    for key in config.subtask_keys:
        models[key] = refit_on_all(
            key,
            config=config,
            physio_in=experiment.physio_in,
            loader=experiment.full_loader(
                key, config.training.merged(best.get(key, {}))["batch_size"], preprocessor, device
            ),
            hyperparameters=best.get(key, {}),
            epochs=config.final_epochs,
            device=device,
        )

    predictions = {}
    for key in config.subtask_keys:
        records = predict_soft(
            models[key],
            loader,
            key,
            device,
            gate=None if subtask_of(key) is Subtask.IDENTIFICATION else models[k1],
        )
        predictions[key] = {record["id"]: record["value"] for record in records}
        print(f"[{key}] {len(predictions[key]):,} predictions over {len(ids):,} test instances")

    base = package(
        predictions,
        modality=config.modality,
        output_dir=config.dirs.submission,
        team_name=config.team_name,
        run_id=config.run_id,
        evaluation_context="soft",
    )
    print(f"submission ready: {base} (+ .zip)")


if __name__ == "__main__":
    raise SystemExit(main())
