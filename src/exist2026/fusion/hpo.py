"""Hyperparameter search with Optuna (TPE sampler + median pruner).

The search is *sequential over the hierarchy*: x.1 is tuned first, its best
configuration is retrained and frozen as the gate, and x.2 / x.3 are then tuned
through that gate. Tuning them against a moving gate would optimize each
subtask against a model that will not exist at inference time.

The objective is ICM-Soft-Norm on validation — the metric the lab ranks on —
and every ``fusion_hidden * num_heads`` combination in the space divides
evenly, so no trial dies on a dimension error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def suggest(trial, epochs: int) -> dict[str, Any]:
    """One sampled configuration."""
    return {
        "lr_head": trial.suggest_float("lr_head", 1e-4, 5e-3, log=True),
        "lr_backbone": trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True),
        "warmup_frac": trial.suggest_float("warmup_frac", 0.0, 0.2),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 2.0),
        "p_drop": trial.suggest_float("p_drop", 0.1, 0.4),
        "sensor_hidden": trial.suggest_categorical("sensor_hidden", [128, 256, 384]),
        "sensor_out": trial.suggest_categorical("sensor_out", [64, 128, 192]),
        "fusion_hidden": trial.suggest_categorical("fusion_hidden", [256, 384, 512, 768]),
        "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "epochs": epochs,
    }


def optimize(
    key: str,
    train_fn: Callable[[Mapping[str, Any], Any], float],
    *,
    config,
    n_trials: int | None = None,
    verbose: bool = True,
):
    """Run a study for one subtask and return it.

    ``train_fn(hyperparameters, trial) -> best validation metric`` keeps this
    module independent of how a model is built or where its data comes from.
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = n_trials or config.hpo.n_trials

    def objective(trial):
        hyperparameters = suggest(trial, config.hpo.epochs)
        best = train_fn(hyperparameters, trial)
        # A trial that never produced a finite metric must not win by NaN.
        return best if (best == best and best > -float("inf")) else -1.0

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.seed, multivariate=True, n_startup_trials=config.hpo.startup_trials),
        pruner=MedianPruner(
            n_startup_trials=config.hpo.startup_trials,
            n_warmup_steps=config.hpo.warmup_steps,
            interval_steps=1,
        ),
        study_name=f"{config.modality.value}_{key}",
    )
    if verbose:
        print(f"Optuna [{key}]: {n_trials} trials x {config.hpo.epochs} epochs")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    if verbose:
        pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
        print(f"  best ICM-Soft-Norm = {study.best_value:.4f} ({pruned}/{len(study.trials)} pruned)")
        for name, value in study.best_params.items():
            print(f"    {name:16s} = {value}")
    return study
