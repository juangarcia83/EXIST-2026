"""Training, gated prediction and checkpoint selection for System A.

Two decisions shape this module:

* **The selection metric is the official one.** Checkpoints are kept by
  ICM-Soft-Norm on the full validation split *with the hierarchical gate
  applied* — the same quantity the lab scores — rather than by validation loss
  or F1. A model can improve its loss while getting worse at the metric that
  decides the ranking.
* **Subtasks x.2 and x.3 train on sexist instances only but are evaluated on
  everything.** Training them on non-sexist instances would teach a NO class
  the deployed cascade never asks them about; evaluating them only on sexist
  instances would hide the cost of the gate's mistakes.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from exist2026.evaluation.pyevall import SOFT_METRICS, certain_no, evaluate, soft_value
from exist2026.fusion.data import to_device
from exist2026.fusion.losses import loss_for
from exist2026.fusion.modeling import build_model
from exist2026.submission import PYEVALL_TEST_CASE
from exist2026.taxonomy import Subtask, subtask_of


@dataclass
class TrainingHistory:
    """Per-epoch record, for the training curves."""

    train_loss: list[float] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)
    icm_soft: list[float] = field(default_factory=list)
    icm_soft_norm: list[float] = field(default_factory=list)

    def append(self, loss: float, f1: float, metrics: Mapping[str, float]) -> None:
        self.train_loss.append(loss)
        self.f1.append(f1)
        self.icm_soft.append(metrics.get("ICMSoft", float("nan")))
        self.icm_soft_norm.append(metrics.get("ICMSoftNorm", float("nan")))


@dataclass
class TrainingResult:
    model: torch.nn.Module
    history: TrainingHistory
    best_metric: float
    checkpoint: Path | None


def device_of(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_optimizer(model: torch.nn.Module, hyperparameters: Mapping[str, Any]):
    """AdamW with a lower learning rate for the (optionally frozen) backbone."""
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if name.startswith("text_encoder.") else head).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": head, "lr": hyperparameters["lr_head"]},
            {"params": backbone, "lr": hyperparameters["lr_backbone"]},
        ],
        weight_decay=hyperparameters["weight_decay"],
    )


@torch.no_grad()
def proxy_f1(model: torch.nn.Module, loader, key: str, device: str) -> float:
    """A cheap hard-label F1, reported alongside the official soft metric.

    It is a sanity signal only: a model can move ICM-Soft while leaving argmax
    decisions untouched, and vice versa. Never select checkpoints on it.
    """
    from sklearn.metrics import f1_score

    model.eval()
    subtask = subtask_of(key)
    target_key = f"y_{key}"
    gold, predicted = [], []
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch)["logits"]
        if subtask is Subtask.IDENTIFICATION:
            predicted.append((F.softmax(logits, -1)[:, 0] >= 0.5).cpu().numpy())
            gold.append((batch[target_key][:, 0] >= 0.5).cpu().numpy())
        elif subtask is Subtask.INTENTION:
            predicted.append(logits.argmax(-1).cpu().numpy())
            gold.append(batch[target_key].argmax(-1).cpu().numpy())
        else:
            predicted.append((torch.sigmoid(logits) >= 0.5).cpu().numpy())
            gold.append((batch[target_key] >= 0.5).cpu().numpy())
    average = "binary" if subtask is Subtask.IDENTIFICATION else "macro"
    return float(f1_score(np.concatenate(gold), np.concatenate(predicted), average=average, zero_division=0))


@torch.no_grad()
def predict_soft(
    model: torch.nn.Module,
    loader,
    key: str,
    device: str,
    *,
    gate: torch.nn.Module | None = None,
    hierarchical: bool = True,
) -> list[dict[str, Any]]:
    """PyEvALL soft records, with the x.1 gate forcing NO downstream.

    When the gate says an instance is not sexist, x.2 and x.3 emit a certain NO
    instead of their own distribution — the deployed behaviour, so this is also
    what validation must score.
    """
    model.eval()
    if gate is not None and gate is not model:
        gate.eval()
    subtask = subtask_of(key)
    records: list[dict[str, Any]] = []

    for batch in loader:
        batch = to_device(batch, device)
        output = model(batch)
        probabilities = (
            (
                torch.sigmoid(output["logits"])
                if subtask is Subtask.CATEGORIZATION
                else F.softmax(output["logits"], -1)
            )
            .cpu()
            .numpy()
        )

        gate_probabilities = None
        if subtask is not Subtask.IDENTIFICATION and hierarchical:
            source = gate if gate is not None else model
            gate_probabilities = F.softmax(source(batch)["logits"], -1).cpu().numpy()

        for row, instance_id in enumerate(batch["id_EXIST"]):
            # Gate order is [P(YES), P(NO)]: index 1 winning means "not sexist".
            gated_no = (
                gate_probabilities is not None and gate_probabilities[row, 1] > gate_probabilities[row, 0]
            )
            value = certain_no(key) if gated_no else soft_value(key, probabilities[row])
            records.append({"test_case": PYEVALL_TEST_CASE, "id": str(instance_id), "value": value})
    return records


def evaluate_soft(
    model: torch.nn.Module,
    loader,
    key: str,
    gold: Sequence[Mapping[str, Any]],
    work_dir: Path,
    device: str,
    *,
    gate: torch.nn.Module | None = None,
) -> dict[str, float]:
    predictions = predict_soft(model, loader, key, device, gate=gate)
    return evaluate(predictions, gold, key, SOFT_METRICS, work_dir=work_dir)


def train_task(
    key: str,
    *,
    config,
    physio_in: int,
    train_loader,
    val_loader,
    gold: Sequence[Mapping[str, Any]],
    checkpoint_name: str,
    hyperparameters: Mapping[str, Any] | None = None,
    gate: torch.nn.Module | None = None,
    trial=None,
    device: str | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train one subtask, keeping the best epoch by ICM-Soft-Norm.

    ``trial`` makes the function Optuna-aware: the metric is reported each epoch
    and an unpromising trial is pruned, which is what makes a 20-trial search
    affordable.
    """
    from transformers import get_linear_schedule_with_warmup

    device = device_of(device)
    hyperparameters = config.training.merged(hyperparameters)
    model = build_model(key, config, hyperparameters, physio_in).to(device)
    optimizer = build_optimizer(model, hyperparameters)
    total_steps = len(train_loader) * hyperparameters["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * hyperparameters["warmup_frac"]), total_steps
    )
    compute_loss = loss_for(key)
    history = TrainingHistory()
    best_metric = -float("inf")
    checkpoint_path = config.dirs.checkpoints / checkpoint_name
    saved: Path | None = None

    for epoch in range(1, hyperparameters["epochs"] + 1):
        model.train()
        running, seen = 0.0, 0
        progress = tqdm(
            train_loader, desc=f"[{key}] epoch {epoch}/{hyperparameters['epochs']}", disable=not verbose
        )
        for batch in progress:
            batch = to_device(batch, device)
            loss = compute_loss(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparameters["grad_clip"])
            optimizer.step()
            scheduler.step()
            size = batch["input_ids"].size(0)
            running += loss.item() * size
            seen += size
            progress.set_postfix(loss=f"{running / max(1, seen):.4f}")

        train_loss = running / max(1, seen)
        f1 = proxy_f1(model, val_loader, key, device)
        try:
            metrics = evaluate_soft(model, val_loader, key, gold, config.dirs.pyevall_work, device, gate=gate)
        except Exception as error:
            if verbose:
                print(f"  [warn] PyEvALL failed this epoch: {error}")
            metrics = {}
        history.append(train_loss, f1, metrics)

        current = metrics.get("ICMSoftNorm", float("nan"))
        if verbose:
            print(
                f"  epoch {epoch}: loss={train_loss:.4f} F1={f1:.3f} "
                f"ICM-Soft={metrics.get('ICMSoft', float('nan')):.4f} "
                f"ICM-Soft-Norm={current:.4f}"
            )

        if trial is not None:
            trial.report(current if current == current else -1.0, step=epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()

        if current == current and current > best_metric:  # skips NaN
            best_metric = current
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "task": key,
                    "epoch": epoch,
                    "icm_soft": metrics.get("ICMSoft"),
                    "icm_soft_norm": current,
                    "hyperparameters": dict(hyperparameters),
                },
                checkpoint_path,
            )
            saved = checkpoint_path
            if verbose:
                print(f"  >> new best checkpoint ({current:.4f}) -> {checkpoint_path.name}")

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return TrainingResult(model=model, history=history, best_metric=best_metric, checkpoint=saved)


def load_checkpoint(key: str, path: Path, config, physio_in: int, device: str | None = None):
    """Rebuild a model from a checkpoint, using the hyperparameters it stored."""
    device = device_of(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    hyperparameters = config.training.merged(payload.get("hyperparameters") or {})
    model = build_model(key, config, hyperparameters, physio_in).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def refit_on_all(
    key: str,
    *,
    config,
    physio_in: int,
    loader,
    hyperparameters: Mapping[str, Any] | None = None,
    epochs: int | None = None,
    device: str | None = None,
    verbose: bool = True,
) -> torch.nn.Module:
    """Retrain on train+validation with the tuned hyperparameters.

    There is no held-out split left here and therefore no checkpoint selection:
    the epoch count is fixed to the one validation already chose.
    """
    from transformers import get_linear_schedule_with_warmup

    device = device_of(device)
    hyperparameters = config.training.merged(hyperparameters)
    if epochs:
        hyperparameters["epochs"] = epochs
    model = build_model(key, config, hyperparameters, physio_in).to(device)
    optimizer = build_optimizer(model, hyperparameters)
    total_steps = len(loader) * hyperparameters["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * hyperparameters["warmup_frac"]), total_steps
    )
    compute_loss = loss_for(key)

    for epoch in range(1, hyperparameters["epochs"] + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in tqdm(
            loader, desc=f"[refit {key}] epoch {epoch}/{hyperparameters['epochs']}", disable=not verbose
        ):
            batch = to_device(batch, device)
            loss = compute_loss(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparameters["grad_clip"])
            optimizer.step()
            scheduler.step()
            size = batch["input_ids"].size(0)
            running += loss.item() * size
            seen += size
        if verbose:
            print(f"  epoch {epoch}: loss={running / max(1, seen):.4f}")

    path = config.dirs.checkpoints / f"refit_{key}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "task": key, "hyperparameters": dict(hyperparameters)}, path)
    if verbose:
        print(f"  refit checkpoint -> {path.name}")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return model


def plot_curves(histories: Mapping[str, TrainingHistory], title: str, path: Path | None = None):
    """Loss against the official metric, one panel per subtask."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(histories), figsize=(5.5 * len(histories), 4), squeeze=False)
    for axis, (key, history) in zip(axes[0], histories.items(), strict=True):
        epochs = range(1, len(history.train_loss) + 1)
        axis.plot(epochs, history.train_loss, "-o", color="tab:gray", label="train loss")
        twin = axis.twinx()
        twin.plot(epochs, history.icm_soft_norm, "-s", color="tab:red", label="ICM-Soft-Norm")
        twin.plot(epochs, history.f1, "-^", color="tab:blue", alpha=0.6, label="F1 (proxy)")
        axis.set_title(key)
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        twin.set_ylabel("ICM-Soft-Norm / F1")
        lines = axis.get_lines() + twin.get_lines()
        axis.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="center right")
        axis.grid(alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=120, bbox_inches="tight")
    return figure
