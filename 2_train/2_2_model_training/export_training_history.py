#!/usr/bin/env python3
"""Reproduce the selected FCNN run and export its per-epoch loss history."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_ENGINEERING_DIR = SCRIPT_DIR.parent / "2_1_feature_engineering"
if str(FEATURE_ENGINEERING_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_ENGINEERING_DIR))

from fcnn_model import MultiTargetFCNN
from feature_engineering import engineer_features
from train_fcnn import read_dataset


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Reproduce the chosen initialization and export train/validation loss."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir.parent.parent / "0_raw_data" / "parameter.csv",
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=script_dir
    )
    parser.add_argument(
        "--spxy-dir", type=Path, default=script_dir.parent.parent / "1_SPXY"
    )
    parser.add_argument("--max-epochs", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    return parser.parse_args()


def save_history(rows: List[Dict[str, float]], path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "learning_rate",
        "improved",
        "best_validation_loss_so_far",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    rows: List[Dict[str, float]], best_epoch: int, best_loss: float, path: Path
) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    validation_loss = [row["validation_loss"] for row in rows]

    fig, axis = plt.subplots(figsize=(9.2, 5.6))
    axis.plot(epochs, train_loss, color="#2563EB", linewidth=1.4, label="Training loss")
    axis.plot(
        epochs,
        validation_loss,
        color="#DC2626",
        linewidth=1.2,
        alpha=0.9,
        label="Validation loss",
    )
    axis.axvline(
        best_epoch,
        color="#111827",
        linestyle="--",
        linewidth=1.0,
        label=f"Best epoch = {best_epoch}",
    )
    axis.scatter(
        [best_epoch], [best_loss], color="#111827", s=28, zorder=4, label=f"Best val MSE = {best_loss:.6f}"
    )
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Standardized MSE (log scale)")
    axis.set_title("Final FCNN training history (initialization seed 42002)")
    axis.grid(True, which="major", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axis.grid(True, which="minor", color="#E5E7EB", linewidth=0.4, alpha=0.5)
    axis.legend(frameon=False, ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    checkpoint_path = artifact_dir / "fcnn_model.pt"
    metrics_path = artifact_dir / "metrics.json"
    split_path = args.spxy_dir.resolve() / "split_indices.json"

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)

    torch.set_num_threads(1)
    random.seed(int(metrics["split"]["seed"]))
    torch.manual_seed(int(metrics["split"]["seed"]))
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    input_names, _, raw_features, targets = read_dataset(args.data.resolve())
    engineered, _ = engineer_features(raw_features, input_names)
    selected_indices = checkpoint["selected_feature_indices"].to(dtype=torch.long)
    model_features = engineered[:, selected_indices]

    train_indices = torch.tensor(split["train_indices"], dtype=torch.long)
    validation_indices = torch.tensor(split["validation_indices"], dtype=torch.long)
    x_train = (model_features[train_indices] - checkpoint["input_mean"]) / checkpoint[
        "input_std"
    ]
    x_validation = (
        model_features[validation_indices] - checkpoint["input_mean"]
    ) / checkpoint["input_std"]
    y_train = (targets[train_indices] - checkpoint["target_mean"]) / checkpoint[
        "target_std"
    ]
    y_validation = (
        targets[validation_indices] - checkpoint["target_mean"]
    ) / checkpoint["target_std"]

    initialization_seed = int(checkpoint["initialization_seed"])
    torch.manual_seed(initialization_seed)
    model = MultiTargetFCNN(x_train.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=180, min_lr=1.0e-5
    )
    criterion = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_prediction = model(x_train)
        train_loss_tensor = criterion(train_prediction, y_train)
        train_loss_tensor.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                criterion(model(x_validation), y_validation).item()
            )
        scheduler.step(validation_loss)

        improved = validation_loss < best_validation_loss - 1.0e-7
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss_tensor.item()),
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "improved": int(improved),
                "best_validation_loss_so_far": best_validation_loss,
            }
        )
        if epochs_without_improvement >= args.patience:
            break

    expected = metrics["selection"]["chosen"]
    if best_epoch != int(expected["best_epoch"]) or epoch != int(expected["epochs_run"]):
        raise RuntimeError(
            f"Training history mismatch: got best/total {best_epoch}/{epoch}, "
            f"expected {expected['best_epoch']}/{expected['epochs_run']}"
        )
    if abs(best_validation_loss - float(expected["validation_loss"])) > 1.0e-8:
        raise RuntimeError(
            f"Best validation loss mismatch: {best_validation_loss} vs {expected['validation_loss']}"
        )

    max_parameter_difference = max(
        float((best_state[name] - checkpoint["model_state_dict"][name]).abs().max().item())
        for name in best_state
    )
    if max_parameter_difference > 1.0e-7:
        raise RuntimeError(
            f"Reproduced weights differ from saved model by {max_parameter_difference}"
        )

    history_path = artifact_dir / "training_history.csv"
    plot_path = artifact_dir / "training_loss_curve.png"
    save_history(history, history_path)
    save_plot(history, best_epoch, best_validation_loss, plot_path)
    print(f"Rows written: {len(history)}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.12f}")
    print(f"Max saved-weight difference: {max_parameter_difference:.3e}")
    print(f"History: {history_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
