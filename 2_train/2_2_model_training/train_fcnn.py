#!/usr/bin/env python3
"""Train and evaluate a fully connected network for UTS and EL.

The script intentionally depends only on PyTorch and Python's standard library.
It reads Excel columns A:R (zero-based columns 0:18) as inputs and the final
two columns as targets.  The test set is held out before all model selection;
normalization and early stopping use training data only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_ENGINEERING_DIR = SCRIPT_DIR.parent / "2_1_feature_engineering"
SPXY_DIR = SCRIPT_DIR.parent.parent / "1_SPXY"
for dependency_dir in (FEATURE_ENGINEERING_DIR, SPXY_DIR):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from fcnn_model import MultiTargetFCNN
from feature_engineering import (
    engineer_features,
    select_engineered_features,
    standardization,
)
from spxy_split import save_split_indices, spxy_split_indices


INPUT_START = 0   # Excel column A
INPUT_STOP = 18   # Excel column R, exclusive
TARGET_COUNT = 2
TARGET_NAMES = ("UTS", "EL")
PASS_THRESHOLDS = {"UTS": 0.9137, "EL": 0.8801}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train a multi-output fully connected neural network."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir.parent.parent / "0_raw_data" / "parameter.csv",
        help="CSV dataset path (default: ../../0_raw_data/parameter.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory for model weights, metrics, and training artifacts.",
    )
    parser.add_argument(
        "--spxy-dir",
        dest="spxy_dir",
        type=Path,
        default=script_dir.parent.parent / "1_SPXY",
        help="Directory for the reproducible SPXY split indices.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Data split seed.")
    parser.add_argument(
        "--split-method",
        choices=("random", "spxy"),
        default="spxy",
        help="random shuffle or deterministic SPXY representative split.",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.15, help="Held-out test fraction."
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.10,
        help="Validation fraction of the non-test rows.",
    )
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--interaction-features",
        type=int,
        default=24,
        help="Top training-only composition interaction features to add; 0 disables them.",
    )
    return parser.parse_args()


def read_dataset(path: Path) -> Tuple[List[str], List[str], Tensor, Tensor]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3:
        raise ValueError(f"Dataset must contain a header and at least two rows: {path}")

    header = [item.strip() for item in rows[0]]
    if len(header) < INPUT_STOP + TARGET_COUNT:
        raise ValueError(f"Expected at least 20 columns, found {len(header)}")

    input_names = header[INPUT_START:INPUT_STOP]
    target_names = header[-TARGET_COUNT:]
    if target_names != list(TARGET_NAMES):
        raise ValueError(
            f"Expected final columns {list(TARGET_NAMES)}, found {target_names}"
        )

    feature_rows: List[List[float]] = []
    target_rows: List[List[float]] = []
    expected_columns = len(header)
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected_columns:
            raise ValueError(
                f"Line {line_number}: expected {expected_columns} values, found {len(row)}"
            )
        try:
            features = [float(value) for value in row[INPUT_START:INPUT_STOP]]
            targets = [float(value) for value in row[-TARGET_COUNT:]]
        except ValueError as exc:
            raise ValueError(f"Line {line_number} contains a non-numeric value") from exc
        if not all(math.isfinite(value) for value in features + targets):
            raise ValueError(f"Line {line_number} contains a non-finite value")
        feature_rows.append(features)
        target_rows.append(targets)

    return input_names, target_names, torch.tensor(feature_rows), torch.tensor(target_rows)


def split_indices(
    row_count: int,
    seed: int,
    test_fraction: float,
    validation_fraction: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test-fraction must be between 0 and 0.5")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(row_count, generator=generator)
    test_count = max(1, round(row_count * test_fraction))
    remaining_count = row_count - test_count
    validation_count = max(1, round(remaining_count * validation_fraction))

    test_indices = order[:test_count]
    validation_indices = order[test_count : test_count + validation_count]
    train_indices = order[test_count + validation_count :]
    return train_indices, validation_indices, test_indices


def r2_scores(actual: Tensor, predicted: Tensor) -> Tensor:
    residual = ((actual - predicted) ** 2).sum(dim=0)
    total = ((actual - actual.mean(dim=0)) ** 2).sum(dim=0)
    return 1.0 - residual / total


def regression_metrics(actual: Tensor, predicted: Tensor) -> Dict[str, Dict[str, float]]:
    scores = r2_scores(actual, predicted)
    errors = predicted - actual
    rmse = torch.sqrt((errors ** 2).mean(dim=0))
    mae = errors.abs().mean(dim=0)
    metrics: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(TARGET_NAMES):
        metrics[name] = {
            "r2": float(scores[index].item()),
            "rmse": float(rmse[index].item()),
            "mae": float(mae[index].item()),
            "threshold": PASS_THRESHOLDS[name],
            "passed": bool(scores[index].item() >= PASS_THRESHOLDS[name]),
        }
    return metrics


def train_candidate(
    x_train: Tensor,
    y_train: Tensor,
    x_validation: Tensor,
    y_validation: Tensor,
    initialization_seed: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> Tuple[MultiTargetFCNN, Dict[str, float]]:
    torch.manual_seed(initialization_seed)
    model = MultiTargetFCNN(x_train.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=180, min_lr=1.0e-5
    )
    criterion = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train)
        loss = criterion(prediction, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(x_validation)
            validation_loss = float(criterion(validation_prediction, y_validation).item())
        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss - 1.0e-7:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_prediction = model(x_validation)
        validation_r2 = r2_scores(y_validation, validation_prediction)
        train_prediction = model(x_train)
        train_r2 = r2_scores(y_train, train_prediction)

    summary = {
        "initialization_seed": initialization_seed,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "validation_loss": best_validation_loss,
        "validation_r2_UTS": float(validation_r2[0].item()),
        "validation_r2_EL": float(validation_r2[1].item()),
        "train_r2_UTS": float(train_r2[0].item()),
        "train_r2_EL": float(train_r2[1].item()),
    }
    return model, summary


def write_predictions(
    path: Path,
    indices: Sequence[int],
    actual: Tensor,
    predicted: Tensor,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_row",
                "UTS_actual",
                "UTS_predicted",
                "UTS_error",
                "EL_actual",
                "EL_predicted",
                "EL_error",
            ]
        )
        for position, index in enumerate(indices):
            uts_actual = float(actual[position, 0].item())
            el_actual = float(actual[position, 1].item())
            uts_predicted = float(predicted[position, 0].item())
            el_predicted = float(predicted[position, 1].item())
            writer.writerow(
                [
                    int(index) + 2,
                    uts_actual,
                    uts_predicted,
                    uts_predicted - uts_actual,
                    el_actual,
                    el_predicted,
                    el_predicted - el_actual,
                ]
            )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass

    start_time = time.perf_counter()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spxy_dir = args.spxy_dir.resolve()
    spxy_dir.mkdir(parents=True, exist_ok=True)

    input_names, target_names, features, targets = read_dataset(data_path)
    if args.split_method == "random":
        train_indices, validation_indices, test_indices = split_indices(
            row_count=features.shape[0],
            seed=args.seed,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
        )
    else:
        train_indices, validation_indices, test_indices = spxy_split_indices(
            features=features,
            targets=targets,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
        )

    engineered_features, engineered_names = engineer_features(features, input_names)
    selected_feature_indices = select_engineered_features(
        engineered=engineered_features,
        targets=targets,
        train_indices=train_indices,
        base_feature_count=len(input_names),
        interaction_count=args.interaction_features,
    )
    selected_feature_names = [
        engineered_names[int(index)] for index in selected_feature_indices.tolist()
    ]
    model_features = engineered_features[:, selected_feature_indices]

    x_train_raw = model_features[train_indices]
    y_train_raw = targets[train_indices]
    x_validation_raw = model_features[validation_indices]
    y_validation_raw = targets[validation_indices]
    x_test_raw = model_features[test_indices]
    y_test_raw = targets[test_indices]

    x_mean, x_std = standardization(x_train_raw)
    y_mean, y_std = standardization(y_train_raw)
    x_train = (x_train_raw - x_mean) / x_std
    y_train = (y_train_raw - y_mean) / y_std
    x_validation = (x_validation_raw - x_mean) / x_std
    y_validation = (y_validation_raw - y_mean) / y_std
    x_test = (x_test_raw - x_mean) / x_std

    candidates: List[Dict[str, float]] = []
    best_model: MultiTargetFCNN | None = None
    best_summary: Dict[str, float] | None = None
    for restart in range(args.restarts):
        initialization_seed = args.seed * 1000 + restart
        model, summary = train_candidate(
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            initialization_seed=initialization_seed,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        candidates.append(summary)
        if best_summary is None or summary["validation_loss"] < best_summary["validation_loss"]:
            best_model = model
            best_summary = summary
        print(
            f"restart={restart + 1:02d}/{args.restarts} "
            f"epoch={int(summary['best_epoch']):4d} "
            f"val_R2=(UTS {summary['validation_r2_UTS']:.4f}, "
            f"EL {summary['validation_r2_EL']:.4f})"
        )

    if best_model is None or best_summary is None:
        raise RuntimeError("No model candidate was trained")

    best_model.eval()
    with torch.no_grad():
        test_prediction_scaled = best_model(x_test)
        test_prediction = test_prediction_scaled * y_std + y_mean
        validation_prediction = best_model(x_validation) * y_std + y_mean
        train_prediction = best_model(x_train) * y_std + y_mean

    train_metrics = regression_metrics(y_train_raw, train_prediction)
    validation_metrics = regression_metrics(y_validation_raw, validation_prediction)
    test_metrics = regression_metrics(y_test_raw, test_prediction)
    overall_passed = all(item["passed"] for item in test_metrics.values())

    checkpoint = {
        "model_class": "MultiTargetFCNN",
        "model_state_dict": best_model.state_dict(),
        "raw_input_names": input_names,
        "selected_feature_indices": selected_feature_indices,
        "selected_feature_names": selected_feature_names,
        "target_names": target_names,
        "input_mean": x_mean,
        "input_std": x_std,
        "target_mean": y_mean,
        "target_std": y_std,
        "split_seed": args.seed,
        "initialization_seed": int(best_summary["initialization_seed"]),
        "torch_version": torch.__version__,
    }
    model_path = output_dir / "fcnn_model.pt"
    torch.save(checkpoint, model_path)

    split_path = spxy_dir / "split_indices.json"
    save_split_indices(split_path, train_indices, validation_indices, test_indices)

    metrics_payload = {
        "data_path": str(data_path),
        "row_count": int(features.shape[0]),
        "input_columns": {"excel_range": "A:R", "names": input_names},
        "feature_engineering": {
            "raw_feature_count": len(input_names),
            "candidate_feature_count": len(engineered_names),
            "selected_feature_count": len(selected_feature_names),
            "selected_feature_names": selected_feature_names,
            "interaction_features_selected_on_training_rows": args.interaction_features,
        },
        "target_columns": target_names,
        "split": {
            "method": args.split_method,
            "seed": args.seed,
            "train_rows": int(train_indices.numel()),
            "validation_rows": int(validation_indices.numel()),
            "test_rows": int(test_indices.numel()),
            "test_fraction": args.test_fraction,
            "validation_fraction_of_non_test": args.validation_fraction,
        },
        "selection": {
            "criterion": "lowest validation MSE in standardized target space",
            "restarts": args.restarts,
            "chosen": best_summary,
            "candidates": candidates,
        },
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "overall_passed": overall_passed,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nHeld-out test results")
    for name in TARGET_NAMES:
        item = test_metrics[name]
        status = "PASS" if item["passed"] else "FAIL"
        print(
            f"{name:>3} R2={item['r2']:.4f} "
            f"(threshold {item['threshold']:.4f}) {status}; "
            f"RMSE={item['rmse']:.4f}; MAE={item['mae']:.4f}"
        )
    print(f"Overall: {'PASS' if overall_passed else 'FAIL'}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
