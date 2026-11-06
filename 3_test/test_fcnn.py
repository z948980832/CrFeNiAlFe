#!/usr/bin/env python3
"""Run the saved FCNN on the held-out SPXY test rows only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_DIR = SCRIPT_DIR.parent / "2_train" / "2_2_model_training"
FEATURE_ENGINEERING_DIR = SCRIPT_DIR.parent / "2_train" / "2_1_feature_engineering"
for dependency_dir in (MODEL_TRAINING_DIR, FEATURE_ENGINEERING_DIR):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from export_spxy_test_results import export_full_results
from fcnn_model import MultiTargetFCNN
from feature_engineering import engineer_features
from train_fcnn import read_dataset, regression_metrics, write_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=SCRIPT_DIR.parent / "0_raw_data" / "parameter.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_TRAINING_DIR / "fcnn_model.pt",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=SCRIPT_DIR.parent / "1_SPXY" / "split_indices.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=SCRIPT_DIR / "test_predictions.csv",
    )
    parser.add_argument(
        "--full-results",
        type=Path,
        default=SCRIPT_DIR / "spxy_test_results_full.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.model.resolve(), map_location="cpu")
    input_names, _, raw_features, targets = read_dataset(args.data.resolve())
    if input_names != list(checkpoint["raw_input_names"]):
        raise ValueError("parameter.csv A:R columns do not match the saved model")

    with args.split.resolve().open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    test_indices = torch.tensor(split["test_indices"], dtype=torch.long)

    engineered, _ = engineer_features(raw_features, input_names)
    selected_indices = checkpoint["selected_feature_indices"].long()
    features = engineered[:, selected_indices]
    test_features = features[test_indices]
    test_features = (
        test_features - checkpoint["input_mean"]
    ) / checkpoint["input_std"]

    model = MultiTargetFCNN(test_features.shape[1])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        scaled = model(test_features)
        predicted = scaled * checkpoint["target_std"] + checkpoint["target_mean"]

    actual = targets[test_indices]
    predictions_path = args.predictions.resolve()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(
        predictions_path,
        [int(value) for value in test_indices.tolist()],
        actual,
        predicted,
    )
    full_results_path = args.full_results.resolve()
    export_full_results(
        args.data.resolve(),
        args.split.resolve(),
        predictions_path,
        full_results_path,
    )

    metrics = regression_metrics(actual, predicted)
    print(f"SPXY test rows: {len(test_indices)}")
    for target in ("UTS", "EL"):
        item = metrics[target]
        print(
            f"{target}: R2={item['r2']:.6f}, RMSE={item['rmse']:.6f}, "
            f"MAE={item['mae']:.6f}"
        )
    print(f"Saved compact results: {predictions_path}")
    print(f"Saved full results: {full_results_path}")


if __name__ == "__main__":
    main()
