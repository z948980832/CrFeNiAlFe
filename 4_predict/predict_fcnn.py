#!/usr/bin/env python3
"""Predict UTS/EL for pre.csv and save an enriched copy in 4_predict/."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent / "2_train"
MODEL_TRAINING_DIR = TRAIN_DIR / "2_2_model_training"
FEATURE_ENGINEERING_DIR = TRAIN_DIR / "2_1_feature_engineering"
for dependency_dir in (MODEL_TRAINING_DIR, FEATURE_ENGINEERING_DIR):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

from fcnn_model import MultiTargetFCNN
from feature_engineering import engineer_features


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Predict UTS and EL with the saved FCNN.")
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=script_dir / "pre_with_predictions.csv",
        help="CSV containing the 18 A:R features (default: pre_with_predictions.csv).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_TRAINING_DIR / "fcnn_model.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "pre_with_predictions.csv",
    )
    return parser.parse_args()


def read_features(
    path: Path, expected_names: List[str]
) -> Tuple[List[str], List[List[str]], torch.Tensor]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    header = [value.strip() for value in rows[0]]
    if header[:18] != expected_names:
        raise ValueError(
            "The first 18 columns do not match the training columns.\n"
            f"Expected: {expected_names}\nFound: {header[:18]}"
        )
    values: List[List[float]] = []
    data_rows: List[List[str]] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(
                f"Line {line_number}: expected {len(header)} columns, found {len(row)}"
            )
        try:
            feature_row = [float(value) for value in row[:18]]
        except ValueError as exc:
            raise ValueError(f"Line {line_number} contains a non-numeric feature") from exc
        if not all(math.isfinite(value) for value in feature_row):
            raise ValueError(f"Line {line_number} contains a non-finite feature")
        values.append(feature_row)
        data_rows.append(row)
    if not values:
        raise ValueError("The input CSV has no data rows")
    return header, data_rows, torch.tensor(values)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model.resolve(), map_location="cpu")
    raw_input_names = list(checkpoint["raw_input_names"])
    header, source_rows, raw_features = read_features(
        args.input_csv.resolve(), raw_input_names
    )

    engineered, _ = engineer_features(raw_features, raw_input_names)
    selected_indices = checkpoint["selected_feature_indices"].long()
    features = engineered[:, selected_indices]
    features = (features - checkpoint["input_mean"]) / checkpoint["input_std"]

    model = MultiTargetFCNN(features.shape[1])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(features)
        predictions = (
            prediction_scaled * checkpoint["target_std"] + checkpoint["target_mean"]
        )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obsolete_prediction_columns = {"pre_UTS", "pre_EL", "pre_UTS*EL"}
    retained_indices = [
        index
        for index, name in enumerate(header)
        if name not in obsolete_prediction_columns
    ]
    output_header = [header[index] for index in retained_indices] + ["pre_UTS", "pre_EL"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(output_header)
        for source_row, prediction in zip(source_rows, predictions):
            writer.writerow(
                [source_row[index] for index in retained_indices]
                + [float(prediction[0].item()), float(prediction[1].item())]
            )
    print(f"Predicted {len(predictions)} rows")
    print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    main()
