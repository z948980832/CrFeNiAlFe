#!/usr/bin/env python3
"""Create the deterministic SPXY train/validation/test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor


def _standardization(values: Tensor) -> Tuple[Tensor, Tensor]:
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    std = torch.where(std < 1.0e-12, torch.ones_like(std), std)
    return mean, std


def _kennard_stone(distance_matrix: Tensor, select_count: int) -> Tuple[Tensor, Tensor]:
    row_count = distance_matrix.shape[0]
    if not 1 < select_count < row_count:
        raise ValueError("Kennard-Stone select_count must be between 2 and n-1")
    flat_index = int(torch.argmax(distance_matrix).item())
    first = flat_index // row_count
    second = flat_index % row_count
    selected = [first, second]
    available = torch.ones(row_count, dtype=torch.bool)
    available[first] = False
    available[second] = False
    nearest = torch.minimum(distance_matrix[:, first], distance_matrix[:, second])
    while len(selected) < select_count:
        candidates = torch.where(available, nearest, torch.full_like(nearest, -1.0))
        next_index = int(torch.argmax(candidates).item())
        selected.append(next_index)
        available[next_index] = False
        nearest = torch.minimum(nearest, distance_matrix[:, next_index])
    return torch.tensor(selected, dtype=torch.long), torch.nonzero(available).flatten()


def spxy_split_indices(
    features: Tensor,
    targets: Tensor,
    test_fraction: float,
    validation_fraction: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return deterministic SPXY train, validation, and test row indices."""
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test-fraction must be between 0 and 0.5")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must both be two-dimensional")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must contain the same number of rows")

    row_count = features.shape[0]
    test_count = max(1, round(row_count * test_fraction))
    development_count = row_count - test_count
    validation_count = max(1, round(development_count * validation_fraction))
    train_count = development_count - validation_count

    x_mean, x_std = _standardization(features)
    y_mean, y_std = _standardization(targets)
    x_scaled = (features - x_mean) / x_std
    y_scaled = (targets - y_mean) / y_std
    x_distance = torch.cdist(x_scaled, x_scaled)
    y_distance = torch.cdist(y_scaled, y_scaled)
    distance = x_distance / x_distance.max() + y_distance / y_distance.max()

    development_indices, test_indices = _kennard_stone(distance, development_count)
    development_distance = distance[development_indices][:, development_indices]
    local_train, local_validation = _kennard_stone(development_distance, train_count)
    return (
        development_indices[local_train],
        development_indices[local_validation],
        test_indices,
    )


def save_split_indices(
    path: Path,
    train_indices: Tensor,
    validation_indices: Tensor,
    test_indices: Tensor,
) -> None:
    payload = {
        "index_definition": "zero-based data row index; CSV header is excluded",
        "source_row_definition": "one-based CSV line number; header is line 1",
        "train_indices": [int(value) for value in train_indices.tolist()],
        "validation_indices": [int(value) for value in validation_indices.tolist()],
        "test_indices": [int(value) for value in test_indices.tolist()],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_spxy_columns(path: Path) -> Tuple[Tensor, Tensor]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3 or len(rows[0]) < 20:
        raise ValueError("Expected a header plus A:R inputs and final UTS/EL targets.")

    feature_rows: List[List[float]] = []
    target_rows: List[List[float]] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) < 20:
            raise ValueError(f"Line {line_number}: expected at least 20 columns")
        try:
            features = [float(value) for value in row[:18]]
            targets = [float(value) for value in row[-2:]]
        except ValueError as exc:
            raise ValueError(f"Line {line_number} contains a non-numeric value") from exc
        if not all(math.isfinite(value) for value in features + targets):
            raise ValueError(f"Line {line_number} contains a non-finite value")
        feature_rows.append(features)
        target_rows.append(targets)
    return torch.tensor(feature_rows), torch.tensor(target_rows)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir.parent / "0_raw_data" / "parameter.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "split_indices.json",
    )
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, targets = read_spxy_columns(args.data.resolve())
    train_indices, validation_indices, test_indices = spxy_split_indices(
        features,
        targets,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
    )
    output_path = args.output.resolve()
    save_split_indices(output_path, train_indices, validation_indices, test_indices)
    print(f"Train rows: {train_indices.numel()}")
    print(f"Validation rows: {validation_indices.numel()}")
    print(f"Test rows: {test_indices.numel()}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
