#!/usr/bin/env python3
"""Combine SPXY test predictions with the original A:R feature rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


INPUT_COUNT = 18


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir.parent / "0_raw_data" / "parameter.csv",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=script_dir.parent / "1_SPXY" / "split_indices.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=script_dir / "test_predictions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "spxy_test_results_full.csv",
    )
    return parser.parse_args()


def read_rows(path: Path) -> tuple[List[str], List[List[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows[0], rows[1:]


def read_predictions(path: Path) -> Dict[int, Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "source_row",
            "UTS_actual",
            "UTS_predicted",
            "UTS_error",
            "EL_actual",
            "EL_predicted",
            "EL_error",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Prediction CSV is missing required columns: {path}")
        predictions = {int(row["source_row"]): row for row in reader}
    return predictions


def export_full_results(
    data_path: Path,
    split_path: Path,
    predictions_path: Path,
    output_path: Path,
) -> None:
    header, rows = read_rows(data_path)
    if len(header) < INPUT_COUNT + 2:
        raise ValueError("parameter.csv must contain A:R features and two targets")
    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    test_indices = [int(value) for value in split["test_indices"]]
    predictions = read_predictions(predictions_path)
    if len(predictions) != len(test_indices):
        raise ValueError(
            f"Expected {len(test_indices)} prediction rows, found {len(predictions)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_header = (
        ["dataset_index", "source_row"]
        + header[:INPUT_COUNT]
        + [
            "UTS_actual",
            "UTS_predicted",
            "UTS_error",
            "UTS_absolute_error",
            "EL_actual",
            "EL_predicted",
            "EL_error",
            "EL_absolute_error",
        ]
    )
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(output_header)
        for dataset_index in test_indices:
            source_row = dataset_index + 2
            if source_row not in predictions:
                raise ValueError(f"Missing prediction for CSV source row {source_row}")
            prediction = predictions[source_row]
            raw_row = rows[dataset_index]
            uts_actual = float(raw_row[-2])
            el_actual = float(raw_row[-1])
            if abs(float(prediction["UTS_actual"]) - uts_actual) > 1.0e-4:
                raise ValueError(f"UTS actual value mismatch at source row {source_row}")
            if abs(float(prediction["EL_actual"]) - el_actual) > 1.0e-4:
                raise ValueError(f"EL actual value mismatch at source row {source_row}")
            uts_predicted = float(prediction["UTS_predicted"])
            el_predicted = float(prediction["EL_predicted"])
            uts_error = uts_predicted - uts_actual
            el_error = el_predicted - el_actual
            writer.writerow(
                [dataset_index, source_row]
                + raw_row[:INPUT_COUNT]
                + [
                    uts_actual,
                    uts_predicted,
                    uts_error,
                    abs(uts_error),
                    el_actual,
                    el_predicted,
                    el_error,
                    abs(el_error),
                ]
            )


def main() -> None:
    args = parse_args()
    export_full_results(
        args.data.resolve(),
        args.split.resolve(),
        args.predictions.resolve(),
        args.output.resolve(),
    )
    print(f"Saved full SPXY test results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
