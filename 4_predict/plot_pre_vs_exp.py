#!/usr/bin/env python3
"""Plot SPXY predictions against experimental UTS and EL values."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_LAST_FOUR = ["pre_UTS_new", "pre_EL_new", "exp_UTS", "exp_EL"]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir / "pre_vs_exp.csv",
        help="CSV whose last four columns contain new predictions and experiments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "pre_vs_exp_scatter.png",
    )
    return parser.parse_args()


def read_last_four(path: Path) -> Dict[str, List[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        last_four = [name.strip() for name in reader.fieldnames[-4:]]
        if last_four != EXPECTED_LAST_FOUR:
            raise ValueError(
                "Unexpected last four columns. "
                f"Expected {EXPECTED_LAST_FOUR}, found {last_four}."
            )

        values = {name: [] for name in EXPECTED_LAST_FOUR}
        for line_number, row in enumerate(reader, start=2):
            for name in EXPECTED_LAST_FOUR:
                try:
                    value = float(row[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Line {line_number}, column {name} is not numeric."
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"Line {line_number}, column {name} is not finite."
                    )
                values[name].append(value)

    if not values[EXPECTED_LAST_FOUR[0]]:
        raise ValueError("CSV contains no data rows.")
    return values


def regression_metrics(actual: List[float], predicted: List[float]) -> Tuple[float, float, float]:
    mean_actual = sum(actual) / len(actual)
    residual_sum = sum((p - a) ** 2 for a, p in zip(actual, predicted))
    total_sum = sum((a - mean_actual) ** 2 for a in actual)
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0 else math.nan
    rmse = math.sqrt(residual_sum / len(actual))
    mae = sum(abs(p - a) for a, p in zip(actual, predicted)) / len(actual)
    return r2, rmse, mae


def draw_panel(
    axis,
    actual: List[float],
    predicted: List[float],
    target: str,
    color: str,
) -> Tuple[float, float, float]:
    lower = min(actual + predicted)
    upper = max(actual + predicted)
    span = upper - lower
    margin = 0.07 * span if span > 0 else 1.0
    limits = (lower - margin, upper + margin)

    axis.scatter(
        actual,
        predicted,
        s=52,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
        zorder=3,
    )
    axis.plot(limits, limits, color="#374151", linewidth=1.2, linestyle="--", zorder=2)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(f"Experimental {target}")
    axis.set_ylabel(f"Predicted {target}")
    axis.set_title(f"{target}: predicted vs experimental", loc="left", weight="bold")
    axis.grid(True, color="#E5E7EB", linewidth=0.7)
    axis.set_axisbelow(True)

    r2, rmse, mae = regression_metrics(actual, predicted)
    axis.text(
        0.04,
        0.96,
        f"n = {len(actual)}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
    )
    return r2, rmse, mae


def main() -> None:
    args = parse_args()
    values = read_last_four(args.data.resolve())

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    uts_metrics = draw_panel(
        axes[0], values["exp_UTS"], values["pre_UTS_new"], "UTS", "#2563EB"
    )
    el_metrics = draw_panel(
        axes[1], values["exp_EL"], values["pre_EL_new"], "EL", "#EA580C"
    )
    fig.suptitle("SPXY model predictions and experimental values", fontsize=14, weight="bold")
    fig.text(0.5, 0.015, "Dashed line: ideal prediction (y = x)", ha="center", fontsize=9, color="#4B5563")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Rows plotted: {len(values['exp_UTS'])}")
    print(f"UTS R2={uts_metrics[0]:.6f}, RMSE={uts_metrics[1]:.6f}, MAE={uts_metrics[2]:.6f}")
    print(f"EL  R2={el_metrics[0]:.6f}, RMSE={el_metrics[1]:.6f}, MAE={el_metrics[2]:.6f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
