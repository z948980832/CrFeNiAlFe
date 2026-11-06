#!/usr/bin/env python3
"""Export feature matrices and plot raw A:R feature correlations with UTS/EL."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
from torch import Tensor

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_TRAINING_DIR = SCRIPT_DIR.parent / "2_2_model_training"
if str(MODEL_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAINING_DIR))

from feature_engineering import engineer_features
from train_fcnn import read_dataset


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=script_dir.parent.parent / "0_raw_data" / "parameter.csv",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=script_dir.parent / "2_2_model_training"
    )
    parser.add_argument(
        "--spxy-dir", type=Path, default=script_dir.parent.parent / "1_SPXY"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir
    )
    parser.add_argument(
        "--rows",
        choices=("all", "train"),
        default="all",
        help="Use all 248 rows for descriptive analysis or only the 190 training rows.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate only the circular heatmap without rewriting feature CSV files.",
    )
    return parser.parse_args()


def pearson_correlations(features: Tensor, targets: Tensor) -> Tensor:
    x_centered = features - features.mean(dim=0)
    y_centered = targets - targets.mean(dim=0)
    denominator = torch.sqrt(
        (x_centered**2).sum(dim=0, keepdim=True).T
        * (y_centered**2).sum(dim=0, keepdim=True)
    )
    return (x_centered.T @ y_centered) / denominator.clamp_min(1.0e-12)


def feature_group(position: int) -> str:
    if position < 18:
        return "raw"
    if position < 28:
        return "presence"
    if position < 31:
        return "composition_summary"
    return "selected_interaction"


def write_feature_matrix(path: Path, names: List[str], values: Tensor) -> None:
    """Write one row per source sample and one column per engineered feature."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(values.tolist())


def compact_label(name: str) -> str:
    replacements = {
        "Mean atom radius": "Mean radius",
        "Atomic size difference": "Size diff.",
        "Electronegativity": "Electroneg.",
        "Standard deviation of electronegativity": "Electroneg. SD",
        "Average VEC": "Mean VEC",
        "Standard deviation of VEC": "VEC SD",
        "Mixing entropy": "Mix entropy",
        "Mixing enthalpy": "Mix enthalpy",
        "component_count": "Component count",
        "composition_max": "Composition max",
        "composition_std": "Composition SD",
    }
    if name.startswith("present_"):
        return f"present {name.removeprefix('present_')}"
    return replacements.get(name, name)


def draw_circular_panel(axis, values: List[float], labels: List[str], target: str) -> None:
    count = len(labels)
    step = 2.0 * math.pi / count
    angles = [position * step for position in range(count)]
    radii = [abs(value) for value in values]
    colors = ["#DC2626" if value >= 0.0 else "#2563EB" for value in values]

    axis.set_theta_offset(math.pi / 2.0)
    axis.set_theta_direction(-1)
    axis.bar(
        angles,
        radii,
        width=step * 0.82,
        bottom=0.0,
        color=colors,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.9,
        zorder=3,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([0.25, 0.50, 0.75, 1.00])
    axis.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8.5, color="#4B5563")
    axis.set_rlabel_position(8)
    axis.set_xticks(angles)
    axis.set_xticklabels([])
    axis.grid(color="#CBD5E1", linewidth=0.65, alpha=0.85)
    axis.spines["polar"].set_color("#94A3B8")
    axis.spines["polar"].set_linewidth(0.8)
    axis.set_facecolor("#F8FAFC")

    for angle, label, value, color in zip(angles, labels, values, colors):
        angle_degrees = math.degrees(angle)
        rotation = 90.0 - angle_degrees
        horizontal_alignment = "left"
        if 90.0 < angle_degrees < 270.0:
            rotation += 180.0
            horizontal_alignment = "right"
        axis.text(
            angle,
            1.075,
            compact_label(label),
            rotation=rotation,
            rotation_mode="anchor",
            ha=horizontal_alignment,
            va="center",
            fontsize=9.2,
            weight="medium",
            color="#111827",
            clip_on=False,
        )
        axis.text(
            angle,
            0.91,
            f"{value:+.2f}",
            ha="center",
            va="center",
            fontsize=7.4,
            weight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
            zorder=6,
        )

    axis.text(
        0.0,
        0.0,
        target,
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color="#111827",
        bbox={"boxstyle": "circle,pad=0.55", "facecolor": "white", "edgecolor": "#CBD5E1"},
        zorder=5,
    )


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    spxy_dir = args.spxy_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(model_dir / "fcnn_model.pt", map_location="cpu")
    input_names, _, raw_features, targets = read_dataset(args.data.resolve())
    engineered, engineered_names = engineer_features(raw_features, input_names)
    selected_indices = checkpoint["selected_feature_indices"].to(dtype=torch.long)
    selected_names = list(checkpoint["selected_feature_names"])
    all_selected_features = engineered[:, selected_indices]

    engineered_csv_path = output_dir / "engineered_features_86.csv"
    selected_csv_path = output_dir / "selected_features_55.csv"
    if not args.plot_only:
        write_feature_matrix(engineered_csv_path, engineered_names, engineered)
        write_feature_matrix(selected_csv_path, selected_names, all_selected_features)

    selected_features = all_selected_features

    if args.rows == "train":
        with (spxy_dir / "split_indices.json").open("r", encoding="utf-8") as handle:
            split = json.load(handle)
        row_indices = torch.tensor(split["train_indices"], dtype=torch.long)
        selected_features = selected_features[row_indices]
        targets = targets[row_indices]

    correlations = pearson_correlations(selected_features, targets)
    csv_path = output_dir / "feature_target_correlations.csv"
    if not args.plot_only:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "feature_order",
                    "feature_group",
                    "feature",
                    "correlation_UTS",
                    "correlation_EL",
                    "absolute_max_correlation",
                    "row_scope",
                    "row_count",
                ]
            )
            for position, name in enumerate(selected_names):
                uts = float(correlations[position, 0].item())
                el = float(correlations[position, 1].item())
                writer.writerow(
                    [
                        position + 1,
                        feature_group(position),
                        name,
                        uts,
                        el,
                        max(abs(uts), abs(el)),
                        args.rows,
                        selected_features.shape[0],
                    ]
                )

    raw_feature_count = len(input_names)
    raw_names = input_names
    raw_correlations = correlations[:raw_feature_count]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18.5, 10.2),
        subplot_kw={"projection": "polar"},
    )
    for target_index, (axis, target) in enumerate(zip(axes, ("UTS", "EL"))):
        values = [
            float(raw_correlations[index, target_index].item())
            for index in range(raw_feature_count)
        ]
        draw_circular_panel(axis, values, raw_names, target)
    scope_text = "all 248 rows" if args.rows == "all" else "190 SPXY training rows"
    fig.suptitle(
        f"Circular correlation map for the original 18 features ({scope_text})",
        fontsize=16,
        weight="bold",
        y=0.975,
    )
    fig.legend(
        handles=[
            Patch(facecolor="#DC2626", edgecolor="none", label="Positive Pearson r"),
            Patch(facecolor="#2563EB", edgecolor="none", label="Negative Pearson r"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.045),
    )
    fig.text(
        0.5,
        0.025,
        "Radial distance from the center represents |Pearson r|; outer circle = 1.00.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.subplots_adjust(left=0.045, right=0.955, top=0.88, bottom=0.12, wspace=0.31)
    plot_path = output_dir / "feature_target_heatmap.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if not args.plot_only:
        print(f"Engineered feature matrix: {engineered_csv_path}")
        print(f"Selected feature matrix: {selected_csv_path}")
    print(f"Rows used: {selected_features.shape[0]}")
    print(f"Features in correlation CSV: {len(selected_names)}")
    print(f"Original features in circular heatmap: {raw_feature_count}")
    if not args.plot_only:
        print(f"Correlations: {csv_path}")
    print(f"Heatmap: {plot_path}")


if __name__ == "__main__":
    main()
