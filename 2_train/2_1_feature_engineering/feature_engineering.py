#!/usr/bin/env python3
"""Feature engineering and training-only feature selection for the FCNN."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import Tensor


def standardization(values: Tensor) -> Tuple[Tensor, Tensor]:
    """Return column means and population standard deviations.

    Near-constant columns receive a scale of one so later transforms cannot
    divide by zero.
    """

    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    std = torch.where(std < 1.0e-12, torch.ones_like(std), std)
    return mean, std


def engineer_features(
    features: Tensor, input_names: Sequence[str]
) -> Tuple[Tensor, List[str]]:
    """Create the 86 candidate features used by the project.

    The first ten raw columns are elemental compositions.  The returned
    matrix contains 18 raw features, 10 presence flags, three composition
    summaries, and 55 second-order composition terms (including squares).
    No target values are used in this step.
    """

    if features.ndim != 2 or features.shape[1] != 18:
        raise ValueError(f"Expected an N x 18 feature matrix, found {tuple(features.shape)}")
    if len(input_names) != 18:
        raise ValueError(f"Expected 18 input names, found {len(input_names)}")

    composition = features[:, :10]
    presence = (composition.abs() > 1.0e-12).to(features.dtype)
    component_count = presence.sum(dim=1, keepdim=True)
    composition_max = composition.max(dim=1, keepdim=True).values
    composition_std = composition.std(dim=1, unbiased=False, keepdim=True)

    blocks = [features, presence, component_count, composition_max, composition_std]
    names = list(input_names)
    names.extend(f"present_{name}" for name in input_names[:10])
    names.extend(("component_count", "composition_max", "composition_std"))

    interactions: List[Tensor] = []
    for left in range(10):
        for right in range(left, 10):
            interactions.append(composition[:, left] * composition[:, right])
            names.append(f"{input_names[left]}*{input_names[right]}")
    blocks.append(torch.stack(interactions, dim=1))
    return torch.cat(blocks, dim=1), names


def select_engineered_features(
    engineered: Tensor,
    targets: Tensor,
    train_indices: Tensor,
    base_feature_count: int,
    interaction_count: int,
) -> Tensor:
    """Keep mandatory features and select interactions using training rows only.

    Each interaction is scored by the larger absolute Pearson correlation
    with UTS or EL.  The top ``interaction_count`` candidates are retained.
    """

    mandatory_count = base_feature_count + 13
    candidate_count = engineered.shape[1] - mandatory_count
    keep_count = min(max(interaction_count, 0), candidate_count)
    mandatory = torch.arange(mandatory_count, dtype=torch.long)
    if keep_count == 0:
        return mandatory

    x = engineered[train_indices, mandatory_count:]
    y = targets[train_indices]
    x_centered = x - x.mean(dim=0)
    y_centered = y - y.mean(dim=0)
    denominator = torch.sqrt(
        (x_centered**2).sum(dim=0, keepdim=True).T
        * (y_centered**2).sum(dim=0, keepdim=True)
    )
    correlations = (x_centered.T @ y_centered) / denominator.clamp_min(1.0e-12)
    scores = correlations.abs().max(dim=1).values
    selected_candidates = torch.topk(scores, k=keep_count).indices + mandatory_count
    return torch.cat((mandatory, selected_candidates.sort().values))
