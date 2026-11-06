#!/usr/bin/env python3
"""Fully connected multi-target model used to predict UTS and EL."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    """Two fully connected layers with a residual connection and SiLU."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.activation = nn.SiLU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(values + self.layers(values))


class MultiTargetFCNN(nn.Module):
    """Shared fully connected trunk with one regression head per target."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, 96), nn.SiLU())
        self.trunk = nn.Sequential(ResidualBlock(96), ResidualBlock(96))
        self.bottleneck = nn.Sequential(nn.Linear(96, 48), nn.SiLU())
        self.uts_head = nn.Sequential(nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 1))
        self.el_head = nn.Sequential(nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 1))

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.bottleneck(self.trunk(self.input_layer(values)))
        return torch.cat((self.uts_head(hidden), self.el_head(hidden)), dim=1)
