from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def build_score_bin_centers(
    score_values: Iterable[float] | None = None,
    score_min: float | None = None,
    score_max: float | None = None,
    num_bins: int | None = None,
) -> torch.Tensor:
    if score_values is not None:
        values = sorted({float(value) for value in score_values})
        if not values:
            raise ValueError("score_values must not be empty")
        return torch.tensor(values, dtype=torch.float32)
    if score_min is None or score_max is None:
        raise ValueError("score_min and score_max are required when score_values is not provided")
    if num_bins is None or int(num_bins) < 2:
        raise ValueError("num_bins must be at least 2 when score_values is not provided")
    return torch.linspace(float(score_min), float(score_max), steps=int(num_bins), dtype=torch.float32)


class DistributionAlignmentLoss(nn.Module):
    def __init__(
        self,
        *,
        mode: str = "softmax",
        temperature: float = 1.0,
        score_values: Iterable[float] | None = None,
        score_min: float | None = None,
        score_max: float | None = None,
        num_bins: int | None = None,
        sigma: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if mode not in {"softmax", "histogram"}:
            raise ValueError(f"Unsupported distribution alignment mode: {mode}")

        self.mode = mode
        self.temperature = float(temperature)
        self.sigma = float(sigma)
        self.eps = float(eps)
        if self.mode == "histogram":
            centers = build_score_bin_centers(
                score_values=score_values,
                score_min=score_min,
                score_max=score_max,
                num_bins=num_bins,
            )
        else:
            centers = torch.empty(0, dtype=torch.float32)
        self.register_buffer("bin_centers", centers)

    def forward(self, prediction_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        if self.mode == "softmax":
            prediction_distribution = F.softmax(prediction_scores.reshape(-1) / self.temperature, dim=0)
            target_distribution = F.softmax(target_scores.reshape(-1) / self.temperature, dim=0).clamp_min(self.eps)
        else:
            prediction_distribution = self._soft_histogram(prediction_scores)
            target_distribution = self._soft_histogram(target_scores).clamp_min(self.eps)
        return F.kl_div(prediction_distribution.clamp_min(self.eps).log(), target_distribution, reduction="sum")

    def describe_bins(self) -> list[float]:
        return [float(value) for value in self.bin_centers.detach().cpu().tolist()]

    def _soft_histogram(self, scores: torch.Tensor) -> torch.Tensor:
        centers = self.bin_centers.to(device=scores.device, dtype=scores.dtype)
        scores = scores.reshape(-1, 1)
        distances = (scores - centers.unsqueeze(0)) / self.sigma
        weights = torch.exp(-0.5 * distances.square())
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        histogram = weights.mean(dim=0)
        return histogram / histogram.sum().clamp_min(self.eps)
