from __future__ import annotations

import torch
from torch import nn


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
        if predictions.numel() < 2:
            return predictions.new_zeros(())

        target_diffs = targets.unsqueeze(1) - targets.unsqueeze(0)
        pred_diffs = predictions.unsqueeze(1) - predictions.unsqueeze(0)

        valid_pairs = target_diffs.abs() > self.margin
        upper_tri = torch.triu(torch.ones_like(valid_pairs, dtype=torch.bool), diagonal=1)
        valid_pairs = valid_pairs & upper_tri
        if not torch.any(valid_pairs):
            return predictions.new_zeros(())

        pair_labels = torch.sign(target_diffs[valid_pairs])
        pair_scores = pred_diffs[valid_pairs]
        return torch.nn.functional.softplus(-pair_labels * pair_scores).mean()
