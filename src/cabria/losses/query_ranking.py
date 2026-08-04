from __future__ import annotations

import torch
import torch.nn.functional as F


def query_pairwise_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.5,
    max_pairs: int = 0,
) -> torch.Tensor:
    if predictions.numel() < 2:
        return predictions.new_zeros(())

    pairs: list[tuple[int, int]] = []
    num_items = predictions.numel()
    for left in range(num_items):
        for right in range(left + 1, num_items):
            truth_diff = targets[left] - targets[right]
            if truth_diff.abs() <= margin:
                continue
            pairs.append((left, right))

    if not pairs:
        return predictions.new_zeros(())

    if max_pairs > 0 and len(pairs) > max_pairs:
        device = predictions.device
        perm = torch.randperm(len(pairs), device=device)[: int(max_pairs)]
        pairs = [pairs[int(i)] for i in perm]

    losses = []
    for left, right in pairs:
        truth_diff = targets[left] - targets[right]
        sign = torch.sign(truth_diff)
        pred_diff = predictions[left] - predictions[right]
        losses.append(F.softplus(-(sign * pred_diff)))

    return torch.stack(losses).mean()


def bradley_terry_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.0,
    max_pairs: int = 0,
) -> torch.Tensor:
    """Pairwise logistic (Bradley–Terry) loss: -log σ(sign(Δy) * Δŷ). Sample-efficient ranking surrogate."""
    if predictions.numel() < 2:
        return predictions.new_zeros(())

    pairs: list[tuple[int, int]] = []
    num_items = predictions.numel()
    for left in range(num_items):
        for right in range(left + 1, num_items):
            truth_diff = targets[left] - targets[right]
            if margin > 0.0 and float(truth_diff.abs()) <= margin:
                continue
            if float(truth_diff.abs()) < 1e-9:
                continue
            pairs.append((left, right))

    if not pairs:
        return predictions.new_zeros(())

    if max_pairs > 0 and len(pairs) > max_pairs:
        device = predictions.device
        perm = torch.randperm(len(pairs), device=device)[: int(max_pairs)]
        pairs = [pairs[int(i)] for i in perm]

    losses = []
    for left, right in pairs:
        truth_diff = targets[left] - targets[right]
        pred_diff = predictions[left] - predictions[right]
        if truth_diff > 0:
            losses.append(-F.logsigmoid(pred_diff))
        else:
            losses.append(-F.logsigmoid(-pred_diff))

    return torch.stack(losses).mean()


def pairwise_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    mode: str = "softplus_margin",
    margin: float = 0.5,
    max_pairs: int = 0,
) -> torch.Tensor:
    m = str(mode or "softplus_margin").lower().strip()
    if m in {"bradley_terry", "bt", "logistic"}:
        return bradley_terry_ranking_loss(predictions, targets, margin=margin, max_pairs=max_pairs)
    return query_pairwise_ranking_loss(predictions, targets, margin=margin, max_pairs=max_pairs)
