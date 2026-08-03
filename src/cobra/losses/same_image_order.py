from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


def same_image_user_order_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    image_ids: list[str],
    user_ids: list[str],
) -> torch.Tensor:
    if predictions.numel() == 0:
        return predictions.new_zeros(())

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, image_id in enumerate(image_ids):
        grouped_indices[str(image_id)].append(index)

    losses = []
    for _, indices in grouped_indices.items():
        if len(indices) < 2:
            continue
        for left_offset in range(len(indices)):
            for right_offset in range(left_offset + 1, len(indices)):
                left = indices[left_offset]
                right = indices[right_offset]
                if user_ids[left] == user_ids[right]:
                    continue
                truth_diff = targets[left] - targets[right]
                if torch.isclose(truth_diff, truth_diff.new_zeros(())):
                    continue
                sign = torch.sign(truth_diff)
                pred_diff = predictions[left] - predictions[right]
                losses.append(F.softplus(-(sign * pred_diff)))

    if not losses:
        return predictions.new_zeros(())
    return torch.stack(losses).mean()
