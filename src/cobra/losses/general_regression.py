from __future__ import annotations

from torch import nn


class GeneralRegressionLoss(nn.Module):
    def __init__(self, loss_name: str = "smooth_l1") -> None:
        super().__init__()
        if loss_name == "mse":
            self.loss = nn.MSELoss()
        elif loss_name == "smooth_l1":
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported regression loss: {loss_name}")

    def forward(self, prediction, target):
        return self.loss(prediction, target)
