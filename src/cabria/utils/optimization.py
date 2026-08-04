from __future__ import annotations

import math
from dataclasses import dataclass

import torch


class EpochLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, total_epochs: int, config: dict | None = None) -> None:
        self.optimizer = optimizer
        self.total_epochs = max(int(total_epochs), 1)
        self.config = config or {}
        self.name = str(self.config.get("name", "none")).lower()
        self.warmup_epochs = max(int(self.config.get("warmup_epochs", 0)), 0)
        self.min_lr_ratio = float(self.config.get("min_lr_ratio", 0.0))
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    @property
    def enabled(self) -> bool:
        return self.name == "cosine"

    def step(self, epoch: int) -> float:
        factor = self._lr_factor(epoch)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor
        return self.optimizer.param_groups[0]["lr"]

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "warmup_epochs": self.warmup_epochs,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
            "total_epochs": self.total_epochs,
            "current_lr": self.current_lr(),
        }

    def _lr_factor(self, epoch: int) -> float:
        if not self.enabled:
            return 1.0
        if self.warmup_epochs > 1 and epoch <= self.warmup_epochs:
            return max(epoch / self.warmup_epochs, 1e-8)
        if self.total_epochs <= self.warmup_epochs:
            return 1.0
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


@dataclass
class EarlyStopping:
    enabled: bool
    patience: int
    min_delta: float = 0.0
    best_metric: float = float("-inf")
    bad_epochs: int = 0
    should_stop: bool = False

    @classmethod
    def from_config(cls, config: dict | None) -> "EarlyStopping":
        payload = config or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            patience=max(int(payload.get("patience", 0)), 0),
            min_delta=float(payload.get("min_delta", 0.0)),
        )

    def update(self, metric: float) -> bool:
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        if self.enabled and self.patience > 0 and self.bad_epochs >= self.patience:
            self.should_stop = True
        return False
