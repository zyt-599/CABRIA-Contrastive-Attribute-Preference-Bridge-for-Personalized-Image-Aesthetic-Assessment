from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Tracker:
    enabled: bool = False

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    def log_summary(self, metrics: dict[str, Any]) -> None:
        return None

    def log_table(self, name: str, rows: list[dict[str, Any]]) -> None:
        return None

    def finish(self) -> None:
        return None

    def define_metric(self, *args, **kwargs) -> None:
        return None


def init_tracker(config: dict[str, Any], job_type: str) -> Tracker:
    return Tracker(enabled=False)
