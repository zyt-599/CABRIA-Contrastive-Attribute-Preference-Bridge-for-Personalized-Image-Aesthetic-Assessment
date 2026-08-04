from __future__ import annotations

import json
import os
import random
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(resolve_path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_json(data: dict[str, Any], path: str | os.PathLike[str]) -> None:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(resolve_path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path: str | os.PathLike[str]) -> Path:
    if isinstance(path, Path):
        return path
    text = str(path)
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 6:
        parts = PurePosixPath(text).parts
        drive = parts[2].upper() + ":"
        suffix = Path(*parts[3:]) if len(parts) > 3 else Path()
        return Path(f"{drive}\\") / suffix
    return Path(text)


def to_linux_path(path: str | os.PathLike[str]) -> str:
    resolved = Path(path)
    if os.name == "nt" and resolved.drive:
        drive = resolved.drive[0].lower()
        rest = resolved.as_posix().split(":", maxsplit=1)[1].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return resolved.as_posix()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
