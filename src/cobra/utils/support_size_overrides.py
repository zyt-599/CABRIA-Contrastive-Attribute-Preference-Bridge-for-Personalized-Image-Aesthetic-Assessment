from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def _recursive_update(target: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _recursive_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def apply_support_size_overrides(config: dict[str, Any], support_size: int) -> dict[str, Any]:
    overrides = config.get("support_size_overrides", {})
    if not isinstance(overrides, Mapping):
        return config
    override = overrides.get(str(int(support_size)), overrides.get(int(support_size)))
    if isinstance(override, Mapping):
        _recursive_update(config, override)
    return config
