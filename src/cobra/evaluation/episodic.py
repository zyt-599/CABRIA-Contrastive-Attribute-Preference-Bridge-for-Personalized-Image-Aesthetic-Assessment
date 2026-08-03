"""Episodic few-shot eval aligned with prior PIAA few-shot practice.

Each user contributes one episode when ``n_images > support_size``:
sample ``support_size`` images for support, use **all remaining** images as query.
No fixed query holdout and no ``min_query + max_support`` user gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cobra.data.personalized_dataset import EpisodeSpec

DEFAULT_EPISODIC_SEED = 42
DEFAULT_EPISODIC_REPEATS = 10
def episodic_user_seed(base_seed: int, repeat_index: int, user_id: str) -> int:
    """Match the fixed held-out protocol support/query split seed."""
    return int(base_seed) + int(repeat_index) * 100000 + sum(ord(ch) for ch in str(user_id))


def load_test_user_ids_from_csv(path: Path) -> list[str]:
    users: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if not token or token.lower() in {"id", "worker", "user_id"}:
                continue
            users.append(token)
    return sorted({str(user) for user in users})


def resolve_episodic_test_users(
    frame,
    *,
    user_source: str,
    split_users: list[str],
    test_users_file: Path | None = None,
) -> list[str]:
    source = str(user_source).lower()
    if source in {"user_heldout", "split", "manifest"}:
        requested = [str(user_id) for user_id in split_users]
    elif source in {"csv", "user_csv"}:
        if test_users_file is None:
            raise ValueError("episodic_test_users_file is required when user_source='csv'.")
        if not test_users_file.exists():
            raise FileNotFoundError(f"Episodic test users file not found: {test_users_file}")
        requested = load_test_user_ids_from_csv(test_users_file)
    else:
        raise ValueError(f"Unknown episodic user_source={user_source!r} (use: user_heldout|split|csv)")

    present = set(frame["user_id"].astype(str).tolist())
    return [user_id for user_id in requested if user_id in present]


def filter_episodic_eligible_users(frame, users: list[str], support_size: int) -> list[str]:
    """Keep users with strictly more images than ``support_size`` (held-out protocol gate)."""
    support_n = int(support_size)
    eligible: list[str] = []
    for user_id in users:
        count = int((frame["user_id"].astype(str) == str(user_id)).sum())
        if count > support_n:
            eligible.append(str(user_id))
    return eligible


def build_episodic_episodes(
    frame,
    users: list[str],
    support_size: int,
    *,
    seed: int = DEFAULT_EPISODIC_SEED,
    repeat_index: int = 0,
) -> list[EpisodeSpec]:
    """Build one episode per eligible user (user-heldout support/query split)."""
    support_n = int(support_size)
    episodes: list[EpisodeSpec] = []
    for user_id in sorted(users):
        indices = frame.index[frame["user_id"].astype(str) == str(user_id)].to_numpy()
        n = int(len(indices))
        if n <= support_n:
            continue
        user_seed = episodic_user_seed(seed, repeat_index, user_id)
        rng = np.random.default_rng(user_seed)
        perm = indices[rng.permutation(n)]
        support_indices = [int(x) for x in perm[:support_n].tolist()]
        query_indices = [int(x) for x in perm[support_n:].tolist()]
        if not query_indices:
            continue
        episodes.append(
            EpisodeSpec(
                user_id=str(user_id),
                support_size=support_n,
                support_indices=support_indices,
                query_indices=query_indices,
            )
        )
    return episodes


def build_episodic_episodes_common_query(
    frame,
    users: list[str],
    support_size: int,
    *,
    query_support_size: int,
    seed: int = DEFAULT_EPISODIC_SEED,
    repeat_index: int = 0,
) -> list[EpisodeSpec]:
    """Build nested support episodes with the same query set across shots.

    The permutation matches :func:`build_episodic_episodes`. Query images are
    always those after ``query_support_size`` support images; smaller support
    sizes use a prefix of the same support pool.
    """
    support_n = int(support_size)
    query_support_n = int(query_support_size)
    if support_n > query_support_n:
        raise ValueError(
            f"support_size={support_n} cannot exceed query_support_size={query_support_n}."
        )
    episodes: list[EpisodeSpec] = []
    for user_id in sorted(users):
        indices = frame.index[frame["user_id"].astype(str) == str(user_id)].to_numpy()
        n = int(len(indices))
        if n <= query_support_n:
            continue
        user_seed = episodic_user_seed(seed, repeat_index, user_id)
        rng = np.random.default_rng(user_seed)
        perm = indices[rng.permutation(n)]
        support_indices = [int(x) for x in perm[:support_n].tolist()]
        query_indices = [int(x) for x in perm[query_support_n:].tolist()]
        if not query_indices:
            continue
        episodes.append(
            EpisodeSpec(
                user_id=str(user_id),
                support_size=support_n,
                support_indices=support_indices,
                query_indices=query_indices,
            )
        )
    return episodes


def resolve_episodic_settings(eval_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    ev = dict(eval_cfg or {})
    raw_file = ev.get("episodic_test_users_file")
    test_users_file = Path(str(raw_file)) if raw_file else None
    return {
        "user_source": str(ev.get("episodic_user_source", "user_heldout")).lower(),
        "seed": int(ev.get("episodic_seed", DEFAULT_EPISODIC_SEED)),
        "repeats": max(1, int(ev.get("episodic_repeats", DEFAULT_EPISODIC_REPEATS))),
        "test_users_file": test_users_file,
    }


def resolve_episodic_val_settings(eval_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Settings for Stage2 training val / checkpoint selection under episodic protocol."""
    ev = dict(eval_cfg or {})
    raw_file = ev.get("episodic_val_test_users_file") or ev.get("episodic_test_users_file")
    test_users_file = Path(str(raw_file)) if raw_file else None
    default_repeats = int(ev.get("selection_repeats", 3))
    return {
        "user_source": str(ev.get("episodic_val_user_source", "split")).lower(),
        "seed": int(ev.get("episodic_val_seed", ev.get("episodic_seed", DEFAULT_EPISODIC_SEED))),
        "repeats": max(1, int(ev.get("episodic_val_repeats", default_repeats))),
        "test_users_file": test_users_file,
    }
