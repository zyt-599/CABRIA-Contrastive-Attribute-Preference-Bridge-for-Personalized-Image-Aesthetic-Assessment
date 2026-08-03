"""Single source of truth for fixed-query holdout defaults (train val + offline eval)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from cobra.data.personalized_dataset import EpisodeSpec
from cobra.data.user_split import build_fixed_query_episodes, filter_users_by_min_row_count

DEFAULT_MIN_QUERY_IMAGES = 256
DEFAULT_HOLDOUT_SEED = 4242
DEFAULT_SUPPORT_SEED = 1002


@dataclass(frozen=True)
class FixedQuerySettings:
    min_query_images: int
    holdout_seed: int
    support_seed: int
    same_user_pool: bool
    support_pool_size: int | None
    user_pool_support_size: int | None


def resolve_fixed_query_settings(eval_cfg: Mapping[str, Any] | None) -> FixedQuerySettings:
    """Resolve knobs from ``config['evaluation']`` with stable defaults."""
    ev = dict(eval_cfg or {})
    min_q = int(ev.get("fixed_query_min_images", ev.get("min_query_images", DEFAULT_MIN_QUERY_IMAGES)))
    holdout = int(ev.get("fixed_query_holdout_seed", DEFAULT_HOLDOUT_SEED))
    raw_sup = ev.get("fixed_query_support_seed", ev.get("fixed_episode_seed", DEFAULT_SUPPORT_SEED))
    if raw_sup is None or (isinstance(raw_sup, str) and raw_sup.strip().lower() in {"", "null", "none"}):
        support_seed = DEFAULT_SUPPORT_SEED
    else:
        support_seed = int(raw_sup)
    same_pool = bool(ev.get("fixed_query_same_user_pool", True))
    raw_pool = ev.get("fixed_query_support_pool_size", ev.get("support_pool_size"))
    support_pool_size = None
    if raw_pool is not None and not (isinstance(raw_pool, str) and raw_pool.strip().lower() in {"", "null", "none"}):
        support_pool_size = int(raw_pool)
    raw_user_pool = ev.get(
        "fixed_query_user_pool_support_size",
        ev.get("user_pool_support_size", support_pool_size),
    )
    user_pool_support_size = None
    if raw_user_pool is not None and not (
        isinstance(raw_user_pool, str) and raw_user_pool.strip().lower() in {"", "null", "none"}
    ):
        user_pool_support_size = int(raw_user_pool)
    return FixedQuerySettings(
        min_query_images=max(1, min_q),
        holdout_seed=holdout,
        support_seed=support_seed,
        same_user_pool=same_pool,
        support_pool_size=support_pool_size,
        user_pool_support_size=user_pool_support_size,
    )


def build_fixed_query_val_episodes(
    frame,
    val_user_ids: list[str],
    support_size: int,
    *,
    eval_cfg: Mapping[str, Any] | None,
    support_seed: int | None = None,
) -> list[EpisodeSpec] | None:
    """Build val episodes under fixed-query protocol.

    Returns ``None`` if the episode list would be empty (caller should fall back to manifest).
    """
    fq = resolve_fixed_query_settings(eval_cfg)
    ss = int(support_seed) if support_seed is not None else fq.support_seed
    pool_size = max(int(support_size), int(fq.support_pool_size)) if fq.support_pool_size is not None else int(support_size)
    user_pool_size = (
        max(int(support_size), int(fq.user_pool_support_size))
        if fq.user_pool_support_size is not None
        else pool_size
    )
    # When user_pool_support_size gates eligibility (same-user-pool), draw support from that
    # pool so s10 val matches canonical nested test (--nested-support, pool=max shot).
    draw_pool_size = max(pool_size, user_pool_size)
    users = list(val_user_ids)
    if fq.same_user_pool:
        users = filter_users_by_min_row_count(frame, users, fq.min_query_images + user_pool_size)
    specs = build_fixed_query_episodes(
        frame,
        users,
        int(support_size),
        holdout_seed=fq.holdout_seed,
        support_seed=ss,
        min_query_images=fq.min_query_images,
        support_pool_size=draw_pool_size,
    )
    return specs if specs else None
