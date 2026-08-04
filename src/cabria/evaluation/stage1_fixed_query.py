"""Fixed-query proxy evaluation for Stage1 (aligns with Stage2 base_macro_srcc protocol)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from cabria.data.episode_sampler import build_episode_dataloader
from cabria.data.personalized_dataset import PersonalizedEpisodeDataset, load_personalized_frame
from cabria.evaluation.fixed_query import build_fixed_query_val_episodes, resolve_fixed_query_settings
from cabria.utils.common import load_json
from cabria.utils.metrics import macro_user_correlations, same_image_rank_correlation


@torch.no_grad()
def evaluate_stage1_fixed_query_base(
    model: torch.nn.Module,
    data_config: dict[str, Any],
    backbone_cfg: dict[str, Any],
    device: torch.device,
    eval_cfg: dict[str, Any] | None,
    *,
    split: str = "val",
) -> dict[str, float]:
    """Macro SRCC of Stage1 generic scores on fixed-query personalized episodes."""
    eval_cfg = dict(eval_cfg or {})
    personalized_cfg = data_config.get("personalized_dataset")
    user_split_cfg = data_config.get("user_split")
    if not personalized_cfg or not user_split_cfg:
        return {
            "fixed_query_macro_srcc": float("nan"),
            "fixed_query_macro_plcc": float("nan"),
            "fixed_query_same_image_srcc": float("nan"),
            "fixed_query_n_episodes": 0.0,
        }

    frame = load_personalized_frame(personalized_cfg["name"], personalized_cfg["root"])
    split_payload = load_json(user_split_cfg["split_file"])
    split_users = [str(user_id) for user_id in split_payload["splits"][split]]
    support_size = int(
        eval_cfg.get(
            "fixed_query_proxy_support_size",
            eval_cfg.get("fixed_query_user_pool_support_size", 100),
        )
    )
    episodes = build_fixed_query_val_episodes(
        frame,
        split_users,
        support_size,
        eval_cfg=eval_cfg,
    )
    if not episodes:
        return {
            "fixed_query_macro_srcc": float("nan"),
            "fixed_query_macro_plcc": float("nan"),
            "fixed_query_same_image_srcc": float("nan"),
            "fixed_query_n_episodes": 0.0,
        }

    fq = resolve_fixed_query_settings(eval_cfg)
    max_query_size = int(eval_cfg.get("fixed_query_eval_query_size", eval_cfg.get("eval_query_size", fq.min_query_images)))
    dataset = PersonalizedEpisodeDataset(
        frame=frame,
        episodes=episodes,
        image_size=int(backbone_cfg["image_size"]),
        train=False,
        mean=backbone_cfg.get("mean"),
        std=backbone_cfg.get("std"),
        use_naflex=bool(backbone_cfg.get("use_naflex", False)),
        preferred_long_side=backbone_cfg.get("preferred_long_side"),
        patch_size=int(backbone_cfg["patch_size"]),
        max_num_patches=backbone_cfg.get("max_num_patches"),
        max_query_size=max_query_size,
        image_prior_lookup=None,
    )
    loader = build_episode_dataloader(
        dataset,
        shuffle=False,
        num_workers=min(int(eval_cfg.get("fixed_query_num_workers", 4)), 4),
        batch_size=1,
        persistent_workers=False,
    )

    rows: list[dict[str, float | str]] = []
    model.eval()
    for batch in tqdm(loader, desc=f"Stage1 fixed-query {split}", leave=False):
        query_batch = batch["query_batch"]
        if isinstance(query_batch, dict):
            query_images = {key: value.to(device, non_blocking=True) for key, value in query_batch.items()}
        else:
            query_images = query_batch.to(device, non_blocking=True)
        outputs = model(query_images)
        predictions = outputs["score"].detach().cpu().view(-1)
        targets = batch["query_scores"].detach().cpu().view(-1)
        for idx, image_id in enumerate(batch["query_image_ids"]):
            rows.append(
                {
                    "user_id": str(batch["user_id"]),
                    "image_id": str(image_id),
                    "score": float(targets[idx]),
                    "prediction": float(predictions[idx]),
                }
            )

    if not rows:
        return {
            "fixed_query_macro_srcc": float("nan"),
            "fixed_query_macro_plcc": float("nan"),
            "fixed_query_same_image_srcc": float("nan"),
            "fixed_query_n_episodes": float(len(episodes)),
        }

    metric_frame = pd.DataFrame(rows)
    macro = macro_user_correlations(metric_frame)
    return {
        "fixed_query_macro_srcc": float(macro["macro_srcc"]),
        "fixed_query_macro_plcc": float(macro["macro_plcc"]),
        "fixed_query_same_image_srcc": float(same_image_rank_correlation(metric_frame)),
        "fixed_query_n_episodes": float(len(episodes)),
    }


def stage1_checkpoint_selection_score(
    metrics: dict[str, float],
    fixed_query_metrics: dict[str, float],
    eval_cfg: dict[str, Any] | None,
    checkpoint_metric: str,
) -> tuple[float, str]:
    """Resolve scalar score and metric key used for Stage1 checkpoint selection."""
    eval_cfg = dict(eval_cfg or {})
    metric_key = str(checkpoint_metric).lower()
    fq_weight = float(eval_cfg.get("checkpoint_fixed_query_weight", 1.0 if metric_key.startswith("fixed_query") else 0.0))
    val_weight = float(eval_cfg.get("checkpoint_val_weight", 1.0 if fq_weight <= 0.0 else 0.0))

    if metric_key == "fixed_query_macro_srcc":
        score = float(fixed_query_metrics.get("fixed_query_macro_srcc", float("nan")))
        return score, "fixed_query_macro_srcc"
    if metric_key == "blend_macro_srcc":
        val_score = float(metrics.get("macro_srcc", float("nan")))
        fq_score = float(fixed_query_metrics.get("fixed_query_macro_srcc", float("nan")))
        if val_score != val_score:
            val_score = 0.0
        if fq_score != fq_score:
            fq_score = 0.0
        blend_val_w = float(eval_cfg.get("blend_val_weight", 0.35))
        blend_fq_w = float(eval_cfg.get("blend_fixed_query_weight", 0.65))
        total_w = blend_val_w + blend_fq_w
        if total_w <= 0.0:
            total_w = 1.0
        score = (blend_val_w * val_score + blend_fq_w * fq_score) / total_w
        return score, "blend_macro_srcc"
    if fq_weight > 0.0 and val_weight > 0.0:
        val_score = float(metrics.get("macro_srcc", float("nan")))
        fq_score = float(fixed_query_metrics.get("fixed_query_macro_srcc", float("nan")))
        if val_score != val_score:
            val_score = 0.0
        if fq_score != fq_score:
            fq_score = 0.0
        total_w = fq_weight + val_weight
        score = (val_weight * val_score + fq_weight * fq_score) / total_w
        return score, f"blend(val={val_weight},fq={fq_weight})"

    key = metric_key if metric_key in metrics else "srcc"
    score = float(metrics.get(key, float("nan")))
    return score, key
