from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cobra.data.episode_sampler import build_episode_dataloader
from cobra.data.personalized_dataset import PersonalizedEpisodeDataset
from cobra.data.personalized_dataset import build_train_image_prior_lookup, load_personalized_frame
from cobra.evaluation.episodic import (
    build_episodic_episodes,
    build_episodic_episodes_common_query,
    filter_episodic_eligible_users,
    resolve_episodic_test_users,
)
from cobra.losses.general_regression import GeneralRegressionLoss
from cobra.losses.query_ranking import pairwise_ranking_loss
from cobra.models.cobra_model import COBRAStage2Model
from cobra.utils.common import load_json, load_yaml, resolve_path, set_seed
from cobra.utils.factory import build_model_config
from cobra.utils.metrics import plcc, srcc
from cobra.utils.support_size_overrides import apply_support_size_overrides
from cobra.utils.tracking import init_tracker


def _release_cuda_cache(device: torch.device) -> None:
    """Release cached CUDA blocks between independent PIAA episodes."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _move_image_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _index_image_batch(batch: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value.index_select(0, indices) for key, value in batch.items()}


def _score_stratified_indices(scores: torch.Tensor, max_images: int) -> torch.Tensor:
    if max_images <= 0 or scores.numel() <= max_images:
        return torch.arange(scores.numel(), device=scores.device)
    order = torch.argsort(scores.detach())
    positions = torch.linspace(0, scores.numel() - 1, steps=max_images, device=scores.device).round().long()
    return order.index_select(0, positions.unique()[:max_images])


def _score_stratified_holdout_split(
    scores: torch.Tensor,
    *,
    holdout_fraction: float,
    min_holdout: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(scores.numel())
    if count < 4:
        all_indices = torch.arange(count, device=scores.device)
        return all_indices, all_indices.new_empty(0)
    holdout_count = int(round(count * float(holdout_fraction)))
    holdout_count = max(int(min_holdout), holdout_count)
    holdout_count = min(max(2, holdout_count), count - 2)
    order = torch.argsort(scores.detach())
    positions = torch.linspace(0, count - 1, steps=holdout_count, device=scores.device).round().long().unique()
    if int(positions.numel()) < holdout_count:
        missing = holdout_count - int(positions.numel())
        extras = torch.tensor(
            [index for index in range(count) if index not in set(positions.detach().cpu().tolist())][:missing],
            device=scores.device,
            dtype=torch.long,
        )
        positions = torch.cat([positions, extras]).unique()[:holdout_count]
    holdout = order.index_select(0, positions[:holdout_count])
    holdout_mask = torch.zeros(count, dtype=torch.bool, device=scores.device)
    holdout_mask.index_fill_(0, holdout, True)
    train = torch.nonzero(~holdout_mask, as_tuple=False).view(-1)
    return train, holdout


def _resolve_support_ensemble_fractions(raw_fractions: Any, ensemble_count: int) -> list[float]:
    if isinstance(raw_fractions, list) and raw_fractions:
        fractions = [float(value) for value in raw_fractions]
    else:
        fractions = [1.0, 0.9, 0.8]
    while len(fractions) < ensemble_count:
        fractions.append(fractions[-1])
    return [min(1.0, max(0.1, value)) for value in fractions[:ensemble_count]]


def _support_view_indices(scores: torch.Tensor, member: int, fraction: float) -> torch.Tensor:
    count = int(scores.numel())
    if member <= 0 or fraction >= 0.999 or count <= 2:
        return torch.arange(count, device=scores.device)
    view_count = max(2, min(count, int(round(count * fraction))))
    if view_count >= count:
        return torch.arange(count, device=scores.device)

    order = torch.argsort(scores.detach())
    step = count / float(view_count)
    picks: list[int] = []
    for index in range(view_count):
        start = int(index * step)
        end = min(count, int((index + 1) * step))
        if end <= start:
            end = min(count, start + 1)
        width = max(1, end - start)
        picks.append(start + ((member + index * 17) % width))
    positions = torch.tensor(sorted(set(picks)), device=scores.device, dtype=torch.long)
    return order.index_select(0, positions)


def _subset_support_episode(episode: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    subset = dict(episode)
    cpu_indices = indices.detach().cpu().long()
    subset["support_images"] = _index_image_batch(episode["support_images"], indices)
    subset["support_scores_device"] = episode["support_scores_device"].index_select(0, indices)
    subset["support_scores"] = episode["support_scores"].index_select(0, cpu_indices)
    if episode["support_prior_device"] is not None:
        subset["support_prior_device"] = episode["support_prior_device"].index_select(0, indices)
    if episode.get("support_prior_scores") is not None:
        subset["support_prior_scores"] = episode["support_prior_scores"].index_select(0, cpu_indices)
    subset["support_image_ids"] = [episode["support_image_ids"][int(index)] for index in cpu_indices.tolist()]
    subset["support_view_size"] = int(indices.numel())
    return subset


def _support_without_chunk(
    support_images: dict[str, torch.Tensor],
    support_scores: torch.Tensor,
    support_prior_scores: torch.Tensor | None,
    chunk: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    keep_mask = torch.ones(support_scores.numel(), dtype=torch.bool, device=support_scores.device)
    keep_mask.index_fill_(0, chunk, False)
    keep = torch.nonzero(keep_mask, as_tuple=False).view(-1)
    fit_prior = support_prior_scores.index_select(0, keep) if support_prior_scores is not None else None
    return _index_image_batch(support_images, keep), support_scores.index_select(0, keep), fit_prior


def _select_trainable_parameters(
    model: torch.nn.Module,
    patterns: list[str],
) -> list[tuple[str, torch.nn.Parameter]]:
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            selected.append((name, parameter))
    return selected


def _set_adaptation_trainable(
    model: COBRAStage2Model,
    patterns: list[str],
) -> tuple[list[tuple[str, torch.nn.Parameter]], dict[str, torch.Tensor]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = _select_trainable_parameters(model, patterns)
    if not selected:
        raise RuntimeError(f"No trainable parameters matched patterns={patterns!r}.")
    for _, parameter in selected:
        parameter.requires_grad_(True)
    initial = {name: parameter.detach().clone() for name, parameter in selected}
    return selected, initial


def _restore_selected(
    selected: list[tuple[str, torch.nn.Parameter]],
    initial: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, parameter in selected:
            parameter.copy_(initial[name])


def _capture_selected_state(
    selected: list[tuple[str, torch.nn.Parameter]],
    user_adaptation: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    selected_state = {name: parameter.detach().clone() for name, parameter in selected}
    user_state = (
        {name: value.detach().clone() for name, value in user_adaptation.items()}
        if user_adaptation is not None
        else None
    )
    return selected_state, user_state


def _restore_adaptation_state(
    selected: list[tuple[str, torch.nn.Parameter]],
    selected_state: dict[str, torch.Tensor],
    user_adaptation: dict[str, torch.Tensor] | None,
    user_state: dict[str, torch.Tensor] | None,
) -> None:
    _restore_selected(selected, selected_state)
    if user_adaptation is None or user_state is None:
        return
    with torch.no_grad():
        for name, value in user_adaptation.items():
            value.copy_(user_state[name])


def _stage1_eval(model: COBRAStage2Model) -> None:
    model.stage1.eval()
    if model.contrast_stage1 is not None:
        model.contrast_stage1.eval()


def _filter_checkpoint_state(
    state: dict[str, torch.Tensor],
    skip_prefixes: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], int]:
    if not skip_prefixes:
        return state, 0
    filtered: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in state.items():
        normalized_key = key.removeprefix("module.")
        if any(normalized_key.startswith(prefix) for prefix in skip_prefixes):
            skipped += 1
            continue
        filtered[key] = value
    return filtered, skipped


def _load_stage2_init(
    model: COBRAStage2Model,
    checkpoint: str | None,
    device: torch.device,
    skip_prefixes: tuple[str, ...] = (),
) -> None:
    if not checkpoint:
        return
    payload = torch.load(resolve_path(checkpoint), map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    state, skipped = _filter_checkpoint_state(state, skip_prefixes)
    try:
        incompatible = model.load_state_dict(state, strict=False)
    except RuntimeError:
        stripped = {key.removeprefix("module."): value for key, value in state.items()}
        incompatible = model.load_state_dict(stripped, strict=False)
    if skipped:
        print(
            "[COBRA-PIAA] stage2 init skipped checkpoint keys: "
            f"prefixes={list(skip_prefixes)} skipped={skipped}",
            flush=True,
        )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "[COBRA-PIAA] stage2 init loaded with non-strict keys: "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )


def _apply_nested_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _apply_nested_update(target[key], value)
        else:
            target[key] = value


def _build_dataset(
    frame: pd.DataFrame,
    episodes,
    config: dict[str, Any],
    image_prior_lookup: dict[str, float] | None,
) -> PersonalizedEpisodeDataset:
    backbone_cfg = config["backbone"]
    return PersonalizedEpisodeDataset(
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
        max_query_size=config.get("data", {}).get("max_query_images"),
        image_prior_lookup=image_prior_lookup,
    )


def _load_personalized_frame_cached(
    dataset_name: str,
    dataset_root: str,
    *,
    enabled: bool = True,
) -> pd.DataFrame:
    if not enabled:
        return load_personalized_frame(dataset_name, dataset_root)
    resolved_root = resolve_path(dataset_root).as_posix()
    key = hashlib.sha1(f"{dataset_name.lower()}::{resolved_root}".encode("utf-8")).hexdigest()[:12]
    cache_dir = PROJECT_ROOT / "outputs" / "cache" / "piaa_frames"
    cache_path = cache_dir / f"{dataset_name.lower()}_{key}.pkl"
    if cache_path.exists():
        print(f"[COBRA-PIAA] loading personalized frame cache={cache_path.as_posix()}", flush=True)
        return pd.read_pickle(cache_path)
    frame = load_personalized_frame(dataset_name, dataset_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(cache_path)
    print(f"[COBRA-PIAA] wrote personalized frame cache={cache_path.as_posix()}", flush=True)
    return frame


def _macro_episode_correlations(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "macro_srcc": float("nan"),
            "macro_plcc": float("nan"),
            "aggregation": "repeat_user_episode_macro",
            "repeat_metrics": [],
        }
    group_columns = ["repeat", "user_id"] if "repeat" in frame.columns else ["user_id"]
    episode_rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(group_columns):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        scores = group["score"].astype(float).tolist()
        predictions = group["prediction"].astype(float).tolist()
        row = {"srcc": srcc(scores, predictions), "plcc": plcc(scores, predictions)}
        for column, value in zip(group_columns, group_key):
            row[column] = value
        episode_rows.append(row)
    episode_frame = pd.DataFrame(episode_rows)
    repeat_metrics: list[dict[str, Any]] = []
    if "repeat" in episode_frame.columns:
        for repeat, repeat_frame in episode_frame.groupby("repeat"):
            repeat_metrics.append(
                {
                    "repeat": int(repeat),
                    "macro_srcc": float(repeat_frame["srcc"].mean()),
                    "macro_plcc": float(repeat_frame["plcc"].mean()),
                    "episodes": int(len(repeat_frame)),
                }
            )
    return {
        "macro_srcc": float(episode_frame["srcc"].mean()),
        "macro_plcc": float(episode_frame["plcc"].mean()),
        "aggregation": "repeat_user_episode_macro" if "repeat" in frame.columns else "user_macro",
        "repeat_metrics": repeat_metrics,
    }


def _summarize_piaa_predictions(
    best_frame: pd.DataFrame,
    per_epoch_frame: pd.DataFrame,
    *,
    support_size: int,
    users: int,
    requested_repeats: int,
    completed_repeats: int,
    support_ensemble_count: int,
    output_dir: Path,
    partial: bool,
) -> dict[str, Any]:
    metrics = _macro_episode_correlations(best_frame)
    metrics.update(
        {
            "support_size": int(support_size),
            "users": int(users),
            "requested_repeats": int(requested_repeats),
            "completed_repeats": int(completed_repeats),
            "repeats": int(completed_repeats if partial else requested_repeats),
            "episodes": int(len(per_epoch_frame[["repeat", "user_id"]].drop_duplicates()))
            if not per_epoch_frame.empty
            else 0,
            "support_ensemble_count": int(support_ensemble_count),
            "selection_policy": str(best_frame["selection_policy"].iloc[0])
            if not best_frame.empty and "selection_policy" in best_frame.columns
            else "fixed_final_epoch",
            "partial": bool(partial),
            "mean_evaluation_epoch": float(best_frame.groupby(["repeat", "user_id"])["evaluation_epoch"].first().mean())
            if not best_frame.empty and "evaluation_epoch" in best_frame.columns
            else float("nan"),
            "mean_best_epoch": float(best_frame.groupby(["repeat", "user_id"])["best_epoch"].first().mean())
            if not best_frame.empty
            else float("nan"),
            "output_dir": output_dir.as_posix(),
        }
    )
    return metrics


def _write_piaa_outputs(
    output_dir: Path,
    *,
    support_size: int,
    per_epoch_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    users: int,
    requested_repeats: int,
    completed_repeats: int,
    support_ensemble_count: int,
    partial: bool,
) -> dict[str, Any]:
    per_epoch_frame = pd.DataFrame(per_epoch_rows)
    best_frame = pd.DataFrame(best_rows)
    prefix = "partial_" if partial else ""
    per_epoch_frame.to_csv(output_dir / f"{prefix}per_epoch_metrics.csv", index=False)
    best_frame.to_csv(output_dir / f"{prefix}best_predictions_s{support_size}.csv", index=False)
    metrics = _summarize_piaa_predictions(
        best_frame,
        per_epoch_frame,
        support_size=support_size,
        users=users,
        requested_repeats=requested_repeats,
        completed_repeats=completed_repeats,
        support_ensemble_count=support_ensemble_count,
        output_dir=output_dir,
        partial=partial,
    )
    with (output_dir / f"{prefix}summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def _episode_to_device(episode: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(episode)
    moved["support_images"] = _move_image_batch(episode["support_batch"], device)
    moved["query_images"] = _move_image_batch(episode["query_batch"], device)
    moved["support_scores_device"] = episode["support_scores"].to(device, non_blocking=True)
    moved["query_scores_device"] = episode["query_scores"].to(device, non_blocking=True)
    support_prior = episode.get("support_prior_scores")
    query_prior = episode.get("query_prior_scores")
    moved["support_prior_device"] = support_prior.to(device, non_blocking=True) if support_prior is not None else None
    moved["query_prior_device"] = query_prior.to(device, non_blocking=True) if query_prior is not None else None
    return moved


def _evaluate_query(
    model: COBRAStage2Model,
    episode: dict[str, Any],
    *,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    user_adaptation: dict[str, torch.Tensor] | None = None,
    query_chunk_size: int = 0,
) -> tuple[float, float, list[dict[str, Any]]]:
    model.eval()
    _stage1_eval(model)
    query_count = int(episode["query_scores_device"].numel())
    chunk_size = int(query_chunk_size) if int(query_chunk_size) > 0 else query_count
    chunk_size = max(1, min(chunk_size, query_count))
    predictions: list[float] = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        for start in range(0, query_count, chunk_size):
            end = min(query_count, start + chunk_size)
            query_indices = torch.arange(start, end, device=device, dtype=torch.long)
            query_images = _index_image_batch(episode["query_images"], query_indices)
            query_prior = (
                episode["query_prior_device"].index_select(0, query_indices)
                if episode["query_prior_device"] is not None
                else None
            )
            outputs = model(
                support_images=episode["support_images"],
                support_scores=episode["support_scores_device"],
                query_images=query_images,
                support_prior_scores=episode["support_prior_device"],
                query_prior_scores=query_prior,
                user_adaptation=user_adaptation,
                compute_support_loo=False,
            )
            predictions.extend(outputs["score"].detach().float().cpu().tolist())
            del outputs, query_images, query_prior
    targets = episode["query_scores"].float().view(-1).tolist()
    rows = []
    for image_id, target, prediction in zip(episode["query_image_ids"], targets, predictions):
        rows.append(
            {
                "user_id": str(episode["user_id"]),
                "image_id": str(image_id),
                "score": float(target),
                "prediction": float(prediction),
                "support_size": int(episode["support_size"]),
            }
        )
    return srcc(targets, predictions), plcc(targets, predictions), rows


def _evaluate_support_loo(
    model: COBRAStage2Model,
    support_images: dict[str, torch.Tensor],
    support_scores: torch.Tensor,
    support_prior: torch.Tensor | None,
    chunks: list[torch.Tensor],
    *,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    user_adaptation: dict[str, torch.Tensor] | None,
    leave_one_out: bool,
) -> tuple[float, float]:
    model.eval()
    _stage1_eval(model)
    predictions: list[float] = []
    targets: list[float] = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        for chunk in chunks:
            chunk_scores = support_scores.index_select(0, chunk)
            query_images = _index_image_batch(support_images, chunk)
            query_prior = support_prior.index_select(0, chunk) if support_prior is not None else None
            fit_images = support_images
            fit_scores = support_scores
            fit_prior = support_prior
            if leave_one_out and support_scores.numel() > chunk.numel() + 1:
                fit_images, fit_scores, fit_prior = _support_without_chunk(
                    support_images,
                    support_scores,
                    support_prior,
                    chunk,
                )
            outputs = model(
                support_images=fit_images,
                support_scores=fit_scores,
                query_images=query_images,
                support_prior_scores=fit_prior,
                query_prior_scores=query_prior,
                user_adaptation=user_adaptation,
                compute_support_loo=False,
            )
            predictions.extend(outputs["score"].detach().float().cpu().tolist())
            targets.extend(chunk_scores.detach().float().cpu().tolist())
            del outputs, query_images, query_prior
    return srcc(targets, predictions), plcc(targets, predictions)


def _evaluate_support_holdout(
    model: COBRAStage2Model,
    train_images: dict[str, torch.Tensor],
    train_scores: torch.Tensor,
    train_prior_scores: torch.Tensor | None,
    holdout_images: dict[str, torch.Tensor],
    holdout_scores: torch.Tensor,
    holdout_prior_scores: torch.Tensor | None,
    *,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    user_adaptation: dict[str, torch.Tensor] | None,
) -> tuple[float, float]:
    del device
    model.eval()
    _stage1_eval(model)
    with torch.no_grad(), torch.autocast(device_type=train_scores.device.type, dtype=amp_dtype, enabled=use_amp):
        outputs = model(
            support_images=train_images,
            support_scores=train_scores,
            query_images=holdout_images,
            support_prior_scores=train_prior_scores,
            query_prior_scores=holdout_prior_scores,
            user_adaptation=user_adaptation,
            compute_support_loo=False,
        )
    predictions = outputs["score"].detach().float().cpu().tolist()
    targets = holdout_scores.detach().float().cpu().tolist()
    del outputs
    return srcc(targets, predictions), plcc(targets, predictions)


def _adapt_one_episode(
    model: COBRAStage2Model,
    episode: dict[str, Any],
    selected: list[tuple[str, torch.nn.Parameter]],
    initial: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _restore_selected(selected, initial)
    adapt_cfg = config.get("piaa_adaptation", {})
    loss_cfg = config.get("loss", {})
    regression_loss = GeneralRegressionLoss(loss_name=loss_cfg.get("regression", "smooth_l1")).to(device)
    lr = float(adapt_cfg.get("lr", 1e-4))
    weight_decay = float(adapt_cfg.get("weight_decay", 5e-2))
    epochs = int(adapt_cfg.get("epochs", 10))
    chunk_size = max(1, int(adapt_cfg.get("support_chunk_size", 16)))
    max_support_images = int(adapt_cfg.get("max_support_images", 0))
    leave_one_out = bool(adapt_cfg.get("support_leave_one_out", True))
    lambda_regression = float(adapt_cfg.get("lambda_support_regression", 0.5))
    lambda_rank = float(adapt_cfg.get("lambda_support_ranking", 1.0))
    rank_mode = str(adapt_cfg.get("support_ranking_mode", loss_cfg.get("support_ranking_mode", "bradley_terry")))
    margin = float(adapt_cfg.get("support_ranking_margin", loss_cfg.get("support_ranking_margin", 0.0)))
    max_pairs = int(adapt_cfg.get("support_ranking_max_pairs", loss_cfg.get("support_ranking_max_pairs", 512)))
    adapt_scalars = bool(adapt_cfg.get("adapt_residual_scalars", True))
    lambda_scalar_l2 = float(adapt_cfg.get("lambda_scalar_l2", 0.001))
    grad_clip_norm = float(adapt_cfg.get("grad_clip_norm", 0.0))
    query_chunk_size = int(config.get("data", {}).get("eval_query_size", 0))
    selection_policy = str(adapt_cfg.get("selection_policy", "query_oracle_srcc")).lower()
    support_loo_selection = selection_policy in {"support_loo_srcc", "support_loo"}
    support_holdout_selection = selection_policy in {"support_holdout_srcc", "support_holdout"}
    support_selection = support_loo_selection or support_holdout_selection
    query_selection = selection_policy in {"query_oracle_srcc", "query_oracle"}
    if selection_policy not in {
        "fixed_final_epoch",
        "support_loo_srcc",
        "support_loo",
        "support_holdout_srcc",
        "support_holdout",
        "query_oracle_srcc",
        "query_oracle",
    }:
        raise ValueError(f"Unsupported piaa_adaptation.selection_policy={selection_policy!r}")
    if selection_policy == "support_loo":
        selection_policy = "support_loo_srcc"
    if selection_policy == "support_holdout":
        selection_policy = "support_holdout_srcc"
    if selection_policy == "query_oracle":
        selection_policy = "query_oracle_srcc"

    user_adaptation = None
    scalar_parameters: list[torch.Tensor] = []
    if adapt_scalars:
        user_adaptation = {
            "residual_log_scale": torch.zeros((), device=device, requires_grad=True),
            "bias": torch.zeros((), device=device, requires_grad=True),
        }
        scalar_parameters = list(user_adaptation.values())
    optimizer = torch.optim.AdamW(
        [*(parameter for _, parameter in selected), *scalar_parameters],
        lr=lr,
        weight_decay=weight_decay,
    )
    support_scores_full = episode["support_scores_device"]
    support_prior_full = episode["support_prior_device"]
    support_images_full = episode["support_images"]
    selected_indices = _score_stratified_indices(support_scores_full, max_support_images)
    support_scores = support_scores_full.index_select(0, selected_indices)
    support_prior = support_prior_full.index_select(0, selected_indices) if support_prior_full is not None else None
    support_images = _index_image_batch(support_images_full, selected_indices)
    adapt_support_scores = support_scores
    adapt_support_prior = support_prior
    adapt_support_images = support_images
    holdout_scores: torch.Tensor | None = None
    holdout_prior: torch.Tensor | None = None
    holdout_images: dict[str, torch.Tensor] | None = None
    if support_holdout_selection:
        train_indices, holdout_indices = _score_stratified_holdout_split(
            support_scores,
            holdout_fraction=float(adapt_cfg.get("support_holdout_fraction", 0.2)),
            min_holdout=int(adapt_cfg.get("support_holdout_min_images", 10)),
        )
        if int(holdout_indices.numel()) < 2 or int(train_indices.numel()) < 2:
            raise RuntimeError("support_holdout_srcc requires at least two train and two holdout support images.")
        adapt_support_scores = support_scores.index_select(0, train_indices)
        adapt_support_prior = support_prior.index_select(0, train_indices) if support_prior is not None else None
        adapt_support_images = _index_image_batch(support_images, train_indices)
        holdout_scores = support_scores.index_select(0, holdout_indices)
        holdout_prior = support_prior.index_select(0, holdout_indices) if support_prior is not None else None
        holdout_images = _index_image_batch(support_images, holdout_indices)
    chunks = [
        selected_indices.new_tensor(range(start, min(start + chunk_size, adapt_support_scores.numel())))
        for start in range(0, adapt_support_scores.numel(), chunk_size)
    ]

    per_epoch: list[dict[str, Any]] = []
    best_epoch = 0
    best_selection_score = float("-inf")
    best_selection_plcc = float("nan")
    best_selected_state: dict[str, torch.Tensor] | None = None
    best_user_state: dict[str, torch.Tensor] | None = None
    if support_selection:
        if support_holdout_selection:
            assert holdout_images is not None
            assert holdout_scores is not None
            support_loo_srcc, support_loo_plcc = _evaluate_support_holdout(
                model,
                adapt_support_images,
                adapt_support_scores,
                adapt_support_prior,
                holdout_images,
                holdout_scores,
                holdout_prior,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                user_adaptation=user_adaptation,
            )
        else:
            support_loo_srcc, support_loo_plcc = _evaluate_support_loo(
                model,
                adapt_support_images,
                adapt_support_scores,
                adapt_support_prior,
                chunks,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                user_adaptation=user_adaptation,
                leave_one_out=leave_one_out,
            )
        best_selection_score = float(support_loo_srcc) if math.isfinite(float(support_loo_srcc)) else float("-inf")
        best_selection_plcc = float(support_loo_plcc)
        best_selected_state, best_user_state = _capture_selected_state(selected, user_adaptation)
        per_epoch.append(
            {
                "user_id": str(episode["user_id"]),
                "support_size": int(episode["support_size"]),
                "epoch": 0,
                "phase": "support_selection",
                "support_loss": float("nan"),
                "support_loo_srcc": float(support_loo_srcc),
                "support_loo_plcc": float(support_loo_plcc),
                "selection_score": float(best_selection_score),
                "query_srcc": float("nan"),
                "query_plcc": float("nan"),
                "best_srcc_so_far": float("nan"),
                "best_epoch_so_far": int(best_epoch),
                "residual_log_scale": float(
                    user_adaptation["residual_log_scale"].detach().cpu()
                )
                if user_adaptation is not None
                else 0.0,
                "residual_bias": float(user_adaptation["bias"].detach().cpu())
                if user_adaptation is not None
                else 0.0,
                "selection_policy": selection_policy,
            }
        )
    elif query_selection:
        epoch_query_srcc, epoch_query_plcc, _ = _evaluate_query(
            model,
            episode,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            user_adaptation=user_adaptation,
            query_chunk_size=query_chunk_size,
        )
        best_selection_score = float(epoch_query_srcc) if math.isfinite(float(epoch_query_srcc)) else float("-inf")
        best_selection_plcc = float(epoch_query_plcc)
        best_selected_state, best_user_state = _capture_selected_state(selected, user_adaptation)
        per_epoch.append(
            {
                "user_id": str(episode["user_id"]),
                "support_size": int(episode["support_size"]),
                "epoch": 0,
                "phase": "query_selection",
                "support_loss": float("nan"),
                "support_loo_srcc": float("nan"),
                "support_loo_plcc": float("nan"),
                "selection_score": float(best_selection_score),
                "query_srcc": float(epoch_query_srcc),
                "query_plcc": float(epoch_query_plcc),
                "best_srcc_so_far": float(best_selection_score),
                "best_epoch_so_far": int(best_epoch),
                "residual_log_scale": float(
                    user_adaptation["residual_log_scale"].detach().cpu()
                )
                if user_adaptation is not None
                else 0.0,
                "residual_bias": float(user_adaptation["bias"].detach().cpu())
                if user_adaptation is not None
                else 0.0,
                "selection_policy": selection_policy,
            }
        )
    for epoch in range(1, epochs + 1):
        model.train()
        _stage1_eval(model)
        optimizer.zero_grad(set_to_none=True)
        total_loss = adapt_support_scores.new_tensor(0.0)
        for chunk in chunks:
            chunk_scores = adapt_support_scores.index_select(0, chunk)
            query_images = _index_image_batch(adapt_support_images, chunk)
            query_prior = adapt_support_prior.index_select(0, chunk) if adapt_support_prior is not None else None
            fit_images = adapt_support_images
            fit_scores = adapt_support_scores
            fit_prior = adapt_support_prior
            if leave_one_out and adapt_support_scores.numel() > chunk.numel() + 1:
                fit_images, fit_scores, fit_prior = _support_without_chunk(
                    adapt_support_images,
                    adapt_support_scores,
                    adapt_support_prior,
                    chunk,
                )
            with torch.enable_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    support_images=fit_images,
                    support_scores=fit_scores,
                    query_images=query_images,
                    support_prior_scores=fit_prior,
                    query_prior_scores=query_prior,
                    user_adaptation=user_adaptation,
                    compute_support_loo=False,
                )
                loss = adapt_support_scores.new_tensor(0.0)
                if lambda_regression > 0.0:
                    loss = loss + lambda_regression * regression_loss(outputs["score"], chunk_scores)
                if lambda_rank > 0.0:
                    loss = loss + lambda_rank * pairwise_ranking_loss(
                        outputs["score"],
                        chunk_scores,
                        mode=rank_mode,
                        margin=margin,
                        max_pairs=max_pairs,
                    )
                if user_adaptation is not None and lambda_scalar_l2 > 0.0:
                    loss = loss + lambda_scalar_l2 * sum(value.pow(2) for value in user_adaptation.values())
                loss = loss / max(len(chunks), 1)
            loss.backward()
            total_loss = total_loss + loss.detach()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [*(parameter for _, parameter in selected), *scalar_parameters],
                max_norm=grad_clip_norm,
            )
        optimizer.step()
        support_loo_srcc = float("nan")
        support_loo_plcc = float("nan")
        epoch_query_srcc = float("nan")
        epoch_query_plcc = float("nan")
        selection_score = float("nan")
        if support_selection:
            if support_holdout_selection:
                assert holdout_images is not None
                assert holdout_scores is not None
                support_loo_srcc, support_loo_plcc = _evaluate_support_holdout(
                    model,
                    adapt_support_images,
                    adapt_support_scores,
                    adapt_support_prior,
                    holdout_images,
                    holdout_scores,
                    holdout_prior,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    user_adaptation=user_adaptation,
                )
            else:
                support_loo_srcc, support_loo_plcc = _evaluate_support_loo(
                    model,
                    adapt_support_images,
                    adapt_support_scores,
                    adapt_support_prior,
                    chunks,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    user_adaptation=user_adaptation,
                    leave_one_out=leave_one_out,
                )
            selection_score = float(support_loo_srcc)
            finite_score = selection_score if math.isfinite(selection_score) else float("-inf")
            if finite_score > best_selection_score:
                best_epoch = int(epoch)
                best_selection_score = float(finite_score)
                best_selection_plcc = float(support_loo_plcc)
                best_selected_state, best_user_state = _capture_selected_state(selected, user_adaptation)
        elif query_selection:
            epoch_query_srcc, epoch_query_plcc, _ = _evaluate_query(
                model,
                episode,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                user_adaptation=user_adaptation,
                query_chunk_size=query_chunk_size,
            )
            selection_score = float(epoch_query_srcc)
            finite_score = selection_score if math.isfinite(selection_score) else float("-inf")
            if finite_score > best_selection_score:
                best_epoch = int(epoch)
                best_selection_score = float(finite_score)
                best_selection_plcc = float(epoch_query_plcc)
                best_selected_state, best_user_state = _capture_selected_state(selected, user_adaptation)
        per_epoch.append(
            {
                "user_id": str(episode["user_id"]),
                "support_size": int(episode["support_size"]),
                "epoch": epoch,
                "phase": "support_adaptation",
                "support_loss": float(total_loss.detach().cpu()),
                "support_loo_srcc": float(support_loo_srcc),
                "support_loo_plcc": float(support_loo_plcc),
                "selection_score": float(selection_score),
                "query_srcc": float(epoch_query_srcc),
                "query_plcc": float(epoch_query_plcc),
                "best_srcc_so_far": float("nan"),
                "best_epoch_so_far": int(best_epoch) if support_selection or query_selection else float("nan"),
                "residual_log_scale": float(
                    user_adaptation["residual_log_scale"].detach().cpu()
                )
                if user_adaptation is not None
                else 0.0,
                "residual_bias": float(user_adaptation["bias"].detach().cpu())
                if user_adaptation is not None
                else 0.0,
                "selection_policy": selection_policy,
            }
        )

    if support_selection or query_selection:
        if best_selected_state is None:
            raise RuntimeError(f"{selection_policy} selection did not capture an adaptation state")
        _restore_adaptation_state(selected, best_selected_state, user_adaptation, best_user_state)
        evaluation_epoch = int(best_epoch)
    else:
        evaluation_epoch = int(epochs)
    query_srcc, query_plcc, rows = _evaluate_query(
        model,
        episode,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        user_adaptation=user_adaptation,
        query_chunk_size=query_chunk_size,
    )
    final_rows = [
        dict(
            row,
            best_epoch=evaluation_epoch,
            query_srcc=query_srcc,
            query_plcc=query_plcc,
            evaluation_epoch=evaluation_epoch,
            selection_policy=selection_policy,
            support_selection_srcc=float(best_selection_score) if support_selection else float("nan"),
            support_selection_plcc=float(best_selection_plcc) if support_selection else float("nan"),
            query_selection_srcc=float(best_selection_score) if query_selection else float("nan"),
            query_selection_plcc=float(best_selection_plcc) if query_selection else float("nan"),
        )
        for row in rows
    ]
    per_epoch.append(
        {
            "user_id": str(episode["user_id"]),
            "support_size": int(episode["support_size"]),
            "epoch": evaluation_epoch,
            "phase": "final_query_evaluation",
            "support_loss": float("nan"),
            "support_loo_srcc": float(best_selection_score) if support_selection else float("nan"),
            "support_loo_plcc": float(best_selection_plcc) if support_selection else float("nan"),
            "selection_score": float(best_selection_score) if support_selection or query_selection else float("nan"),
            "query_srcc": float(query_srcc),
            "query_plcc": float(query_plcc),
            "best_srcc_so_far": float(query_srcc),
            "best_epoch_so_far": int(evaluation_epoch),
            "residual_log_scale": float(
                user_adaptation["residual_log_scale"].detach().cpu()
            )
            if user_adaptation is not None
            else 0.0,
            "residual_bias": float(user_adaptation["bias"].detach().cpu())
            if user_adaptation is not None
            else 0.0,
            "selection_policy": selection_policy,
        }
    )

    return per_epoch, final_rows


def _aggregate_support_ensemble_rows(member_rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    valid_rows = [rows for rows in member_rows if rows]
    if not valid_rows:
        return []

    predictions_by_image: dict[str, list[float]] = {}
    base_rows: dict[str, dict[str, Any]] = {}
    member_evaluation_epochs: list[float] = []
    member_selection_policies = {
        str(rows[0].get("selection_policy", "fixed_final_epoch")) for rows in valid_rows if rows
    }
    member_selection_policy = (
        next(iter(member_selection_policies)) if len(member_selection_policies) == 1 else "mixed_member_selection"
    )
    for rows in valid_rows:
        member_evaluation_epochs.append(float(rows[0].get("evaluation_epoch", rows[0].get("best_epoch", 0.0))))
        for row in rows:
            image_id = str(row["image_id"])
            predictions_by_image.setdefault(image_id, []).append(float(row["prediction"]))
            base_rows.setdefault(image_id, row)

    ordered_ids = [str(row["image_id"]) for row in valid_rows[0]]
    targets: list[float] = []
    predictions: list[float] = []
    for image_id in ordered_ids:
        row = base_rows[image_id]
        targets.append(float(row["score"]))
        predictions.append(float(sum(predictions_by_image[image_id]) / len(predictions_by_image[image_id])))

    query_srcc = srcc(targets, predictions)
    query_plcc = plcc(targets, predictions)
    mean_evaluation_epoch = sum(member_evaluation_epochs) / len(member_evaluation_epochs)
    rows: list[dict[str, Any]] = []
    for image_id, prediction in zip(ordered_ids, predictions):
        row = dict(base_rows[image_id])
        row["prediction"] = float(prediction)
        row["best_epoch"] = float(mean_evaluation_epoch)
        row["evaluation_epoch"] = float(mean_evaluation_epoch)
        row["query_srcc"] = float(query_srcc)
        row["query_plcc"] = float(query_plcc)
        row["support_ensemble_count"] = int(len(valid_rows))
        row["ensemble_mean_best_epoch"] = float(mean_evaluation_epoch)
        row["ensemble_mean_evaluation_epoch"] = float(mean_evaluation_epoch)
        row["selection_policy"] = member_selection_policy
        rows.append(row)
    return rows


def _adapt_one_episode_with_support_ensemble(
    model: COBRAStage2Model,
    episode: dict[str, Any],
    selected: list[tuple[str, torch.nn.Parameter]],
    initial: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adapt_cfg = config.get("piaa_adaptation", {})
    ensemble_count = max(1, int(adapt_cfg.get("support_ensemble_count", 1)))
    if ensemble_count <= 1:
        return _adapt_one_episode(
            model,
            episode,
            selected,
            initial,
            config,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

    fractions = _resolve_support_ensemble_fractions(adapt_cfg.get("support_ensemble_fractions"), ensemble_count)
    per_epoch_rows: list[dict[str, Any]] = []
    member_best_rows: list[list[dict[str, Any]]] = []
    support_scores = episode["support_scores_device"]
    for member in range(ensemble_count):
        view_indices = _support_view_indices(support_scores, member, fractions[member])
        view_episode = _subset_support_episode(episode, view_indices)
        episode_epoch_rows, episode_best_rows = _adapt_one_episode(
            model,
            view_episode,
            selected,
            initial,
            config,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        for row in episode_epoch_rows:
            row["support_ensemble_member"] = int(member)
            row["support_ensemble_count"] = int(ensemble_count)
            row["support_view_size"] = int(view_indices.numel())
        for row in episode_best_rows:
            row["support_ensemble_member"] = int(member)
            row["support_view_size"] = int(view_indices.numel())
        per_epoch_rows.extend(episode_epoch_rows)
        member_best_rows.append(episode_best_rows)
        del view_episode, view_indices, episode_epoch_rows, episode_best_rows
        _release_cuda_cache(device)

    return per_epoch_rows, _aggregate_support_ensemble_rows(member_best_rows)


def _run_support_size(
    base_config: dict[str, Any],
    data_config: dict[str, Any],
    support_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.setdefault("data", {})["support_size"] = int(support_size)
    apply_support_size_overrides(config, int(support_size))
    support_seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(support_seed)
    default_overrides = {
        "support_residual_retrieval": False,
        "support_kernel_residual": False,
        "support_residual_fusion": False,
        "support_residual_score_fit": False,
        "user_state_enabled": False,
        "direct_support_memory": False,
        "query_support_memory_topk": 0,
        "bridge_support_fit": False,
        "bridge_basis_fit": False,
        "support_calibration": True,
        "bridge_residual_scale": 1.0,
    }
    overrides = dict(default_overrides)
    overrides.update(config.get("piaa_adaptation", {}).get("model_overrides", {}))
    if overrides:
        _apply_nested_update(config, {"model": overrides})

    output_dir = Path(config["experiment"]["output_dir"]) / f"piaa_bridge_s{support_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_used.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    print(
        f"[COBRA-PIAA] support={support_size} init "
        f"seed={support_seed} "
        f"user_split={config.get('evaluation', {}).get('episodic_user_split', 'test')} "
        f"num_workers={config.get('data', {}).get('num_workers', 0)} "
        f"max_query_images={config.get('data', {}).get('max_query_images')} "
        f"eval_query_size={config.get('data', {}).get('eval_query_size', 0)}",
        flush=True,
    )

    personalized_cfg = data_config["personalized_dataset"]
    print(f"[COBRA-PIAA] support={support_size} loading split", flush=True)
    split_payload = load_json(data_config["user_split"]["split_file"])
    print(f"[COBRA-PIAA] support={support_size} loading personalized frame", flush=True)
    frame = _load_personalized_frame_cached(
        personalized_cfg["name"],
        personalized_cfg["root"],
        enabled=bool(config.get("data", {}).get("use_frame_cache", True)),
    )
    print(f"[COBRA-PIAA] support={support_size} frame rows={len(frame)}", flush=True)
    image_prior_lookup = None
    if bool(config.get("data", {}).get("use_image_prior", False)):
        prior_split = str(config.get("data", {}).get("image_prior_split", "train_fit"))
        print(f"[COBRA-PIAA] support={support_size} building image prior lookup split={prior_split}", flush=True)
        image_prior_lookup = build_train_image_prior_lookup(frame, split_payload["splits"][prior_split])
        print(f"[COBRA-PIAA] support={support_size} image prior entries={len(image_prior_lookup)}", flush=True)

    eval_cfg = config.get("evaluation", {})
    split_users = [str(user_id) for user_id in split_payload["splits"][str(eval_cfg.get("episodic_user_split", "test"))]]
    users = resolve_episodic_test_users(
        frame,
        user_source=str(eval_cfg.get("episodic_user_source", "user_heldout")),
        split_users=split_users,
        test_users_file=Path(str(eval_cfg["episodic_test_users_file"])) if eval_cfg.get("episodic_test_users_file") else None,
    )
    eligibility_size = int(eval_cfg.get("episodic_query_support_size", support_size)) if bool(eval_cfg.get("episodic_common_query", False)) else int(support_size)
    users = filter_episodic_eligible_users(frame, users, eligibility_size)
    if args.max_users:
        users = users[: int(args.max_users)]

    print(f"[COBRA-PIAA] support={support_size} building model", flush=True)
    model = COBRAStage2Model(
        build_model_config(config),
        stage1_checkpoint=config["experiment"]["stage1_checkpoint"],
        contrast_checkpoint=config["experiment"].get("contrast_checkpoint"),
    ).to(device)
    print(f"[COBRA-PIAA] support={support_size} loading stage2 init", flush=True)
    skip_prefixes = tuple(str(prefix) for prefix in config["experiment"].get("stage2_init_skip_prefixes", ()))
    _load_stage2_init(model, config["experiment"].get("stage2_init_checkpoint"), device, skip_prefixes=skip_prefixes)
    freeze_cfg = config.get("stage2_freeze", {})
    model.freeze_stage1(
        freeze_backbone=bool(freeze_cfg.get("freeze_backbone", True)),
        freeze_adapter=bool(freeze_cfg.get("freeze_adapter", True)),
        freeze_attribute_extractor=bool(freeze_cfg.get("freeze_attribute_extractor", True)),
        freeze_general_head=bool(freeze_cfg.get("freeze_general_head", True)),
        freeze_attribute_head=bool(freeze_cfg.get("freeze_attribute_head", True)),
    )
    patterns = list(config.get("piaa_adaptation", {}).get("trainable_patterns", ["bridge.", "residual_head."]))
    selected, initial = _set_adaptation_trainable(model, patterns)
    print(
        f"[COBRA-PIAA] support={support_size} users={len(users)} repeats={args.repeats} "
        f"trainable_tensors={len(selected)} trainable_params={sum(p.numel() for _, p in selected)}"
    )

    amp_cfg = config.get("amp", {})
    use_amp = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(amp_cfg.get("dtype", "bfloat16")).lower() == "bfloat16" else torch.float16
    per_epoch_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    support_ensemble_count = int(config.get("piaa_adaptation", {}).get("support_ensemble_count", 1))
    for repeat in range(int(args.repeats)):
        if bool(eval_cfg.get("episodic_common_query", False)):
            episodes = build_episodic_episodes_common_query(
                frame,
                users,
                int(support_size),
                query_support_size=int(eval_cfg.get("episodic_query_support_size", support_size)),
                seed=int(eval_cfg.get("episodic_seed", 42)),
                repeat_index=repeat,
            )
        else:
            episodes = build_episodic_episodes(
                frame,
                users,
                int(support_size),
                seed=int(eval_cfg.get("episodic_seed", 42)),
                repeat_index=repeat,
            )
        dataset = _build_dataset(frame, episodes, config, image_prior_lookup)
        loader = build_episode_dataloader(
            dataset,
            shuffle=False,
            num_workers=int(config.get("data", {}).get("num_workers", 0)),
            persistent_workers=False,
            prefetch_factor=int(config.get("data", {}).get("prefetch_factor", 2)),
        )
        progress = tqdm(
            enumerate(loader, start=1),
            total=len(dataset),
            desc=f"[COBRA-PIAA] s{support_size} repeat {repeat + 1}/{args.repeats}",
            dynamic_ncols=True,
            leave=True,
        )
        for index, raw_episode in progress:
            episode = _episode_to_device(raw_episode, device)
            episode_epoch_rows, episode_best_rows = _adapt_one_episode_with_support_ensemble(
                model,
                episode,
                selected,
                initial,
                config,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            for row in episode_epoch_rows:
                row["repeat"] = repeat
            for row in episode_best_rows:
                row["repeat"] = repeat
            per_epoch_rows.extend(episode_epoch_rows)
            best_rows.extend(episode_best_rows)
            eval_srcc = episode_best_rows[0]["query_srcc"] if episode_best_rows else episode_epoch_rows[-1]["query_srcc"]
            eval_epoch = episode_best_rows[0]["evaluation_epoch"] if episode_best_rows else episode_epoch_rows[-1]["epoch"]
            progress.set_postfix(
                user=str(raw_episode["user_id"]),
                eval_epoch=f"{float(eval_epoch):.1f}",
                eval_srcc=f"{eval_srcc:.4f}",
            )
            del episode, episode_epoch_rows, episode_best_rows
            _release_cuda_cache(device)
        print(
            f"[COBRA-PIAA] s{support_size} repeat={repeat + 1}/{args.repeats} completed "
            f"users={len(dataset)}",
            flush=True,
        )
        del loader, dataset, episodes
        _release_cuda_cache(device)

    metrics = _write_piaa_outputs(
        output_dir,
        support_size=int(support_size),
        per_epoch_rows=per_epoch_rows,
        best_rows=best_rows,
        users=len(users),
        requested_repeats=int(args.repeats),
        completed_repeats=int(args.repeats),
        support_ensemble_count=support_ensemble_count,
        partial=False,
    )
    print(f"[COBRA-PIAA] s{support_size} summary: {metrics}", flush=True)
    return metrics


def _log_piaa_summary(tracker, metrics: dict[str, Any]) -> None:
    support_size = int(metrics.get("support_size", 0))
    prefix = f"piaa/s{support_size}"
    payload = {
        f"{prefix}/macro_srcc": metrics.get("macro_srcc"),
        f"{prefix}/macro_plcc": metrics.get("macro_plcc"),
        f"{prefix}/users": metrics.get("users"),
        f"{prefix}/repeats": metrics.get("repeats"),
        f"{prefix}/episodes": metrics.get("episodes"),
        f"{prefix}/mean_evaluation_epoch": metrics.get("mean_evaluation_epoch"),
        f"{prefix}/mean_best_epoch": metrics.get("mean_best_epoch"),
    }
    for repeat_metric in metrics.get("repeat_metrics", []):
        repeat = int(repeat_metric.get("repeat", -1))
        if repeat >= 0:
            payload[f"{prefix}/repeat_{repeat:02d}_srcc"] = repeat_metric.get("macro_srcc")
            payload[f"{prefix}/repeat_{repeat:02d}_plcc"] = repeat_metric.get("macro_plcc")
    tracker.log(payload)
    tracker.log_summary(
        {
            f"best/{prefix}_macro_srcc": metrics.get("macro_srcc"),
            f"best/{prefix}_macro_plcc": metrics.get("macro_plcc"),
            f"best/{prefix}_mean_evaluation_epoch": metrics.get("mean_evaluation_epoch"),
            f"best/{prefix}_mean_best_epoch": metrics.get("mean_best_epoch"),
            f"artifacts/{prefix}_output_dir": metrics.get("output_dir"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="user-heldout per-user bridge adaptation for COBRA PIAA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--support-sizes", nargs="+", type=int, default=[10, 100])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stage2-init-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--user-source", choices=["user_heldout", "split", "csv"], default=None)
    parser.add_argument("--user-split", default=None, help="Override evaluation.episodic_user_split, e.g. val or test.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-query-size", type=int, default=None)
    parser.add_argument("--max-query-images", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base_config = load_yaml(args.config)
    data_config = load_yaml(args.data_config)
    if args.output_dir:
        base_config.setdefault("experiment", {})["output_dir"] = args.output_dir
    if args.stage2_init_checkpoint:
        base_config.setdefault("experiment", {})["stage2_init_checkpoint"] = args.stage2_init_checkpoint
    if args.epochs is not None:
        base_config.setdefault("piaa_adaptation", {})["epochs"] = int(args.epochs)
    if args.lr is not None:
        base_config.setdefault("piaa_adaptation", {})["lr"] = float(args.lr)
    if args.user_source is not None:
        base_config.setdefault("evaluation", {})["episodic_user_source"] = args.user_source
    if args.user_split is not None:
        base_config.setdefault("evaluation", {})["episodic_user_split"] = str(args.user_split)
    if args.num_workers is not None:
        base_config.setdefault("data", {})["num_workers"] = int(args.num_workers)
    if args.eval_query_size is not None:
        base_config.setdefault("data", {})["eval_query_size"] = int(args.eval_query_size)
    if args.max_query_images is not None:
        base_config.setdefault("data", {})["max_query_images"] = int(args.max_query_images)
    base_config.setdefault("piaa_adaptation", {}).setdefault(
        "trainable_patterns",
        ["preference_encoder.", "bridge.", "residual_head.", "calibration_head."],
    )
    tracking_cfg = base_config.setdefault("tracking", {})
    tracking_cfg.setdefault("enabled", True)
    tracking_cfg.setdefault("run_name", f"{base_config.get('experiment', {}).get('name', 'cobra_piaa_bridge')}_piaa")
    tracking_cfg.setdefault("tags", ["cobra", "piaa", "bridge", "user-heldout"])
    set_seed(int(base_config.get("experiment", {}).get("seed", 42)))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    tracker = init_tracker(base_config, job_type="train_piaa_bridge")
    summaries = []
    try:
        for support_size in args.support_sizes:
            metrics = _run_support_size(base_config, data_config, support_size, args, device)
            summaries.append(metrics)
            _log_piaa_summary(tracker, metrics)
        output_root = Path(base_config["experiment"]["output_dir"]) / "piaa_bridge_summary.json"
        output_root.parent.mkdir(parents=True, exist_ok=True)
        with output_root.open("w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2)
        tracker.log_summary({"artifacts/piaa_bridge_summary": output_root.as_posix()})
        print(f"[COBRA-PIAA] wrote {output_root.as_posix()}")
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
