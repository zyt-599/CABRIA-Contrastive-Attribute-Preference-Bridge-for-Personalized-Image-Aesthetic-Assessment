from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm.auto import tqdm

from cobra.data.image_batch import pad_image_list, prepare_image_tensor
from cobra.data.personalized_dataset import load_personalized_frame
from cobra.models.cobra_model import COBRAStage1Model
from cobra.utils.checkpoint import save_checkpoint
from cobra.utils.common import ensure_dir, load_json, resolve_path
from cobra.utils.factory import build_model_config
from cobra.utils.optimization import EpochLRScheduler
from cobra.utils.tracking import Tracker


class ContrastPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=-1)


class ContrastScoreRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _distributed_state() -> tuple[bool, int, int, int]:
    if not (dist.is_available() and dist.is_initialized()):
        return False, 0, 1, 0
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    return True, rank, world_size, local_rank


def _is_main_process() -> bool:
    distributed, rank, _world_size, _local_rank = _distributed_state()
    return (not distributed) or rank == 0


class ScorePrototypeHead(nn.Module):
    def __init__(self, feature_dim: int, num_bins: int) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_bins, feature_dim) * 0.02)

    def forward(self, features: torch.Tensor, temperature: float) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return torch.matmul(features, prototypes.transpose(0, 1)) / max(float(temperature), 1e-6)


def _score_bin(score: float, score_min: float, score_max: float, num_bins: int) -> int:
    if score_max <= score_min:
        raise ValueError("score_max must be greater than score_min.")
    normalized = (float(score) - score_min) / (score_max - score_min)
    clipped = min(max(normalized, 0.0), 1.0 - 1e-8)
    return int(clipped * num_bins)


def _build_user_bin_pool(frame, num_bins: int, score_min: float, score_max: float) -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    for user_id, user_frame in frame.groupby("user_id"):
        by_bin: dict[int, list[Path]] = {}
        all_paths: list[Path] = []
        for row in user_frame.itertuples(index=False):
            path = resolve_path(row.image_path)
            all_paths.append(path)
            bin_index = _score_bin(float(row.score), score_min=score_min, score_max=score_max, num_bins=num_bins)
            by_bin.setdefault(bin_index, []).append(path)
        pool[str(user_id)] = {"all": all_paths, "by_bin": by_bin}
    return pool


def _build_user_score_pool(frame) -> dict[str, list[tuple[Path, float]]]:
    pool: dict[str, list[tuple[Path, float]]] = {}
    for user_id, user_frame in frame.groupby("user_id"):
        entries: list[tuple[Path, float]] = []
        for row in user_frame.itertuples(index=False):
            entries.append((resolve_path(row.image_path), float(row.score)))
        if len(entries) >= 2:
            pool[str(user_id)] = entries
    return pool


def _build_score_bin_pool(frame, num_bins: int, score_min: float, score_max: float) -> dict[int, list[tuple[Path, float]]]:
    pool: dict[int, list[tuple[Path, float]]] = {bin_index: [] for bin_index in range(num_bins)}
    for row in frame.itertuples(index=False):
        score = float(row.score)
        bin_index = _score_bin(score, score_min=score_min, score_max=score_max, num_bins=num_bins)
        pool[bin_index].append((resolve_path(row.image_path), score))
    return {bin_index: paths for bin_index, paths in pool.items() if paths}


def _sample_score_batch(
    score_pool: dict[int, list[tuple[Path, float]]],
    samples_per_bin: int,
    rng: random.Random,
) -> tuple[list[Path], list[int], list[float], int]:
    paths: list[Path] = []
    labels: list[int] = []
    scores: list[float] = []
    for bin_index in sorted(score_pool):
        bin_entries = score_pool[bin_index]
        if len(bin_entries) >= samples_per_bin:
            selected_entries = rng.sample(bin_entries, samples_per_bin)
        else:
            selected_entries = [rng.choice(bin_entries) for _ in range(samples_per_bin)]
        paths.extend([entry[0] for entry in selected_entries])
        scores.extend([entry[1] for entry in selected_entries])
        labels.extend([bin_index] * len(selected_entries))
    if len(set(labels)) < 2:
        raise RuntimeError("Stage1 contrast score-bin sampling requires at least two non-empty score bins.")
    order = list(range(len(paths)))
    rng.shuffle(order)
    return (
        [paths[index] for index in order],
        [labels[index] for index in order],
        [scores[index] for index in order],
        len(set(labels)),
    )


def _sample_pair(paths: list[Path], rng: random.Random) -> tuple[Path, Path] | None:
    if len(paths) < 2:
        return None
    first, second = rng.sample(paths, 2)
    return first, second


def _sample_user_pair(
    user_entry: dict[str, Any],
    selected_bin: int,
    rng: random.Random,
) -> tuple[Path, Path] | None:
    by_bin: dict[int, list[Path]] = user_entry["by_bin"]
    pair = _sample_pair(by_bin.get(selected_bin, []), rng)
    if pair is not None:
        return pair
    eligible_bins = [paths for paths in by_bin.values() if len(paths) >= 2]
    if eligible_bins:
        return _sample_pair(rng.choice(eligible_bins), rng)
    return _sample_pair(user_entry["all"], rng)


def _sample_user_pair_batch(
    user_pool: dict[str, dict[str, Any]],
    contrast_bins: int,
    users_per_batch: int,
    rng: random.Random,
) -> tuple[list[Path], list[Path], int, int, list[str]]:
    eligible_by_bin = {
        bin_index: [
            user_id
            for user_id, user_entry in user_pool.items()
            if len(user_entry["by_bin"].get(bin_index, [])) >= 2
        ]
        for bin_index in range(contrast_bins)
    }
    usable_bins = [bin_index for bin_index, user_ids in eligible_by_bin.items() if len(user_ids) >= 2]
    if not usable_bins:
        raise RuntimeError("Stage1 contrast user-level sampling requires at least two users with two images in the same score bin.")
    selected_bin = rng.choice(usable_bins)
    user_ids = list(eligible_by_bin[selected_bin])
    rng.shuffle(user_ids)
    user_limit = len(user_ids) if users_per_batch <= 0 else min(users_per_batch, len(user_ids))
    selected_users = user_ids[:user_limit]
    left_paths: list[Path] = []
    right_paths: list[Path] = []
    for user_id in selected_users:
        pair = _sample_pair(user_pool[user_id]["by_bin"][selected_bin], rng)
        if pair is None:
            continue
        left_paths.append(pair[0])
        right_paths.append(pair[1])
    if len(left_paths) < 2:
        raise RuntimeError("Stage1 contrast user-pair sampling produced fewer than two pairs.")
    return left_paths, right_paths, len(left_paths), selected_bin, selected_users[: len(left_paths)]


def _sample_user_retrieval_batch(
    user_score_pool: dict[str, list[tuple[Path, float]]],
    support_size: int,
    query_size: int,
    rng: random.Random,
) -> tuple[list[Path], list[float], int, int, str]:
    min_required = max(2, int(support_size) + int(query_size))
    eligible_users = [user_id for user_id, entries in user_score_pool.items() if len(entries) >= min_required]
    if not eligible_users:
        eligible_users = [user_id for user_id, entries in user_score_pool.items() if len(entries) >= 2]
    if not eligible_users:
        raise RuntimeError("Stage1 contrast retrieval sampling requires at least one user with two images.")
    user_id = rng.choice(eligible_users)
    entries = user_score_pool[user_id]
    support_count = min(max(int(support_size), 1), len(entries) - 1)
    query_count = min(max(int(query_size), 1), len(entries) - support_count)
    if len(entries) >= support_count + query_count:
        selected_entries = rng.sample(entries, support_count + query_count)
    else:
        selected_entries = [rng.choice(entries) for _ in range(support_count + query_count)]
    support_entries = selected_entries[:support_count]
    query_entries = selected_entries[support_count:]
    paths = [entry[0] for entry in support_entries + query_entries]
    scores = [entry[1] for entry in support_entries + query_entries]
    return paths, scores, support_count, query_count, user_id


def _load_image(path: Path, image_cfg: dict[str, Any], train: bool) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as image:
        return prepare_image_tensor(
            image=image.convert("RGB"),
            train=train,
            mean=image_cfg["mean"],
            std=image_cfg["std"],
            fixed_image_size=image_cfg["image_size"],
            use_naflex=image_cfg["use_naflex"],
            preferred_long_side=image_cfg["preferred_long_side"],
            patch_size=image_cfg["patch_size"],
            max_num_patches=image_cfg["max_num_patches"],
        )


def _build_image_batch_cpu(paths: list[Path], image_cfg: dict[str, Any], train: bool) -> dict[str, torch.Tensor]:
    tensors: list[torch.Tensor] = []
    shapes: list[tuple[int, int]] = []
    for path in paths:
        try:
            tensor, shape = _load_image(path, image_cfg, train=train)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise RuntimeError(f"Failed to load contrast image: {path.as_posix()}") from exc
        tensors.append(tensor)
        shapes.append(shape)
    return pad_image_list(
        tensors,
        shapes,
        use_naflex=image_cfg["use_naflex"],
        patch_size=image_cfg["patch_size"],
        max_num_patches=image_cfg["max_num_patches"],
    )


def _move_image_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _concat_image_batches(
    left_batch: dict[str, torch.Tensor],
    right_batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    combined: dict[str, torch.Tensor] = {}
    for key, left_value in left_batch.items():
        right_value = right_batch[key]
        if left_value.dim() != right_value.dim() or left_value.shape[1:] != right_value.shape[1:]:
            return None
        combined[key] = torch.cat([left_value, right_value], dim=0)
    return combined


def _contrast_pair_features(
    model: COBRAStage1Model,
    left_batch: dict[str, torch.Tensor],
    right_batch: dict[str, torch.Tensor],
    feature_source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_batch = _concat_image_batches(left_batch, right_batch)
    if combined_batch is None:
        return (
            _contrast_features(model, left_batch, feature_source=feature_source),
            _contrast_features(model, right_batch, feature_source=feature_source),
        )
    combined_features = _contrast_features(model, combined_batch, feature_source=feature_source)
    batch_size = left_batch["pixel_values"].size(0)
    return combined_features[:batch_size], combined_features[batch_size:]


class ContrastPairIterableDataset(IterableDataset[dict[str, Any]]):
    def __init__(
        self,
        user_pool: dict[str, dict[str, Any]],
        image_cfg: dict[str, Any],
        contrast_bins: int,
        users_per_batch: int,
        steps_per_epoch: int,
        seed: int,
        epoch: int,
    ) -> None:
        super().__init__()
        self.user_pool = user_pool
        self.image_cfg = image_cfg
        self.contrast_bins = contrast_bins
        self.users_per_batch = users_per_batch
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = epoch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        rng = random.Random(self.seed + self.epoch * 1_000_003 + worker_id)
        for _step in range(worker_id, self.steps_per_epoch, num_workers):
            left_paths, right_paths, num_pairs, selected_bin, user_ids = _sample_user_pair_batch(
                user_pool=self.user_pool,
                contrast_bins=self.contrast_bins,
                users_per_batch=self.users_per_batch,
                rng=rng,
            )
            yield {
                "left_images": _build_image_batch_cpu(left_paths, self.image_cfg, train=True),
                "right_images": _build_image_batch_cpu(right_paths, self.image_cfg, train=True),
                "num_pairs": num_pairs,
                "selected_bin": selected_bin,
                "user_ids": user_ids,
            }


class ContrastHybridIterableDataset(IterableDataset[dict[str, Any]]):
    def __init__(
        self,
        user_pool: dict[str, dict[str, Any]],
        score_pool: dict[int, list[tuple[Path, float]]],
        retrieval_pool: dict[str, list[tuple[Path, float]]] | None,
        image_cfg: dict[str, Any],
        contrast_bins: int,
        users_per_batch: int,
        samples_per_bin: int,
        retrieval_support_size: int,
        retrieval_support_sizes: list[int] | None,
        retrieval_query_size: int,
        steps_per_epoch: int,
        seed: int,
        epoch: int,
    ) -> None:
        super().__init__()
        self.user_pool = user_pool
        self.score_pool = score_pool
        self.retrieval_pool = retrieval_pool or {}
        self.image_cfg = image_cfg
        self.contrast_bins = contrast_bins
        self.users_per_batch = users_per_batch
        self.samples_per_bin = samples_per_bin
        self.retrieval_support_size = retrieval_support_size
        self.retrieval_support_sizes = [max(int(size), 1) for size in (retrieval_support_sizes or [retrieval_support_size])]
        self.retrieval_query_size = retrieval_query_size
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = epoch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        rng = random.Random(self.seed + self.epoch * 1_000_003 + worker_id)
        for _step in range(worker_id, self.steps_per_epoch, num_workers):
            left_paths, right_paths, num_pairs, selected_bin, user_ids = _sample_user_pair_batch(
                user_pool=self.user_pool,
                contrast_bins=self.contrast_bins,
                users_per_batch=self.users_per_batch,
                rng=rng,
            )
            score_paths, labels, scores, num_bins = _sample_score_batch(
                score_pool=self.score_pool,
                samples_per_bin=self.samples_per_bin,
                rng=rng,
            )
            item = {
                "left_images": _build_image_batch_cpu(left_paths, self.image_cfg, train=True),
                "right_images": _build_image_batch_cpu(right_paths, self.image_cfg, train=True),
                "num_pairs": num_pairs,
                "selected_bin": selected_bin,
                "user_ids": user_ids,
                "score_images": _build_image_batch_cpu(score_paths, self.image_cfg, train=True),
                "score_labels": torch.tensor(labels, dtype=torch.long),
                "score_scores": torch.tensor(scores, dtype=torch.float32),
                "num_score_images": len(score_paths),
                "num_bins": num_bins,
            }
            if self.retrieval_pool:
                retrieval_support_size = rng.choice(self.retrieval_support_sizes)
                retrieval_paths, retrieval_scores, support_count, query_count, retrieval_user_id = _sample_user_retrieval_batch(
                    user_score_pool=self.retrieval_pool,
                    support_size=retrieval_support_size,
                    query_size=self.retrieval_query_size,
                    rng=rng,
                )
                item.update(
                    {
                        "retrieval_images": _build_image_batch_cpu(retrieval_paths, self.image_cfg, train=True),
                        "retrieval_scores": torch.tensor(retrieval_scores, dtype=torch.float32),
                        "retrieval_support_size": support_count,
                        "retrieval_query_size": query_count,
                        "retrieval_user_id": retrieval_user_id,
                    }
                )
            yield item


class ContrastScoreIterableDataset(IterableDataset[dict[str, Any]]):
    def __init__(
        self,
        score_pool: dict[int, list[tuple[Path, float]]],
        image_cfg: dict[str, Any],
        samples_per_bin: int,
        steps_per_epoch: int,
        seed: int,
        epoch: int,
    ) -> None:
        super().__init__()
        self.score_pool = score_pool
        self.image_cfg = image_cfg
        self.samples_per_bin = samples_per_bin
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = epoch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        rng = random.Random(self.seed + self.epoch * 1_000_003 + worker_id)
        for _step in range(worker_id, self.steps_per_epoch, num_workers):
            paths, labels, scores, num_bins = _sample_score_batch(
                score_pool=self.score_pool,
                samples_per_bin=self.samples_per_bin,
                rng=rng,
            )
            yield {
                "images": _build_image_batch_cpu(paths, self.image_cfg, train=True),
                "labels": torch.tensor(labels, dtype=torch.long),
                "scores": torch.tensor(scores, dtype=torch.float32),
                "num_images": len(paths),
                "num_bins": num_bins,
            }


def _seed_contrast_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    torch.manual_seed(worker_seed + worker_id)


def _paired_user_info_nce_loss(left_features: torch.Tensor, right_features: torch.Tensor, temperature: float) -> torch.Tensor:
    if left_features.shape != right_features.shape:
        raise ValueError("left_features and right_features must have the same shape.")
    batch_size = left_features.size(0)
    if batch_size < 2:
        return left_features.sum() * 0.0
    left_features = F.normalize(left_features, dim=-1)
    right_features = F.normalize(right_features, dim=-1)
    diagonal_mask = ~torch.eye(batch_size, dtype=torch.bool, device=left_features.device)

    left_left = torch.matmul(left_features, left_features.transpose(0, 1))
    left_right = torch.matmul(left_features, right_features.transpose(0, 1))
    right_right = torch.matmul(right_features, right_features.transpose(0, 1))
    right_left = torch.matmul(right_features, left_features.transpose(0, 1))

    left_positive = left_right.diagonal().unsqueeze(1)
    right_positive = right_left.diagonal().unsqueeze(1)
    left_logits = torch.cat(
        (
            left_positive,
            left_left.masked_select(diagonal_mask).view(batch_size, batch_size - 1),
            left_right.masked_select(diagonal_mask).view(batch_size, batch_size - 1),
        ),
        dim=1,
    )
    right_logits = torch.cat(
        (
            right_positive,
            right_right.masked_select(diagonal_mask).view(batch_size, batch_size - 1),
            right_left.masked_select(diagonal_mask).view(batch_size, batch_size - 1),
        ),
        dim=1,
    )
    labels = torch.zeros(batch_size, dtype=torch.long, device=left_features.device)
    loss_left = F.cross_entropy(left_logits / temperature, labels)
    loss_right = F.cross_entropy(right_logits / temperature, labels)
    return 0.5 * (loss_left + loss_right)


def _supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if features.size(0) != labels.size(0):
        raise ValueError("features and labels must have the same batch dimension.")
    if features.size(0) < 2:
        return features.sum() * 0.0
    features = F.normalize(features, dim=-1)
    labels = labels.view(-1, 1)
    self_mask = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    positive_mask = torch.eq(labels, labels.transpose(0, 1)).to(features.device) & ~self_mask
    logits = torch.matmul(features, features.transpose(0, 1)) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = ~self_mask
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return features.sum() * 0.0
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def _soft_ordinal_contrastive_loss(
    features: torch.Tensor,
    scores: torch.Tensor,
    temperature: float,
    sigma: float,
) -> torch.Tensor:
    if features.size(0) != scores.size(0):
        raise ValueError("features and scores must have the same batch dimension.")
    if features.size(0) < 2:
        return features.sum() * 0.0
    features = F.normalize(features, dim=-1)
    scores = scores.view(-1, 1)
    self_mask = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    score_distance = torch.abs(scores - scores.transpose(0, 1))
    target_weights = torch.exp(-score_distance / max(float(sigma), 1e-6)).masked_fill(self_mask, 0.0)
    target_weights = target_weights / target_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    logits = torch.matmul(features, features.transpose(0, 1)) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return -(target_weights * log_prob).sum(dim=1).mean()


def _rank_n_contrast_loss(
    features: torch.Tensor,
    scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if features.size(0) != scores.size(0):
        raise ValueError("features and scores must have the same batch dimension.")
    batch_size = features.size(0)
    if batch_size < 3:
        return features.sum() * 0.0

    features = F.normalize(features, dim=-1)
    logits = torch.matmul(features, features.transpose(0, 1)) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    scores = scores.view(-1)
    distances = torch.abs(scores.view(-1, 1) - scores.view(1, -1))
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    candidate_mask = ~self_mask

    # For anchor i and positive j, all samples k with d(i,k) >= d(i,j) are negatives.
    denom_mask = distances.unsqueeze(1) >= distances.unsqueeze(2)
    denom_mask = denom_mask & candidate_mask.unsqueeze(1)
    positives = candidate_mask
    log_denominator = torch.logsumexp(logits.unsqueeze(1).masked_fill(~denom_mask, float("-inf")), dim=2)
    losses = log_denominator - logits
    return losses[positives].mean()


def _normalize_scores(scores: torch.Tensor, score_min: float, score_max: float) -> torch.Tensor:
    return ((scores - score_min) / max(score_max - score_min, 1e-6)).clamp(0.0, 1.0)


def _standardize_residuals(residuals: torch.Tensor, clamp_abs: float) -> torch.Tensor:
    centered = residuals.float() - residuals.float().mean()
    scaled = centered / centered.std(unbiased=False).clamp_min(1e-6)
    if clamp_abs > 0.0:
        scaled = scaled.clamp(min=-float(clamp_abs), max=float(clamp_abs))
    return scaled


def _prepare_residual_targets(
    residuals: torch.Tensor,
    residual_target_scale: float,
    residual_standardize_clamp: float,
    residual_standardize: bool,
) -> torch.Tensor:
    targets = residuals.float() * float(residual_target_scale)
    if residual_standardize:
        return _standardize_residuals(targets, clamp_abs=float(residual_standardize_clamp))
    if residual_standardize_clamp > 0.0:
        targets = targets.clamp(min=-float(residual_standardize_clamp), max=float(residual_standardize_clamp))
    return targets


@torch.no_grad()
def _stage1_base_scores(
    base_model: COBRAStage1Model,
    images: dict[str, torch.Tensor],
) -> torch.Tensor:
    base_model.eval()
    return base_model(images)["score"].detach().float()


def _pairwise_score_ranking_loss(
    predicted_scores: torch.Tensor,
    target_scores: torch.Tensor,
    min_delta: float,
    temperature: float,
) -> torch.Tensor:
    score_diff = target_scores.view(-1, 1) - target_scores.view(1, -1)
    pred_diff = predicted_scores.view(-1, 1) - predicted_scores.view(1, -1)
    valid = torch.abs(score_diff) >= min_delta
    if not torch.any(valid):
        return predicted_scores.sum() * 0.0
    target_sign = torch.sign(score_diff[valid])
    # If target_i > target_j, then pred_i should be larger than pred_j.
    ordered_pred_diff = -target_sign * pred_diff[valid]
    weights = torch.abs(score_diff[valid]) / torch.abs(score_diff[valid]).mean().clamp_min(1e-6)
    return (F.softplus(ordered_pred_diff / max(float(temperature), 1e-6)) * weights).mean()


def _spearman_corr(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.numel() < 2:
        return predicted.new_zeros(())
    pred_rank = torch.argsort(torch.argsort(predicted.float())).float()
    target_rank = torch.argsort(torch.argsort(target.float())).float()
    pred_centered = pred_rank - pred_rank.mean()
    target_centered = target_rank - target_rank.mean()
    denom = pred_centered.norm() * target_centered.norm()
    if torch.isclose(denom, denom.new_zeros(())):
        return predicted.new_zeros(())
    return (pred_centered * target_centered).sum() / denom.clamp_min(1e-6)


def _support_residual_retrieval_loss(
    support_features: torch.Tensor,
    query_features: torch.Tensor,
    support_residuals: torch.Tensor,
    query_residuals: torch.Tensor,
    temperature: float,
    target_sigma: float,
    value_weight: float,
    rank_weight: float,
    rank_min_delta: float,
    rank_temperature: float,
    query_base_scores: torch.Tensor | None = None,
    query_scores: torch.Tensor | None = None,
    score_value_weight: float = 0.0,
    score_rank_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if support_features.numel() == 0 or query_features.numel() == 0:
        zero = support_features.sum() * 0.0
        return zero, {
            "mae": zero.detach(),
            "pair_acc": zero.detach(),
            "srcc": zero.detach(),
            "score_mae": zero.detach(),
            "score_pair_acc": zero.detach(),
            "score_srcc": zero.detach(),
        }
    support_features = F.normalize(support_features, dim=-1)
    query_features = F.normalize(query_features, dim=-1)
    support_residuals = support_residuals.float().view(-1)
    query_residuals = query_residuals.float().view(-1)

    logits = torch.matmul(query_features, support_features.transpose(0, 1)) / max(float(temperature), 1e-6)
    log_probs = F.log_softmax(logits, dim=-1)
    residual_distance = torch.abs(query_residuals.view(-1, 1) - support_residuals.view(1, -1))
    target_weights = torch.exp(-residual_distance / max(float(target_sigma), 1e-6))
    target_weights = target_weights / target_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    distribution_loss = -(target_weights.detach() * log_probs).sum(dim=1).mean()

    weights = torch.softmax(logits, dim=-1)
    predicted_residuals = torch.matmul(weights, support_residuals.view(-1, 1)).squeeze(-1)
    value_loss = F.smooth_l1_loss(predicted_residuals, query_residuals)
    rank_loss = _pairwise_score_ranking_loss(
        predicted_residuals,
        query_residuals,
        min_delta=rank_min_delta,
        temperature=rank_temperature,
    )
    loss = distribution_loss + float(value_weight) * value_loss + float(rank_weight) * rank_loss
    if query_base_scores is not None and query_scores is not None and (score_value_weight > 0.0 or score_rank_weight > 0.0):
        predicted_scores = query_base_scores.float().view(-1) + predicted_residuals
        target_scores = query_scores.float().view(-1)
        score_value_loss = F.smooth_l1_loss(predicted_scores, target_scores)
        score_rank_loss = _pairwise_score_ranking_loss(
            predicted_scores,
            target_scores,
            min_delta=rank_min_delta,
            temperature=rank_temperature,
        )
        loss = loss + float(score_value_weight) * score_value_loss + float(score_rank_weight) * score_rank_loss
        score_diff = target_scores.view(-1, 1) - target_scores.view(1, -1)
        score_pred_diff = predicted_scores.view(-1, 1) - predicted_scores.view(1, -1)
        score_valid = torch.abs(score_diff) >= rank_min_delta
        score_pair_acc = (
            (torch.sign(score_diff[score_valid]) == torch.sign(score_pred_diff[score_valid])).float().mean()
            if torch.any(score_valid)
            else predicted_scores.new_zeros(())
        )
        score_mae = torch.mean(torch.abs(predicted_scores.detach() - target_scores.detach()))
        score_srcc = _spearman_corr(predicted_scores.detach(), target_scores.detach()).detach()
    else:
        zero = loss.detach() * 0.0
        score_pair_acc = zero
        score_mae = zero
        score_srcc = zero

    diff = query_residuals.view(-1, 1) - query_residuals.view(1, -1)
    pred_diff = predicted_residuals.view(-1, 1) - predicted_residuals.view(1, -1)
    valid = torch.abs(diff) >= rank_min_delta
    pair_acc = (
        (torch.sign(diff[valid]) == torch.sign(pred_diff[valid])).float().mean()
        if torch.any(valid)
        else predicted_residuals.new_zeros(())
    )
    metrics = {
        "mae": torch.mean(torch.abs(predicted_residuals.detach() - query_residuals.detach())),
        "pair_acc": pair_acc.detach(),
        "srcc": _spearman_corr(predicted_residuals.detach(), query_residuals.detach()).detach(),
        "score_mae": score_mae.detach(),
        "score_pair_acc": score_pair_acc.detach(),
        "score_srcc": score_srcc.detach(),
    }
    return loss, metrics


def _prototype_expected_scores(logits: torch.Tensor) -> torch.Tensor:
    centers = torch.linspace(0.0, 1.0, logits.size(1), device=logits.device, dtype=logits.dtype)
    probabilities = F.softmax(logits, dim=-1)
    return torch.sum(probabilities * centers.view(1, -1), dim=-1)


def _paired_retrieval_metrics(left_features: torch.Tensor, right_features: torch.Tensor) -> dict[str, float]:
    left_features = F.normalize(left_features, dim=-1)
    right_features = F.normalize(right_features, dim=-1)
    similarities = torch.matmul(left_features, right_features.transpose(0, 1))
    batch_size = similarities.size(0)
    targets = torch.arange(batch_size, device=similarities.device)
    left_top1 = similarities.argmax(dim=1).eq(targets).float().mean()
    right_top1 = similarities.argmax(dim=0).eq(targets).float().mean()
    left_ranks = torch.argsort(torch.argsort(-similarities, dim=1), dim=1)[targets, targets].float() + 1.0
    right_order = torch.argsort(torch.argsort(-similarities.transpose(0, 1), dim=1), dim=1)
    right_ranks = right_order[targets, targets].float() + 1.0
    return {
        "top1": float(0.5 * (left_top1.item() + right_top1.item())),
        "mean_rank": float(0.5 * (left_ranks.mean().item() + right_ranks.mean().item())),
    }


def _score_retrieval_metrics(features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    features = F.normalize(features, dim=-1)
    similarities = torch.matmul(features, features.transpose(0, 1))
    batch_size = similarities.size(0)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    similarities = similarities.masked_fill(self_mask, float("-inf"))
    nearest = similarities.argmax(dim=1)
    top1 = labels[nearest].eq(labels).float().mean()

    same_label = labels.view(-1, 1).eq(labels.view(1, -1)) & ~self_mask
    ranks = torch.argsort(torch.argsort(-similarities, dim=1), dim=1).float() + 1.0
    positive_ranks = ranks.masked_fill(~same_label, float("inf")).min(dim=1).values
    valid = torch.isfinite(positive_ranks)
    mean_rank = positive_ranks[valid].mean() if torch.any(valid) else torch.tensor(0.0, device=features.device)
    return {"top1": float(top1.item()), "mean_rank": float(mean_rank.item())}


@torch.no_grad()
def _evaluate_contrast_proxy(
    model: COBRAStage1Model,
    predictor: ContrastPredictor,
    user_pool: dict[str, dict[str, Any]],
    image_cfg: dict[str, Any],
    device: torch.device,
    contrast_bins: int,
    users_per_batch: int,
    temperature: float,
    steps: int,
    seed: int,
    feature_source: str,
) -> dict[str, float]:
    model.eval()
    predictor.eval()
    rng = random.Random(seed)
    losses: list[float] = []
    accuracies: list[float] = []
    ranks: list[float] = []
    for _ in range(max(int(steps), 1)):
        left_paths, right_paths, _num_pairs, _selected_bin, _user_ids = _sample_user_pair_batch(
            user_pool=user_pool,
            contrast_bins=contrast_bins,
            users_per_batch=users_per_batch,
            rng=rng,
        )
        left_batch = _move_image_batch_to_device(
            _build_image_batch_cpu(left_paths, image_cfg=image_cfg, train=False),
            device,
        )
        right_batch = _move_image_batch_to_device(
            _build_image_batch_cpu(right_paths, image_cfg=image_cfg, train=False),
            device,
        )
        left_raw, right_raw = _contrast_pair_features(model, left_batch, right_batch, feature_source=feature_source)
        left_projected = predictor(left_raw)
        right_projected = predictor(right_raw)
        loss = _paired_user_info_nce_loss(left_projected, right_projected, temperature=temperature)
        metrics = _paired_retrieval_metrics(left_projected, right_projected)
        losses.append(float(loss.item()))
        accuracies.append(metrics["top1"])
        ranks.append(metrics["mean_rank"])
    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "top1": float(sum(accuracies) / max(len(accuracies), 1)),
        "mean_rank": float(sum(ranks) / max(len(ranks), 1)),
    }


@torch.no_grad()
def _evaluate_score_proxy(
    model: COBRAStage1Model,
    base_model: COBRAStage1Model,
    predictor: ContrastPredictor,
    score_head: ContrastScoreRegressor,
    residual_head: ContrastScoreRegressor,
    prototype_head: ScorePrototypeHead,
    score_pool: dict[int, list[tuple[Path, float]]],
    image_cfg: dict[str, Any],
    device: torch.device,
    samples_per_bin: int,
    temperature: float,
    prototype_temperature: float,
    rnc_temperature: float,
    ordinal_sigma: float,
    score_min: float,
    score_max: float,
    steps: int,
    seed: int,
    feature_source: str,
    contrast_objective: str,
    residual_target_scale: float,
    residual_standardize_clamp: float,
    residual_standardize: bool,
) -> dict[str, float]:
    model.eval()
    predictor.eval()
    score_head.eval()
    residual_head.eval()
    prototype_head.eval()
    rng = random.Random(seed)
    losses: list[float] = []
    accuracies: list[float] = []
    ranks: list[float] = []
    maes: list[float] = []
    head_maes: list[float] = []
    pair_accs: list[float] = []
    prototype_accs: list[float] = []
    residual_maes: list[float] = []
    residual_pair_accs: list[float] = []
    residual_proxy_losses: list[float] = []
    for _ in range(max(int(steps), 1)):
        paths, labels, scores, _num_bins = _sample_score_batch(
            score_pool=score_pool,
            samples_per_bin=samples_per_bin,
            rng=rng,
        )
        image_batch = _move_image_batch_to_device(
            _build_image_batch_cpu(paths, image_cfg=image_cfg, train=False),
            device,
        )
        label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        score_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
        normalized_scores = _normalize_scores(score_tensor, score_min=score_min, score_max=score_max)
        base_scores = _stage1_base_scores(base_model, image_batch)
        residual_targets = _prepare_residual_targets(
            score_tensor - base_scores,
            residual_target_scale=float(residual_target_scale),
            residual_standardize_clamp=float(residual_standardize_clamp),
            residual_standardize=bool(residual_standardize),
        )
        raw_features = _contrast_features(model, image_batch, feature_source=feature_source)
        projected_features = predictor(raw_features)
        head_scores = score_head(projected_features).sigmoid()
        residual_predictions = residual_head(projected_features)
        prototype_logits = prototype_head(projected_features, temperature=prototype_temperature)
        predicted_scores = _prototype_expected_scores(prototype_logits)
        loss = (
            _rank_n_contrast_loss(projected_features, score_tensor, temperature=rnc_temperature)
            if contrast_objective == "rnc_rank_score_contrast"
            else _soft_ordinal_contrastive_loss(
                projected_features,
                score_tensor,
                temperature=temperature,
                sigma=ordinal_sigma,
            )
        )
        metrics = _score_retrieval_metrics(projected_features, label_tensor)
        losses.append(float(loss.item()))
        accuracies.append(metrics["top1"])
        ranks.append(metrics["mean_rank"])
        maes.append(float(torch.mean(torch.abs(predicted_scores - normalized_scores)).item()))
        head_maes.append(float(torch.mean(torch.abs(head_scores - normalized_scores)).item()))
        residual_maes.append(float(torch.mean(torch.abs(residual_predictions - residual_targets)).item()))
        residual_proxy_losses.append(
            float(
                _soft_ordinal_contrastive_loss(
                    projected_features,
                    residual_targets,
                    temperature=temperature,
                    sigma=ordinal_sigma,
                ).item()
            )
        )
        prototype_accs.append(float(prototype_logits.argmax(dim=1).eq(label_tensor).float().mean().item()))
        score_diff = normalized_scores.view(-1, 1) - normalized_scores.view(1, -1)
        pred_diff = predicted_scores.view(-1, 1) - predicted_scores.view(1, -1)
        valid = torch.abs(score_diff) >= 0.1
        if torch.any(valid):
            pair_accs.append(float((torch.sign(score_diff[valid]) == torch.sign(pred_diff[valid])).float().mean().item()))
        residual_diff = residual_targets.view(-1, 1) - residual_targets.view(1, -1)
        residual_pred_diff = residual_predictions.view(-1, 1) - residual_predictions.view(1, -1)
        residual_valid = torch.abs(residual_diff) >= 0.25
        if torch.any(residual_valid):
            residual_pair_accs.append(
                float((torch.sign(residual_diff[residual_valid]) == torch.sign(residual_pred_diff[residual_valid])).float().mean().item())
            )
    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "top1": float(sum(accuracies) / max(len(accuracies), 1)),
        "mean_rank": float(sum(ranks) / max(len(ranks), 1)),
        "score_mae": float(sum(maes) / max(len(maes), 1)),
        "score_head_mae": float(sum(head_maes) / max(len(head_maes), 1)),
        "pair_acc": float(sum(pair_accs) / max(len(pair_accs), 1)),
        "prototype_acc": float(sum(prototype_accs) / max(len(prototype_accs), 1)),
        "residual_mae": float(sum(residual_maes) / max(len(residual_maes), 1)),
        "residual_pair_acc": float(sum(residual_pair_accs) / max(len(residual_pair_accs), 1)),
        "residual_proxy_loss": float(sum(residual_proxy_losses) / max(len(residual_proxy_losses), 1)),
    }


@torch.no_grad()
def _evaluate_support_retrieval_proxy(
    model: COBRAStage1Model,
    base_model: COBRAStage1Model,
    predictor: ContrastPredictor,
    user_score_pool: dict[str, list[tuple[Path, float]]],
    image_cfg: dict[str, Any],
    device: torch.device,
    support_size: int,
    query_size: int,
    temperature: float,
    target_sigma: float,
    value_weight: float,
    rank_weight: float,
    rank_min_delta: float,
    rank_temperature: float,
    residual_target_scale: float,
    residual_standardize_clamp: float,
    residual_standardize: bool,
    score_value_weight: float,
    score_rank_weight: float,
    steps: int,
    seed: int,
    feature_source: str,
) -> dict[str, float]:
    model.eval()
    base_model.eval()
    predictor.eval()
    rng = random.Random(seed)
    losses: list[float] = []
    direct_losses: list[float] = []
    maes: list[float] = []
    pair_accs: list[float] = []
    srccs: list[float] = []
    direct_maes: list[float] = []
    direct_pair_accs: list[float] = []
    direct_srccs: list[float] = []
    for _ in range(max(int(steps), 1)):
        paths, scores, support_count, query_count, _user_id = _sample_user_retrieval_batch(
            user_score_pool=user_score_pool,
            support_size=support_size,
            query_size=query_size,
            rng=rng,
        )
        image_batch = _move_image_batch_to_device(
            _build_image_batch_cpu(paths, image_cfg=image_cfg, train=False),
            device,
        )
        score_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
        raw_features = _contrast_features(model, image_batch, feature_source=feature_source)
        projected_features = predictor(raw_features)
        base_scores = _stage1_base_scores(base_model, image_batch)
        residual_targets = _prepare_residual_targets(
            score_tensor - base_scores,
            residual_target_scale=float(residual_target_scale),
            residual_standardize_clamp=float(residual_standardize_clamp),
            residual_standardize=bool(residual_standardize),
        )
        support_slice = slice(0, support_count)
        query_slice = slice(support_count, support_count + query_count)
        projected_loss, projected_metrics = _support_residual_retrieval_loss(
            support_features=projected_features[support_slice],
            query_features=projected_features[query_slice],
            support_residuals=residual_targets[support_slice],
            query_residuals=residual_targets[query_slice],
            temperature=temperature,
            target_sigma=target_sigma,
            value_weight=value_weight,
            rank_weight=rank_weight,
            rank_min_delta=rank_min_delta,
            rank_temperature=rank_temperature,
            query_base_scores=base_scores[query_slice],
            query_scores=score_tensor[query_slice],
            score_value_weight=score_value_weight,
            score_rank_weight=score_rank_weight,
        )
        direct_loss, direct_metrics = _support_residual_retrieval_loss(
            support_features=raw_features[support_slice],
            query_features=raw_features[query_slice],
            support_residuals=residual_targets[support_slice],
            query_residuals=residual_targets[query_slice],
            temperature=temperature,
            target_sigma=target_sigma,
            value_weight=value_weight,
            rank_weight=rank_weight,
            rank_min_delta=rank_min_delta,
            rank_temperature=rank_temperature,
            query_base_scores=base_scores[query_slice],
            query_scores=score_tensor[query_slice],
            score_value_weight=score_value_weight,
            score_rank_weight=score_rank_weight,
        )
        losses.append(float(projected_loss.item()))
        direct_losses.append(float(direct_loss.item()))
        maes.append(float(projected_metrics["mae"].item()))
        pair_accs.append(float(projected_metrics["pair_acc"].item()))
        srccs.append(float(projected_metrics["srcc"].item()))
        direct_maes.append(float(direct_metrics["mae"].item()))
        direct_pair_accs.append(float(direct_metrics["pair_acc"].item()))
        direct_srccs.append(float(direct_metrics["srcc"].item()))
    return {
        "support_retrieval_loss": float(sum(losses) / max(len(losses), 1)),
        "support_retrieval_direct_loss": float(sum(direct_losses) / max(len(direct_losses), 1)),
        "support_retrieval_mae": float(sum(maes) / max(len(maes), 1)),
        "support_retrieval_pair_acc": float(sum(pair_accs) / max(len(pair_accs), 1)),
        "support_retrieval_srcc": float(sum(srccs) / max(len(srccs), 1)),
        "support_retrieval_direct_mae": float(sum(direct_maes) / max(len(direct_maes), 1)),
        "support_retrieval_direct_pair_acc": float(sum(direct_pair_accs) / max(len(direct_pair_accs), 1)),
        "support_retrieval_direct_srcc": float(sum(direct_srccs) / max(len(direct_srccs), 1)),
    }


def _freeze_for_contrast(model: COBRAStage1Model, config: dict[str, Any]) -> None:
    freeze_cfg = config.get("contrast_freeze", {})
    train_adapter = bool(freeze_cfg.get("train_adapter", True))
    train_attribute_extractor = bool(freeze_cfg.get("train_attribute_extractor", True))
    train_attribute_head = bool(freeze_cfg.get("train_attribute_head", True))
    train_projection_head = bool(freeze_cfg.get("train_projection_head", False))
    train_backbone = bool(freeze_cfg.get("train_backbone", False))

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.backbone.backbone.parameters():
        parameter.requires_grad = train_backbone
    for parameter in model.backbone.adapter.parameters():
        parameter.requires_grad = train_adapter
    for parameter in model.backbone.output_norm.parameters():
        parameter.requires_grad = train_adapter
    for parameter in model.attribute_extractor.parameters():
        parameter.requires_grad = train_attribute_extractor
    for parameter in model.attribute_head.parameters():
        parameter.requires_grad = train_attribute_head
    for parameter in model.projection_head.parameters():
        parameter.requires_grad = train_projection_head
    for parameter in model.general_head.parameters():
        parameter.requires_grad = False

    print(
        "[COBRA] Stage1 contrast freeze policy: "
        f"train_backbone={train_backbone}, train_adapter={train_adapter}, "
        f"train_attribute_extractor={train_attribute_extractor}, "
        f"train_attribute_head={train_attribute_head}, train_projection_head={train_projection_head}, "
        "train_general_head=False"
    )


def _contrast_features(
    model: COBRAStage1Model,
    images: dict[str, torch.Tensor],
    feature_source: str,
) -> torch.Tensor:
    outputs = model(images)
    source = str(feature_source).lower()
    if source == "attribute_tokens":
        pooled = outputs["attribute_tokens"].mean(dim=1)
    elif source == "attribute_pooled":
        pooled = outputs["attribute_pooled"]
    elif source == "projected_tokens":
        pooled = outputs["projected_tokens"].mean(dim=1)
    else:
        raise ValueError(f"Unsupported contrast feature_source: {feature_source!r}.")
    return F.normalize(pooled, dim=-1)


def _save_contrast_checkpoint(
    path: Path,
    model: COBRAStage1Model,
    optimizer: torch.optim.Optimizer,
    scheduler: EpochLRScheduler,
    epoch: int,
    epoch_loss: float,
    proxy_metrics: dict[str, float],
    giaa_checkpoint: str | Path,
    checkpoint_policy: str,
    contrast_objective: str,
) -> None:
    save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler.state_dict(),
        epoch=epoch,
        metrics={
            "contrast_loss": epoch_loss,
            "val_proxy_loss": proxy_metrics["loss"],
            "val_proxy_top1": proxy_metrics["top1"],
            "val_proxy_mean_rank": proxy_metrics["mean_rank"],
            "val_proxy_score_mae": proxy_metrics.get("score_mae", 0.0),
            "val_proxy_score_head_mae": proxy_metrics.get("score_head_mae", 0.0),
            "val_proxy_pair_acc": proxy_metrics.get("pair_acc", 0.0),
            "val_proxy_prototype_acc": proxy_metrics.get("prototype_acc", 0.0),
            "val_proxy_residual_mae": proxy_metrics.get("residual_mae", 0.0),
            "val_proxy_residual_pair_acc": proxy_metrics.get("residual_pair_acc", 0.0),
            "val_proxy_residual_proxy_loss": proxy_metrics.get("residual_proxy_loss", 0.0),
            "val_proxy_support_retrieval_loss": proxy_metrics.get("support_retrieval_loss", 0.0),
            "val_proxy_support_retrieval_srcc": proxy_metrics.get("support_retrieval_srcc", 0.0),
            "val_proxy_support_retrieval_pair_acc": proxy_metrics.get("support_retrieval_pair_acc", 0.0),
            "val_proxy_support_retrieval_direct_srcc": proxy_metrics.get("support_retrieval_direct_srcc", 0.0),
            "val_proxy_support_retrieval_direct_pair_acc": proxy_metrics.get("support_retrieval_direct_pair_acc", 0.0),
        },
        extra={
            "giaa_checkpoint": resolve_path(giaa_checkpoint).as_posix(),
            "contrast_objective": contrast_objective,
            "checkpoint_policy": checkpoint_policy,
        },
    )


def train_stage1_contrast(config: dict, data_config: dict, device: torch.device, tracker: Tracker | None = None) -> Path:
    distributed, rank, world_size, local_rank = _distributed_state()
    is_main = rank == 0
    model_core = COBRAStage1Model(build_model_config(config)).to(device)
    base_model = COBRAStage1Model(build_model_config(config)).to(device)
    giaa_checkpoint = config["experiment"]["giaa_checkpoint"]
    checkpoint = torch.load(resolve_path(giaa_checkpoint), map_location="cpu")
    model_core.load_state_dict(checkpoint["model"], strict=True)
    base_model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    base_model.eval()
    if is_main:
        print(f"[COBRA] Stage1 contrast initialized from GIAA checkpoint: {resolve_path(giaa_checkpoint).as_posix()}")

    _freeze_for_contrast(model_core, config)
    contrast_cfg = config.get("contrast", {})
    contrast_objective = str(contrast_cfg.get("objective", "score_supervised_contrast"))
    feature_source = str(contrast_cfg.get("feature_source", "attribute_tokens")).lower()
    predictor_input_dim = (
        int(config["backbone"]["embed_dim"])
        if feature_source in {"attribute_tokens", "attribute_pooled"}
        else int(config["model"]["projection_dim"])
    )
    predictor = ContrastPredictor(
        input_dim=predictor_input_dim,
        hidden_dim=int(contrast_cfg.get("predictor_hidden_dim", config["model"]["projection_dim"])),
        output_dim=int(contrast_cfg.get("predictor_dim", config["model"]["projection_dim"])),
        dropout=float(contrast_cfg.get("predictor_dropout", 0.0)),
    ).to(device)
    score_head = ContrastScoreRegressor(
        input_dim=int(contrast_cfg.get("predictor_dim", config["model"]["projection_dim"])),
        hidden_dim=int(contrast_cfg.get("score_head_hidden_dim", config["model"]["projection_dim"])),
        dropout=float(contrast_cfg.get("score_head_dropout", 0.0)),
    ).to(device)
    residual_head = ContrastScoreRegressor(
        input_dim=int(contrast_cfg.get("predictor_dim", config["model"]["projection_dim"])),
        hidden_dim=int(contrast_cfg.get("residual_head_hidden_dim", contrast_cfg.get("score_head_hidden_dim", config["model"]["projection_dim"]))),
        dropout=float(contrast_cfg.get("residual_head_dropout", contrast_cfg.get("score_head_dropout", 0.0))),
    ).to(device)
    prototype_head = ScorePrototypeHead(
        feature_dim=int(contrast_cfg.get("predictor_dim", config["model"]["projection_dim"])),
        num_bins=int(contrast_cfg.get("contrast_bins", contrast_cfg.get("num_bins", 5))),
    ).to(device)

    trainable_parameters = (
        [parameter for parameter in model_core.parameters() if parameter.requires_grad]
        + list(predictor.parameters())
        + list(score_head.parameters())
        + list(residual_head.parameters())
        + list(prototype_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["optimization"]["lr"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    scheduler_cfg = config["optimization"].get("scheduler", {})
    scheduler_total_epochs = int(scheduler_cfg.get("total_epochs", config["optimization"]["epochs"]))
    scheduler = EpochLRScheduler(
        optimizer,
        total_epochs=scheduler_total_epochs,
        config=scheduler_cfg,
    )
    model: nn.Module = model_core
    if distributed:
        model = DDP(
            model_core,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        predictor = DDP(
            predictor,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        score_head = DDP(
            score_head,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        residual_head = DDP(
            residual_head,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        prototype_head = DDP(
            prototype_head,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        if is_main:
            print(f"[COBRA] Stage1 contrast DDP enabled: world_size={world_size}")

    dataset_cfg = data_config["personalized_dataset"]
    split_payload = load_json(data_config["user_split"]["split_file"])
    train_split = str(config["data"].get("train_split", "train_fit"))
    default_val_split = data_config.get("general_dataset", {}).get("val_split", "val")
    val_split = str(config["data"].get("val_split", default_val_split))
    train_users = {str(user_id) for user_id in split_payload["splits"][train_split]}
    val_users = {str(user_id) for user_id in split_payload["splits"].get(val_split, [])}
    full_frame = load_personalized_frame(dataset_cfg["name"], dataset_cfg["root"])
    frame = full_frame[full_frame["user_id"].astype(str).isin(train_users)].reset_index(drop=True)
    val_frame = full_frame[full_frame["user_id"].astype(str).isin(val_users)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows found for Stage1 contrast split {train_split!r}.")
    score_min = float(contrast_cfg.get("score_min", frame["score"].min()))
    score_max = float(contrast_cfg.get("score_max", frame["score"].max()))
    contrast_bins = int(contrast_cfg.get("contrast_bins", contrast_cfg.get("num_bins", 5)))
    score_pool = _build_score_bin_pool(frame, num_bins=contrast_bins, score_min=score_min, score_max=score_max)
    val_score_pool = (
        _build_score_bin_pool(val_frame, num_bins=contrast_bins, score_min=score_min, score_max=score_max)
        if not val_frame.empty
        else {}
    )
    user_pool = _build_user_bin_pool(frame, num_bins=contrast_bins, score_min=score_min, score_max=score_max)
    val_user_pool = _build_user_bin_pool(val_frame, num_bins=contrast_bins, score_min=score_min, score_max=score_max) if not val_frame.empty else {}
    retrieval_pool = _build_user_score_pool(frame)
    val_retrieval_pool = _build_user_score_pool(val_frame) if not val_frame.empty else {}
    if len(score_pool) < 2:
        raise ValueError(f"Stage1 contrast split {train_split!r} has fewer than two non-empty score bins.")

    backbone_cfg = config["backbone"]
    image_cfg = {
        "image_size": int(backbone_cfg["image_size"]),
        "mean": backbone_cfg.get("mean") or [0.5, 0.5, 0.5],
        "std": backbone_cfg.get("std") or [0.5, 0.5, 0.5],
        "use_naflex": bool(backbone_cfg.get("use_naflex", False)),
        "preferred_long_side": backbone_cfg.get("preferred_long_side"),
        "patch_size": int(backbone_cfg["patch_size"]),
        "max_num_patches": backbone_cfg.get("max_num_patches"),
    }
    output_dir = ensure_dir(config["experiment"]["output_dir"])
    best_checkpoint = output_dir / "best_stage1_contrast.pt"
    latest_checkpoint = output_dir / "latest_stage1_contrast.pt"
    final_checkpoint = output_dir / "final_stage1_contrast.pt"
    best_loss = float("inf")
    best_proxy_top1 = float("-inf")
    best_proxy_loss = float("inf")
    best_selection_min = float("inf")
    best_selection_max = float("-inf")
    global_step = 0
    proxy_steps = int(contrast_cfg.get("proxy_steps", 10))
    temperature = float(contrast_cfg.get("temperature", 0.07))
    direct_loss_weight = float(contrast_cfg.get("direct_loss_weight", 0.0))
    user_level_loss_weight = float(contrast_cfg.get("user_level_loss_weight", 1.0))
    direct_user_level_loss_weight = float(contrast_cfg.get("direct_user_level_loss_weight", 0.0))
    predictor_loss_weight = float(contrast_cfg.get("predictor_loss_weight", 1.0))
    score_loss_weight = float(contrast_cfg.get("score_loss_weight", 0.0))
    rank_loss_weight = float(contrast_cfg.get("rank_loss_weight", 0.0))
    residual_contrast_loss_weight = float(contrast_cfg.get("residual_contrast_loss_weight", 0.0))
    direct_residual_contrast_loss_weight = float(contrast_cfg.get("direct_residual_contrast_loss_weight", 0.0))
    residual_score_loss_weight = float(contrast_cfg.get("residual_score_loss_weight", 0.0))
    residual_rank_loss_weight = float(contrast_cfg.get("residual_rank_loss_weight", 0.0))
    residual_target_scale = float(contrast_cfg.get("residual_target_scale", 1.0))
    residual_standardize_clamp = float(contrast_cfg.get("residual_standardize_clamp", 3.0))
    residual_standardize = bool(contrast_cfg.get("residual_standardize", True))
    support_retrieval_loss_weight = float(contrast_cfg.get("support_retrieval_loss_weight", 0.0))
    direct_support_retrieval_loss_weight = float(contrast_cfg.get("direct_support_retrieval_loss_weight", 0.0))
    support_retrieval_value_weight = float(contrast_cfg.get("support_retrieval_value_weight", 1.0))
    support_retrieval_rank_weight = float(contrast_cfg.get("support_retrieval_rank_weight", 0.5))
    support_retrieval_score_value_weight = float(contrast_cfg.get("support_retrieval_score_value_weight", 0.0))
    support_retrieval_score_rank_weight = float(contrast_cfg.get("support_retrieval_score_rank_weight", 0.0))
    support_retrieval_target_sigma = float(contrast_cfg.get("support_retrieval_target_sigma", 0.5))
    support_retrieval_temperature = float(contrast_cfg.get("support_retrieval_temperature", temperature))
    support_retrieval_support_size = int(contrast_cfg.get("support_retrieval_support_size", 32))
    support_retrieval_support_sizes = [
        int(size)
        for size in contrast_cfg.get("support_retrieval_support_sizes", [support_retrieval_support_size])
    ]
    support_retrieval_query_size = int(contrast_cfg.get("support_retrieval_query_size", 48))
    rnc_loss_weight = float(contrast_cfg.get("rnc_loss_weight", 0.0))
    direct_rnc_loss_weight = float(contrast_cfg.get("direct_rnc_loss_weight", 0.0))
    prototype_loss_weight = float(contrast_cfg.get("prototype_loss_weight", 0.0))
    prototype_score_loss_weight = float(contrast_cfg.get("prototype_score_loss_weight", 0.0))
    prototype_rank_loss_weight = float(contrast_cfg.get("prototype_rank_loss_weight", 0.0))
    prototype_temperature = float(contrast_cfg.get("prototype_temperature", temperature))
    rnc_temperature = float(contrast_cfg.get("rnc_temperature", temperature))
    ordinal_sigma = float(contrast_cfg.get("ordinal_sigma", 0.5))
    rank_min_delta = float(contrast_cfg.get("rank_min_delta", 0.1))
    rank_temperature = float(contrast_cfg.get("rank_temperature", 0.1))
    checkpoint_metric = str(contrast_cfg.get("checkpoint_metric", "train_loss")).lower()
    use_user_level_objective = contrast_objective in {
        "user_level_contrast",
        "user_level_hybrid",
        "support_residual_retrieval_hybrid",
    }
    use_support_retrieval_objective = (
        contrast_objective == "support_residual_retrieval_hybrid"
        or support_retrieval_loss_weight > 0.0
        or direct_support_retrieval_loss_weight > 0.0
    )
    samples_per_bin = int(contrast_cfg.get("samples_per_bin", 32))
    if samples_per_bin < 2:
        raise ValueError("contrast.samples_per_bin must be at least 2 for supervised contrastive positives.")
    eval_samples_per_bin = max(2, min(samples_per_bin, int(contrast_cfg.get("eval_samples_per_bin", samples_per_bin))))
    steps_per_epoch = int(config["data"]["steps_per_epoch"])
    users_per_batch = int(config["data"].get("users_per_batch", 0))
    num_workers = max(int(config["data"].get("num_workers", 0)), 0)
    prefetch_factor = max(int(config["data"].get("prefetch_factor", 2)), 1)
    pin_memory = bool(config["data"].get("pin_memory", device.type == "cuda"))
    amp_cfg = config.get("amp", {})
    amp_enabled = bool(amp_cfg.get("enabled", device.type == "cuda")) and device.type == "cuda"
    amp_dtype_name = str(amp_cfg.get("dtype", "bfloat16"))
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    print(
        "[COBRA] Stage1 contrast data: "
        f"objective={contrast_objective}, train_split={train_split}, train_bins={len(score_pool)}, "
        f"val_split={val_split}, val_bins={len(val_score_pool)}, "
        f"samples_per_bin={samples_per_bin}, steps_per_epoch={steps_per_epoch}, "
        f"num_workers={num_workers}, prefetch_factor={prefetch_factor}, pin_memory={pin_memory}, "
        f"amp={amp_enabled}:{amp_dtype_name}, proxy_steps={proxy_steps}, "
        f"feature_source={feature_source}, contrast_bins={contrast_bins}, "
        f"users_per_batch={users_per_batch}, user_level_loss_weight={user_level_loss_weight}, "
        f"direct_user_level_loss_weight={direct_user_level_loss_weight}, "
        f"direct_loss_weight={direct_loss_weight}, predictor_loss_weight={predictor_loss_weight}, "
        f"score_loss_weight={score_loss_weight}, rank_loss_weight={rank_loss_weight}, "
        f"residual_contrast_loss_weight={residual_contrast_loss_weight}, "
        f"direct_residual_contrast_loss_weight={direct_residual_contrast_loss_weight}, "
        f"residual_score_loss_weight={residual_score_loss_weight}, "
        f"residual_rank_loss_weight={residual_rank_loss_weight}, "
        f"residual_target_scale={residual_target_scale}, "
        f"residual_standardize={residual_standardize}, "
        f"support_retrieval_loss_weight={support_retrieval_loss_weight}, "
        f"direct_support_retrieval_loss_weight={direct_support_retrieval_loss_weight}, "
        f"support_retrieval_support_size={support_retrieval_support_size}, "
        f"support_retrieval_support_sizes={support_retrieval_support_sizes}, "
        f"support_retrieval_query_size={support_retrieval_query_size}, "
        f"support_retrieval_score_value_weight={support_retrieval_score_value_weight}, "
        f"support_retrieval_score_rank_weight={support_retrieval_score_rank_weight}, "
        f"rnc_loss_weight={rnc_loss_weight}, direct_rnc_loss_weight={direct_rnc_loss_weight}, "
        f"prototype_loss_weight={prototype_loss_weight}, "
        f"prototype_score_loss_weight={prototype_score_loss_weight}, "
        f"prototype_rank_loss_weight={prototype_rank_loss_weight}, "
        f"ordinal_sigma={ordinal_sigma}, "
        f"checkpoint_metric={checkpoint_metric}, fixed_epochs={int(config['optimization']['epochs'])}, "
        f"scheduler_total_epochs={scheduler_total_epochs}"
    )

    for epoch in range(1, int(config["optimization"]["epochs"]) + 1):
        current_lr = scheduler.step(epoch)
        model.train()
        predictor.train()
        score_head.train()
        prototype_head.train()
        losses: list[float] = []
        if use_user_level_objective:
            epoch_dataset = ContrastHybridIterableDataset(
                user_pool=user_pool,
                score_pool=score_pool,
                retrieval_pool=retrieval_pool if use_support_retrieval_objective else None,
                image_cfg=image_cfg,
                contrast_bins=contrast_bins,
                users_per_batch=users_per_batch,
                samples_per_bin=samples_per_bin,
                retrieval_support_size=support_retrieval_support_size,
                retrieval_support_sizes=support_retrieval_support_sizes,
                retrieval_query_size=support_retrieval_query_size,
                steps_per_epoch=steps_per_epoch,
                seed=int(config["experiment"]["seed"]) + rank * 1_000_003,
                epoch=epoch,
            )
        else:
            epoch_dataset = ContrastScoreIterableDataset(
                score_pool=score_pool,
                image_cfg=image_cfg,
                samples_per_bin=samples_per_bin,
                steps_per_epoch=steps_per_epoch,
                seed=int(config["experiment"]["seed"]) + rank * 1_000_003,
                epoch=epoch,
            )
        loader_kwargs: dict[str, Any] = {
            "batch_size": None,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "worker_init_fn": _seed_contrast_worker if num_workers > 0 else None,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["persistent_workers"] = False
            generator = torch.Generator()
            generator.manual_seed(int(config["experiment"]["seed"]) + epoch)
            loader_kwargs["generator"] = generator
        train_loader = DataLoader(epoch_dataset, **loader_kwargs)
        loop = tqdm(train_loader, total=steps_per_epoch, desc=f"Stage1 Contrast Epoch {epoch}", leave=False, disable=not is_main)
        for batch in loop:
            if use_user_level_objective:
                image_batch = _move_image_batch_to_device(batch["score_images"], device)
                left_batch = _move_image_batch_to_device(batch["left_images"], device)
                right_batch = _move_image_batch_to_device(batch["right_images"], device)
                labels = batch["score_labels"].to(device=device, non_blocking=True)
                scores = batch["score_scores"].to(device=device, non_blocking=True)
                num_images = int(batch["num_score_images"]) + 2 * int(batch["num_pairs"])
                retrieval_image_batch = (
                    _move_image_batch_to_device(batch["retrieval_images"], device)
                    if "retrieval_images" in batch
                    else None
                )
                retrieval_scores = (
                    batch["retrieval_scores"].to(device=device, non_blocking=True)
                    if "retrieval_scores" in batch
                    else None
                )
                retrieval_support_count = int(batch.get("retrieval_support_size", 0))
                retrieval_query_count = int(batch.get("retrieval_query_size", 0))
                if retrieval_image_batch is not None:
                    num_images += retrieval_support_count + retrieval_query_count
            else:
                image_batch = _move_image_batch_to_device(batch["images"], device)
                labels = batch["labels"].to(device=device, non_blocking=True)
                scores = batch["scores"].to(device=device, non_blocking=True)
                num_images = int(batch["num_images"])
                retrieval_image_batch = None
                retrieval_scores = None
                retrieval_support_count = 0
                retrieval_query_count = 0
            normalized_scores = _normalize_scores(scores, score_min=score_min, score_max=score_max)
            with torch.no_grad():
                base_scores = _stage1_base_scores(base_model, image_batch)
                residual_targets = _prepare_residual_targets(
                    scores - base_scores,
                    residual_target_scale=residual_target_scale,
                    residual_standardize_clamp=residual_standardize_clamp,
                    residual_standardize=residual_standardize,
                )
            num_bins = int(batch["num_bins"])

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                if use_user_level_objective:
                    left_raw, right_raw = _contrast_pair_features(
                        model,
                        left_batch,
                        right_batch,
                        feature_source=feature_source,
                    )
                    left_projected = predictor(left_raw)
                    right_projected = predictor(right_raw)
                    user_level_loss = _paired_user_info_nce_loss(left_projected, right_projected, temperature=temperature)
                    direct_user_level_loss = _paired_user_info_nce_loss(left_raw, right_raw, temperature=temperature)
                    pair_metrics = _paired_retrieval_metrics(left_projected.detach(), right_projected.detach())
                else:
                    user_level_loss = labels.new_zeros((), dtype=torch.float32)
                    direct_user_level_loss = labels.new_zeros((), dtype=torch.float32)
                    pair_metrics = {"top1": 0.0, "mean_rank": 0.0}
                raw_features = _contrast_features(
                    model,
                    image_batch,
                    feature_source=feature_source,
                )
                projected_features = predictor(raw_features)
                predictor_loss = _soft_ordinal_contrastive_loss(
                    projected_features,
                    scores,
                    temperature=temperature,
                    sigma=ordinal_sigma,
                )
                direct_loss = _soft_ordinal_contrastive_loss(
                    raw_features,
                    scores,
                    temperature=temperature,
                    sigma=ordinal_sigma,
                )
                rnc_loss = (
                    _rank_n_contrast_loss(projected_features, scores, temperature=rnc_temperature)
                    if rnc_loss_weight > 0.0
                    else projected_features.new_zeros(())
                )
                direct_rnc_loss = (
                    _rank_n_contrast_loss(raw_features, scores, temperature=rnc_temperature)
                    if direct_rnc_loss_weight > 0.0
                    else raw_features.new_zeros(())
                )
                predicted_scores = score_head(projected_features).sigmoid()
                predicted_residuals = residual_head(projected_features)
                score_loss = F.smooth_l1_loss(predicted_scores, normalized_scores)
                rank_loss = _pairwise_score_ranking_loss(
                    predicted_scores,
                    normalized_scores,
                    min_delta=rank_min_delta,
                    temperature=rank_temperature,
                )
                residual_contrast_loss = _soft_ordinal_contrastive_loss(
                    projected_features,
                    residual_targets,
                    temperature=temperature,
                    sigma=ordinal_sigma,
                )
                direct_residual_contrast_loss = _soft_ordinal_contrastive_loss(
                    raw_features,
                    residual_targets,
                    temperature=temperature,
                    sigma=ordinal_sigma,
                )
                residual_score_loss = F.smooth_l1_loss(predicted_residuals, residual_targets)
                residual_rank_loss = _pairwise_score_ranking_loss(
                    predicted_residuals,
                    residual_targets,
                    min_delta=0.25,
                    temperature=rank_temperature,
                )
                if retrieval_image_batch is not None and retrieval_scores is not None and retrieval_query_count > 0:
                    retrieval_raw_features = _contrast_features(
                        model,
                        retrieval_image_batch,
                        feature_source=feature_source,
                    )
                    retrieval_projected_features = predictor(retrieval_raw_features)
                    with torch.no_grad():
                        retrieval_base_scores = _stage1_base_scores(base_model, retrieval_image_batch)
                        retrieval_residual_targets = _prepare_residual_targets(
                            retrieval_scores - retrieval_base_scores,
                            residual_target_scale=residual_target_scale,
                            residual_standardize_clamp=residual_standardize_clamp,
                            residual_standardize=residual_standardize,
                        )
                    support_slice = slice(0, retrieval_support_count)
                    query_slice = slice(retrieval_support_count, retrieval_support_count + retrieval_query_count)
                    support_retrieval_loss, support_retrieval_metrics = _support_residual_retrieval_loss(
                        support_features=retrieval_projected_features[support_slice],
                        query_features=retrieval_projected_features[query_slice],
                        support_residuals=retrieval_residual_targets[support_slice],
                        query_residuals=retrieval_residual_targets[query_slice],
                        temperature=support_retrieval_temperature,
                        target_sigma=support_retrieval_target_sigma,
                        value_weight=support_retrieval_value_weight,
                        rank_weight=support_retrieval_rank_weight,
                        rank_min_delta=rank_min_delta,
                        rank_temperature=rank_temperature,
                        query_base_scores=retrieval_base_scores[query_slice],
                        query_scores=retrieval_scores[query_slice],
                        score_value_weight=support_retrieval_score_value_weight,
                        score_rank_weight=support_retrieval_score_rank_weight,
                    )
                    direct_support_retrieval_loss, direct_support_retrieval_metrics = _support_residual_retrieval_loss(
                        support_features=retrieval_raw_features[support_slice],
                        query_features=retrieval_raw_features[query_slice],
                        support_residuals=retrieval_residual_targets[support_slice],
                        query_residuals=retrieval_residual_targets[query_slice],
                        temperature=support_retrieval_temperature,
                        target_sigma=support_retrieval_target_sigma,
                        value_weight=support_retrieval_value_weight,
                        rank_weight=support_retrieval_rank_weight,
                        rank_min_delta=rank_min_delta,
                        rank_temperature=rank_temperature,
                        query_base_scores=retrieval_base_scores[query_slice],
                        query_scores=retrieval_scores[query_slice],
                        score_value_weight=support_retrieval_score_value_weight,
                        score_rank_weight=support_retrieval_score_rank_weight,
                    )
                else:
                    support_retrieval_loss = projected_features.new_zeros(())
                    direct_support_retrieval_loss = projected_features.new_zeros(())
                    support_retrieval_metrics = {
                        "mae": projected_features.new_zeros(()),
                        "pair_acc": projected_features.new_zeros(()),
                        "srcc": projected_features.new_zeros(()),
                    }
                    direct_support_retrieval_metrics = support_retrieval_metrics
                prototype_logits = prototype_head(projected_features, temperature=prototype_temperature)
                prototype_scores = _prototype_expected_scores(prototype_logits)
                prototype_loss = F.cross_entropy(prototype_logits, labels)
                prototype_score_loss = F.smooth_l1_loss(prototype_scores, normalized_scores)
                prototype_rank_loss = _pairwise_score_ranking_loss(
                    prototype_scores,
                    normalized_scores,
                    min_delta=rank_min_delta,
                    temperature=rank_temperature,
                )
                loss = (
                    user_level_loss_weight * user_level_loss
                    + direct_user_level_loss_weight * direct_user_level_loss
                    + predictor_loss_weight * predictor_loss
                    + direct_loss_weight * direct_loss
                    + score_loss_weight * score_loss
                    + rank_loss_weight * rank_loss
                    + residual_contrast_loss_weight * residual_contrast_loss
                    + direct_residual_contrast_loss_weight * direct_residual_contrast_loss
                    + residual_score_loss_weight * residual_score_loss
                    + residual_rank_loss_weight * residual_rank_loss
                    + support_retrieval_loss_weight * support_retrieval_loss
                    + direct_support_retrieval_loss_weight * direct_support_retrieval_loss
                    + rnc_loss_weight * rnc_loss
                    + direct_rnc_loss_weight * direct_rnc_loss
                    + prototype_loss_weight * prototype_loss
                    + prototype_score_loss_weight * prototype_score_loss
                    + prototype_rank_loss_weight * prototype_rank_loss
                )
            loss.backward()
            optimizer.step()
            global_step += 1
            losses.append(float(loss.item()))
            loop.set_postfix(loss=f"{loss.item():.4f}", images=num_images, bins=num_bins)
            if tracker is not None and global_step % max(int(config.get("tracking", {}).get("log_every_steps", 20)), 1) == 0:
                metrics = _score_retrieval_metrics(projected_features.detach(), labels.detach())
                tracker.log(
                    {
                        "train/global_step": global_step,
                        "train/loss": float(loss.item()),
                        "train/user_level_loss": float(user_level_loss.item()),
                        "train/direct_user_level_loss": float(direct_user_level_loss.item()),
                        "train/predictor_loss": float(predictor_loss.item()),
                        "train/direct_loss": float(direct_loss.item()),
                        "train/rnc_loss": float(rnc_loss.item()),
                        "train/direct_rnc_loss": float(direct_rnc_loss.item()),
                        "train/score_loss": float(score_loss.item()),
                        "train/rank_loss": float(rank_loss.item()),
                        "train/residual_contrast_loss": float(residual_contrast_loss.item()),
                        "train/direct_residual_contrast_loss": float(direct_residual_contrast_loss.item()),
                        "train/residual_score_loss": float(residual_score_loss.item()),
                        "train/residual_rank_loss": float(residual_rank_loss.item()),
                        "train/support_retrieval_loss": float(support_retrieval_loss.item()),
                        "train/direct_support_retrieval_loss": float(direct_support_retrieval_loss.item()),
                        "train/support_retrieval_mae": float(support_retrieval_metrics["mae"].item()),
                        "train/support_retrieval_pair_acc": float(support_retrieval_metrics["pair_acc"].item()),
                        "train/support_retrieval_srcc": float(support_retrieval_metrics["srcc"].item()),
                        "train/support_retrieval_score_srcc": float(support_retrieval_metrics["score_srcc"].item()),
                        "train/direct_support_retrieval_srcc": float(direct_support_retrieval_metrics["srcc"].item()),
                        "train/direct_support_retrieval_score_srcc": float(direct_support_retrieval_metrics["score_srcc"].item()),
                        "train/prototype_loss": float(prototype_loss.item()),
                        "train/prototype_score_loss": float(prototype_score_loss.item()),
                        "train/prototype_rank_loss": float(prototype_rank_loss.item()),
                        "train/top1": metrics["top1"],
                        "train/mean_rank": metrics["mean_rank"],
                        "train/user_top1": pair_metrics["top1"],
                        "train/user_mean_rank": pair_metrics["mean_rank"],
                        "train/score_mae": float(torch.mean(torch.abs(predicted_scores.detach() - normalized_scores)).item()),
                        "train/residual_mae": float(torch.mean(torch.abs(predicted_residuals.detach() - residual_targets)).item()),
                        "train/prototype_score_mae": float(
                            torch.mean(torch.abs(prototype_scores.detach() - normalized_scores)).item()
                        ),
                        "train/prototype_acc": float(prototype_logits.detach().argmax(dim=1).eq(labels).float().mean().item()),
                        "train/images": float(num_images),
                        "train/bins": float(num_bins),
                        "optimization/lr": current_lr,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        epoch_loss = float(sum(losses) / max(len(losses), 1))
        score_proxy_metrics = (
            _evaluate_score_proxy(
                model=model,
                base_model=base_model,
                predictor=predictor,
                score_head=score_head,
                residual_head=residual_head,
                prototype_head=prototype_head,
                score_pool=val_score_pool,
                image_cfg=image_cfg,
                device=device,
                samples_per_bin=eval_samples_per_bin,
                temperature=temperature,
                prototype_temperature=prototype_temperature,
                rnc_temperature=rnc_temperature,
                ordinal_sigma=ordinal_sigma,
                score_min=score_min,
                score_max=score_max,
                steps=proxy_steps,
                seed=int(config["experiment"]["seed"]) + 10_000,
                feature_source=feature_source,
                contrast_objective=contrast_objective,
                residual_target_scale=residual_target_scale,
                residual_standardize_clamp=residual_standardize_clamp,
                residual_standardize=residual_standardize,
            )
            if len(val_score_pool) >= 2
            else {
                "loss": epoch_loss,
                "top1": 0.0,
                "mean_rank": 0.0,
                "score_mae": 0.0,
                "score_head_mae": 0.0,
                "pair_acc": 0.0,
                "prototype_acc": 0.0,
                "residual_mae": 0.0,
                "residual_pair_acc": 0.0,
                "residual_proxy_loss": 0.0,
            }
        )
        if use_user_level_objective and len(val_user_pool) >= 2:
            user_proxy_metrics = _evaluate_contrast_proxy(
                model=model,
                predictor=predictor,
                user_pool=val_user_pool,
                image_cfg=image_cfg,
                device=device,
                contrast_bins=contrast_bins,
                users_per_batch=users_per_batch,
                temperature=temperature,
                steps=proxy_steps,
                seed=int(config["experiment"]["seed"]) + 20_000,
                feature_source=feature_source,
            )
            proxy_metrics = {
                **score_proxy_metrics,
                "loss": user_proxy_metrics["loss"],
                "top1": user_proxy_metrics["top1"],
                "mean_rank": user_proxy_metrics["mean_rank"],
                "score_proxy_loss": score_proxy_metrics["loss"],
                "score_proxy_top1": score_proxy_metrics["top1"],
            }
        else:
            proxy_metrics = score_proxy_metrics
        if use_support_retrieval_objective and val_retrieval_pool:
            support_proxy_metrics = _evaluate_support_retrieval_proxy(
                model=model,
                base_model=base_model,
                predictor=predictor,
                user_score_pool=val_retrieval_pool,
                image_cfg=image_cfg,
                device=device,
                support_size=support_retrieval_support_size,
                query_size=support_retrieval_query_size,
                temperature=support_retrieval_temperature,
                target_sigma=support_retrieval_target_sigma,
                value_weight=support_retrieval_value_weight,
                rank_weight=support_retrieval_rank_weight,
                rank_min_delta=rank_min_delta,
                rank_temperature=rank_temperature,
                residual_target_scale=residual_target_scale,
                residual_standardize_clamp=residual_standardize_clamp,
                residual_standardize=residual_standardize,
                score_value_weight=support_retrieval_score_value_weight,
                score_rank_weight=support_retrieval_score_rank_weight,
                steps=proxy_steps,
                seed=int(config["experiment"]["seed"]) + 30_000,
                feature_source=feature_source,
            )
            proxy_metrics = {**proxy_metrics, **support_proxy_metrics}
        if is_main:
            _save_contrast_checkpoint(
                latest_checkpoint,
                model_core,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                epoch_loss=epoch_loss,
                proxy_metrics=proxy_metrics,
                giaa_checkpoint=giaa_checkpoint,
                checkpoint_policy="latest",
                contrast_objective=contrast_objective,
            )
        if tracker is not None:
            tracker.log(
                {
                    "train/loss": epoch_loss,
                    "train/bins": float(len(score_pool)),
                    "train/images_per_step": float(samples_per_bin * len(score_pool)),
                    "val_proxy/loss": proxy_metrics["loss"],
                    "val_proxy/top1": proxy_metrics["top1"],
                    "val_proxy/mean_rank": proxy_metrics["mean_rank"],
                    "val_proxy/score_mae": proxy_metrics.get("score_mae", 0.0),
                    "val_proxy/score_head_mae": proxy_metrics.get("score_head_mae", 0.0),
                    "val_proxy/residual_mae": proxy_metrics.get("residual_mae", 0.0),
                    "val_proxy/residual_pair_acc": proxy_metrics.get("residual_pair_acc", 0.0),
                    "val_proxy/residual_proxy_loss": proxy_metrics.get("residual_proxy_loss", 0.0),
                    "val_proxy/support_retrieval_loss": proxy_metrics.get("support_retrieval_loss", 0.0),
                    "val_proxy/support_retrieval_srcc": proxy_metrics.get("support_retrieval_srcc", 0.0),
                    "val_proxy/support_retrieval_pair_acc": proxy_metrics.get("support_retrieval_pair_acc", 0.0),
                    "val_proxy/support_retrieval_direct_srcc": proxy_metrics.get("support_retrieval_direct_srcc", 0.0),
                    "val_proxy/support_retrieval_direct_pair_acc": proxy_metrics.get("support_retrieval_direct_pair_acc", 0.0),
                    "val_proxy/pair_acc": proxy_metrics.get("pair_acc", 0.0),
                    "val_proxy/prototype_acc": proxy_metrics.get("prototype_acc", 0.0),
                    "optimization/lr": current_lr,
                    "epoch": epoch,
                },
                step=epoch,
            )
        if is_main:
            tqdm.write(
                f"[COBRA] Stage1 Contrast Epoch {epoch}: "
                f"loss={epoch_loss:.4f}, val_top1={proxy_metrics['top1']:.4f}, "
                f"val_loss={proxy_metrics['loss']:.4f}, "
                f"val_mae={proxy_metrics.get('score_mae', 0.0):.4f}, "
                f"val_head_mae={proxy_metrics.get('score_head_mae', 0.0):.4f}, "
                f"val_residual_mae={proxy_metrics.get('residual_mae', 0.0):.4f}, "
                f"val_residual_pair={proxy_metrics.get('residual_pair_acc', 0.0):.4f}, "
                f"val_support_ret_srcc={proxy_metrics.get('support_retrieval_srcc', 0.0):.4f}, "
                f"val_support_ret_direct_srcc={proxy_metrics.get('support_retrieval_direct_srcc', 0.0):.4f}, "
                f"val_pair={proxy_metrics.get('pair_acc', 0.0):.4f}, "
                f"val_proto={proxy_metrics.get('prototype_acc', 0.0):.4f}, "
                f"images_per_step={samples_per_bin * len(score_pool)}"
            )
        if checkpoint_metric == "val_proxy_top1":
            selection_value = proxy_metrics["top1"]
            is_better = selection_value > best_selection_max
        elif checkpoint_metric == "val_proxy_loss":
            selection_value = proxy_metrics["loss"]
            is_better = selection_value < best_selection_min
        elif checkpoint_metric == "val_proxy_score_mae":
            selection_value = proxy_metrics.get("score_mae", float("inf"))
            is_better = selection_value < best_selection_min
        elif checkpoint_metric == "val_proxy_pair_acc":
            selection_value = proxy_metrics.get("pair_acc", 0.0)
            is_better = selection_value > best_selection_max
        elif checkpoint_metric == "val_proxy_residual_pair_acc":
            selection_value = proxy_metrics.get("residual_pair_acc", 0.0)
            is_better = selection_value > best_selection_max
        elif checkpoint_metric == "val_proxy_residual_mae":
            selection_value = proxy_metrics.get("residual_mae", float("inf"))
            is_better = selection_value < best_selection_min
        elif checkpoint_metric == "val_proxy_support_retrieval_srcc":
            selection_value = proxy_metrics.get("support_retrieval_srcc", 0.0)
            is_better = selection_value > best_selection_max
        elif checkpoint_metric == "val_proxy_support_retrieval_direct_srcc":
            selection_value = proxy_metrics.get("support_retrieval_direct_srcc", 0.0)
            is_better = selection_value > best_selection_max
        elif checkpoint_metric == "val_proxy_support_retrieval_pair_acc":
            selection_value = proxy_metrics.get("support_retrieval_pair_acc", 0.0)
            is_better = selection_value > best_selection_max
        elif checkpoint_metric == "final":
            selection_value = epoch_loss
            is_better = epoch == int(config["optimization"]["epochs"])
        else:
            selection_value = epoch_loss
            is_better = selection_value < best_selection_min
        if is_main and is_better:
            best_loss = epoch_loss
            best_proxy_top1 = proxy_metrics["top1"]
            best_proxy_loss = proxy_metrics["loss"]
            if checkpoint_metric in {
                "val_proxy_top1",
                "val_proxy_pair_acc",
                "val_proxy_residual_pair_acc",
                "val_proxy_support_retrieval_srcc",
                "val_proxy_support_retrieval_direct_srcc",
                "val_proxy_support_retrieval_pair_acc",
            }:
                best_selection_max = selection_value
            else:
                best_selection_min = selection_value
            _save_contrast_checkpoint(
                best_checkpoint,
                model_core,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                epoch_loss=epoch_loss,
                proxy_metrics=proxy_metrics,
                giaa_checkpoint=giaa_checkpoint,
                checkpoint_policy=f"best:{checkpoint_metric}",
                contrast_objective=contrast_objective,
            )
            if tracker is not None:
                tracker.log_summary(
                    {
                        "best/stage1_contrast_loss": best_loss,
                        "best/stage1_contrast_val_proxy_loss": best_proxy_loss,
                        "best/stage1_contrast_val_proxy_top1": best_proxy_top1,
                        "best/stage1_contrast_val_proxy_score_mae": proxy_metrics.get("score_mae", 0.0),
                        "best/stage1_contrast_val_proxy_pair_acc": proxy_metrics.get("pair_acc", 0.0),
                        "best/stage1_contrast_val_proxy_residual_mae": proxy_metrics.get("residual_mae", 0.0),
                        "best/stage1_contrast_val_proxy_residual_pair_acc": proxy_metrics.get("residual_pair_acc", 0.0),
                        "best/stage1_contrast_val_proxy_support_retrieval_srcc": proxy_metrics.get("support_retrieval_srcc", 0.0),
                        "best/stage1_contrast_val_proxy_support_retrieval_pair_acc": proxy_metrics.get("support_retrieval_pair_acc", 0.0),
                        "best/stage1_contrast_val_proxy_support_retrieval_direct_srcc": proxy_metrics.get("support_retrieval_direct_srcc", 0.0),
                        "best/stage1_contrast_epoch": epoch,
                        "best/stage1_contrast_selection_metric": checkpoint_metric,
                        "best/stage1_contrast_selection_value": selection_value,
                        "artifacts/best_stage1_contrast_checkpoint": best_checkpoint.as_posix(),
                    }
                )
        if distributed:
            dist.barrier()
    if is_main:
        _save_contrast_checkpoint(
            final_checkpoint,
            model_core,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=int(config["optimization"]["epochs"]),
            epoch_loss=epoch_loss,
            proxy_metrics=proxy_metrics,
            giaa_checkpoint=giaa_checkpoint,
            checkpoint_policy="final",
            contrast_objective=contrast_objective,
        )
    if is_main and checkpoint_metric == "final":
        _save_contrast_checkpoint(
            best_checkpoint,
            model_core,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=int(config["optimization"]["epochs"]),
            epoch_loss=epoch_loss,
            proxy_metrics=proxy_metrics,
            giaa_checkpoint=giaa_checkpoint,
            checkpoint_policy="final",
            contrast_objective=contrast_objective,
        )
    if tracker is not None:
        tracker.log_summary(
            {
                "artifacts/final_stage1_contrast_checkpoint": final_checkpoint.as_posix(),
                "artifacts/latest_stage1_contrast_checkpoint": latest_checkpoint.as_posix(),
                "best/stage1_contrast_fixed_epochs": int(config["optimization"]["epochs"]),
            }
        )
    return best_checkpoint
