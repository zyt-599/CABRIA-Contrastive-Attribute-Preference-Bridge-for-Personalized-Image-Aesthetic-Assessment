from __future__ import annotations

import copy
import gc
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from cabria.data.episode_sampler import build_episode_dataloader
from cabria.data.personalized_dataset import (
    PersonalizedEpisodeDataset,
    build_train_image_prior_lookup,
    load_personalized_frame,
)
from cabria.data.user_split import build_episode_specs, build_fixed_query_episodes, deserialize_episode_specs, filter_users_by_min_row_count
from cabria.evaluation.fixed_query import build_fixed_query_val_episodes, resolve_fixed_query_settings
from cabria.evaluation.episodic import (
    build_episodic_episodes,
    build_episodic_episodes_common_query,
    filter_episodic_eligible_users,
    resolve_episodic_test_users,
    resolve_episodic_val_settings,
)
from cabria.engine.evaluator import evaluate_stage2_decomposed
from cabria.engine.stage2_tta import adapt_user_on_support, tta_enabled
from cabria.losses.distribution_alignment import DistributionAlignmentLoss
from cabria.losses.general_regression import GeneralRegressionLoss
from cabria.losses.query_ranking import pairwise_ranking_loss
from cabria.losses.same_image_order import same_image_user_order_loss
from cabria.models.cabria_model import CABRIAStage2Model
from cabria.utils.checkpoint import save_checkpoint
from cabria.utils.common import ensure_dir, load_json, resolve_path
from cabria.utils.factory import build_model_config
from cabria.utils.metrics import macro_user_correlations, same_image_rank_correlation, threshold_accuracy
from cabria.utils.optimization import EarlyStopping, EpochLRScheduler
from cabria.utils.tracking import Tracker


def _stage2_checkpoint_label(config: dict, support_size: str) -> str:
    experiment_cfg = config.get("experiment", {}) or {}
    explicit_label = experiment_cfg.get("checkpoint_label")
    if explicit_label:
        return str(explicit_label)

    loss_cfg = config.get("loss", {}) or {}
    nested_sizes = sorted(
        {
            int(size)
            for size in loss_cfg.get("train_support_sizes", [])
            if 0 < int(size) <= int(support_size)
        }
    )
    if len(nested_sizes) > 1:
        return "multishot"
    return f"s{int(support_size)}"


def _stage2_resume_path(config: dict, output_dir: Path, checkpoint_label: str) -> Path | None:
    experiment_cfg = config.get("experiment", {})
    checkpoint = experiment_cfg.get("resume_checkpoint")
    if checkpoint:
        return resolve_path(str(checkpoint))
    if bool(experiment_cfg.get("resume", False)):
        return output_dir / f"checkpoint_latest_{checkpoint_label}.pt"
    return None


def _support_query_residual_alignment_loss(
    query_embeddings: torch.Tensor,
    support_embeddings: torch.Tensor,
    query_targets: torch.Tensor,
    support_targets: torch.Tensor,
    query_base: torch.Tensor,
    support_base: torch.Tensor,
    *,
    feature_temperature: float,
    residual_temperature: float,
) -> torch.Tensor:
    if query_embeddings.numel() == 0 or support_embeddings.numel() == 0:
        return query_embeddings.new_zeros(())
    query_residual = (query_targets.flatten() - query_base.detach().flatten()).float()
    support_residual = (support_targets.flatten() - support_base.detach().flatten()).float()
    if query_residual.numel() == 0 or support_residual.numel() == 0:
        return query_embeddings.new_zeros(())
    query_norm = F.normalize(query_embeddings.float(), dim=-1)
    support_norm = F.normalize(support_embeddings.float(), dim=-1)
    logits = torch.matmul(query_norm, support_norm.transpose(0, 1)) / max(float(feature_temperature), 1e-6)
    residual_distance = torch.abs(query_residual.unsqueeze(1) - support_residual.unsqueeze(0))
    target_distribution = torch.softmax(-residual_distance / max(float(residual_temperature), 1e-6), dim=1).detach()
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_distribution * log_probs).sum(dim=1).mean().to(dtype=query_embeddings.dtype)


def _support_anchored_query_ranking_loss(
    query_predictions: torch.Tensor,
    query_targets: torch.Tensor,
    support_predictions: torch.Tensor,
    support_targets: torch.Tensor,
    *,
    min_delta: float,
    max_pairs: int,
) -> torch.Tensor:
    if query_predictions.numel() == 0 or support_predictions.numel() == 0:
        return query_predictions.new_zeros(())

    target_delta = query_targets.flatten().unsqueeze(1) - support_targets.flatten().unsqueeze(0)
    pair_mask = target_delta.abs() > min_delta
    if not bool(pair_mask.any()):
        return query_predictions.new_zeros(())

    pred_delta = query_predictions.flatten().unsqueeze(1) - support_predictions.flatten().unsqueeze(0)
    signs = target_delta.sign()
    pair_losses = F.softplus(-(signs * pred_delta))[pair_mask]
    if max_pairs > 0 and pair_losses.numel() > max_pairs:
        indices = torch.linspace(
            0,
            pair_losses.numel() - 1,
            steps=max_pairs,
            device=pair_losses.device,
        ).long()
        pair_losses = pair_losses.index_select(0, indices)
    return pair_losses.mean()


def _float_or_default(value, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


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


def _rank_subset(dataset, *, distributed: bool, rank: int, world_size: int):
    if not distributed:
        return dataset
    return Subset(dataset, list(range(rank, len(dataset), world_size)))


def _metrics_for_prediction_column(frame: pd.DataFrame, column: str) -> dict[str, float]:
    eval_frame = frame[["user_id", "image_id", "score", column]].rename(columns={column: "prediction"})
    metrics = macro_user_correlations(eval_frame)
    metrics["same_image_srcc"] = same_image_rank_correlation(eval_frame)
    return metrics


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    value = left.astype(float).corr(right.astype(float), method=method)
    return float(value) if value == value else float("nan")


def _macro_residual_corr(frame: pd.DataFrame, column: str, method: str) -> float:
    values: list[float] = []
    for _user_id, group in frame.groupby("user_id"):
        value = _safe_corr(group[column], group["target_residual"], method)
        if value == value:
            values.append(value)
    return float(sum(values) / len(values)) if values else float("nan")


def _residual_correlation_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if "target_residual" not in frame.columns:
        return {}
    metrics: dict[str, float] = {}
    for column in [
        "residual_score",
        "bridge_residual_score",
        "bridge_basis_residual_score",
        "retrieved_residual_score",
        "kernel_residual_score",
        "user_state_residual_score",
        "support_estimator_residual_score",
        "basis_residual_score",
        "local_residual_score",
    ]:
        if column not in frame.columns:
            continue
        prefix = column.removesuffix("_score")
        metrics[f"{prefix}_global_spearman"] = _safe_corr(frame[column], frame["target_residual"], "spearman")
        metrics[f"{prefix}_macro_spearman"] = _macro_residual_corr(frame, column, "spearman")
        metrics[f"{prefix}_global_pearson"] = _safe_corr(frame[column], frame["target_residual"], "pearson")
        metrics[f"{prefix}_macro_pearson"] = _macro_residual_corr(frame, column, "pearson")
    return metrics


def _metrics_from_predictions(
    frame: pd.DataFrame,
    *,
    acc_threshold: float | None,
) -> dict[str, float]:
    final_metrics = _metrics_for_prediction_column(frame, "prediction")
    base_metrics = _metrics_for_prediction_column(frame, "base_prediction")
    stage1_metrics = _metrics_for_prediction_column(frame, "stage1_prediction")
    metrics = dict(final_metrics)
    metrics.update({f"base_{key}": value for key, value in base_metrics.items()})
    metrics.update({f"stage1_{key}": value for key, value in stage1_metrics.items()})
    metrics["gain_vs_base_macro_srcc"] = metrics["macro_srcc"] - metrics["base_macro_srcc"]
    metrics["gain_vs_base_macro_plcc"] = metrics["macro_plcc"] - metrics["base_macro_plcc"]
    metrics.update(_residual_correlation_metrics(frame))
    if acc_threshold is not None:
        metrics["acc"] = threshold_accuracy(frame, threshold=acc_threshold)
    return metrics


def train_stage2(config: dict, data_config: dict, device: torch.device, tracker: Tracker | None = None) -> Path:
    personalized_cfg = data_config["personalized_dataset"]
    print(f"[CABRIA] Stage2 device: {device}")
    print(f"[CABRIA] Stage2 personalized dataset: {personalized_cfg['name']}")
    print(f"[CABRIA] Stage2 dataset root: {personalized_cfg['root']}")
    print(f"[CABRIA] Stage2 split manifest: {data_config['user_split']['split_file']}")
    split_payload = load_json(data_config["user_split"]["split_file"])
    frame = load_personalized_frame(personalized_cfg["name"], personalized_cfg["root"])
    episodes = {
        split_name: deserialize_episode_specs(specs_by_support)
        for split_name, specs_by_support in split_payload["episodes"].items()
    }
    support_size = str(config["data"]["support_size"])
    train_split_name = str(config["data"].get("train_split", "train"))
    image_prior_split_name = str(config["data"].get("image_prior_split", train_split_name))
    if train_split_name not in split_payload["splits"]:
        raise KeyError(f"Stage2 train split '{train_split_name}' not found in split manifest")
    if image_prior_split_name not in split_payload["splits"]:
        raise KeyError(f"Stage2 image prior split '{image_prior_split_name}' not found in split manifest")
    image_prior_lookup = None
    if bool(config["data"].get("use_image_prior", False)):
        image_prior_lookup = build_train_image_prior_lookup(frame, split_payload["splits"][image_prior_split_name])
        print(f"[CABRIA] Stage2 image prior enabled: split={image_prior_split_name}, samples={len(image_prior_lookup)}")
    print(f"[CABRIA] Stage2 support size: {support_size}")
    print(f"[CABRIA] Stage2 train split: {train_split_name}")
    print(
        "[CABRIA] Stage2 query limits: "
        f"train_query_size={config['data'].get('train_query_size')}, "
        f"eval_query_size={config['data'].get('eval_query_size')}"
    )
    print(
        "[CABRIA] Stage2 episode counts: "
        f"train={len(episodes[train_split_name][support_size])}, "
        f"val={len(episodes['val'][support_size])}, "
        f"test={len(episodes['test'][support_size])}"
    )
    print(f"[CABRIA] Stage2 normalized samples: {len(frame)}")

    backbone_cfg = config["backbone"]
    print(f"[CABRIA] Stage2 loading Stage1 checkpoint: {config['experiment']['stage1_checkpoint']}")
    contrast_checkpoint = config["experiment"].get("contrast_checkpoint")
    if contrast_checkpoint:
        print(f"[CABRIA] Stage2 loading contrast checkpoint: {contrast_checkpoint}")
    model = CABRIAStage2Model(
        build_model_config(config),
        stage1_checkpoint=config["experiment"]["stage1_checkpoint"],
        contrast_checkpoint=contrast_checkpoint,
    ).to(device)
    init_checkpoint = config["experiment"].get("stage2_init_checkpoint")
    if init_checkpoint:
        init_path = resolve_path(str(init_checkpoint))
        payload = torch.load(init_path, map_location=device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        init_strict = bool(config["experiment"].get("stage2_init_strict", True))
        incompatible = model.load_state_dict(state, strict=init_strict)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "[CABRIA] Stage2 warm-start non-strict keys: "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
            )
        print(
            f"[CABRIA] Stage2 warm-started from checkpoint: {init_path.as_posix()} "
            f"strict={init_strict}"
        )
    model.freeze_stage1(
        freeze_backbone=bool(config["stage2_freeze"]["freeze_backbone"]),
        freeze_adapter=bool(config["stage2_freeze"].get("freeze_adapter", config["stage2_freeze"]["freeze_backbone"])),
        freeze_attribute_extractor=bool(config["stage2_freeze"]["freeze_attribute_extractor"]),
        freeze_general_head=bool(config["stage2_freeze"]["freeze_general_head"]),
        freeze_attribute_head=bool(config["stage2_freeze"].get("freeze_attribute_head", True)),
    )
    print(
        "[CABRIA] Stage2 freeze policy: "
        f"backbone={config['stage2_freeze']['freeze_backbone']}, "
        f"adapter={config['stage2_freeze'].get('freeze_adapter', config['stage2_freeze']['freeze_backbone'])}, "
        f"attribute_extractor={config['stage2_freeze']['freeze_attribute_extractor']}, "
        f"general_head={config['stage2_freeze']['freeze_general_head']}, "
        f"attribute_head={config['stage2_freeze'].get('freeze_attribute_head', True)}"
    )

    distributed, rank, world_size, local_rank = _distributed_state()
    is_main = rank == 0
    model_core = model
    for name, parameter in model_core.named_parameters():
        if name.startswith("tta_") and name.endswith("_template"):
            parameter.requires_grad_(False)
    if distributed:
        model = DDP(
            model_core,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=bool(config.get("distributed", {}).get("find_unused_parameters", False)),
            broadcast_buffers=False,
            init_sync=False,
        )
        if is_main:
            find_unused = bool(config.get("distributed", {}).get("find_unused_parameters", False))
            print(f"[CABRIA] Stage2 DDP enabled: world_size={world_size}, find_unused_parameters={find_unused}")

    dynamic_train_episodes = bool(config["data"].get("dynamic_train_episodes", False))
    train_episodes_per_user = int(
        config["data"].get(
            "train_episodes_per_user",
            data_config.get("user_split", {}).get("train_episodes_per_user", 1),
        )
    )
    base_seed = int(config.get("experiment", {}).get("seed", data_config.get("user_split", {}).get("seed", 42)))
    eval_cfg = config.get("evaluation", {}) or {}
    diagnostics_cfg = config.get("diagnostics", {}) or {}
    diagnostics_max_train_batches = max(int(diagnostics_cfg.get("max_train_batches", 0)), 0)
    diagnostics_max_val_users = max(int(diagnostics_cfg.get("max_val_users", 0)), 0)
    selection_repeats = int(eval_cfg.get("selection_repeats", 1))
    _fixed_seed_raw = eval_cfg.get("fixed_episode_seed")
    if _fixed_seed_raw is None or (isinstance(_fixed_seed_raw, str) and _fixed_seed_raw.strip().lower() in {"", "null", "none"}):
        fixed_episode_seed = None
    else:
        fixed_episode_seed = int(_fixed_seed_raw)
    if fixed_episode_seed is not None and selection_repeats > 1 and is_main:
        print(
            "[CABRIA] Stage2 evaluation: fixed_episode_seed is set; "
            f"ignoring selection_repeats={selection_repeats} (single deterministic val pass)."
        )
    if fixed_episode_seed is not None and is_main:
        print(
            f"[CABRIA] Stage2 val support seed anchor: fixed_episode_seed={fixed_episode_seed} "
            "(fixed-query val; same defaults as cabria.evaluation.fixed_query + scripts/evaluate.py)."
        )
    selection_base_penalty = float(eval_cfg.get("selection_base_penalty", 0.0))
    selection_gain_weight = float(eval_cfg.get("selection_gain_weight", 1.0))
    selection_same_image_weight = float(eval_cfg.get("selection_same_image_weight", 0.0))
    selection_residual_corr_weight = float(eval_cfg.get("selection_residual_corr_weight", 0.0))
    selection_residual_corr_metric = str(
        eval_cfg.get("selection_residual_corr_metric", "retrieved_residual_macro_spearman")
    )

    def make_dataset(split_specs, train: bool, max_query_size: int | None) -> PersonalizedEpisodeDataset:
        return PersonalizedEpisodeDataset(
            frame=frame,
            episodes=split_specs,
            image_size=int(backbone_cfg["image_size"]),
            train=train,
            mean=backbone_cfg.get("mean"),
            std=backbone_cfg.get("std"),
            use_naflex=bool(backbone_cfg.get("use_naflex", False)),
            preferred_long_side=backbone_cfg.get("preferred_long_side"),
            patch_size=int(backbone_cfg["patch_size"]),
            max_num_patches=backbone_cfg.get("max_num_patches"),
            max_query_size=max_query_size,
            image_prior_lookup=image_prior_lookup,
            cache_resized_images=bool(config["data"].get("cache_resized_images", False)),
            image_cache_size=int(config["data"].get("image_cache_size", 0)),
        )

    dataloader_persistent_workers = bool(config["data"].get("persistent_workers", False))
    dataloader_prefetch_factor = int(config["data"].get("prefetch_factor", 2))
    eval_num_workers = int(config["data"].get("eval_num_workers", config["data"]["num_workers"]))

    def build_train_loader_for_epoch(epoch: int):
        if dynamic_train_episodes:
            epoch_seed = base_seed + epoch * 1009
            split_episodes = build_episode_specs(
                frame,
                [str(user_id) for user_id in split_payload["splits"][train_split_name]],
                [int(support_size)],
                seed=epoch_seed,
                episodes_per_user=train_episodes_per_user,
            )
            train_specs = split_episodes[support_size]
        else:
            train_specs = episodes[train_split_name][support_size]
        train_dataset = make_dataset(
            train_specs,
            train=True,
            max_query_size=config["data"].get("train_query_size"),
        )
        train_sampler = None
        if distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=base_seed + epoch,
            )
        train_loader = build_episode_dataloader(
            train_dataset,
            shuffle=train_sampler is None,
            num_workers=int(config["data"]["num_workers"]),
            batch_size=int(config["data"].get("train_episode_batch_size", 1)),
            persistent_workers=dataloader_persistent_workers and not dynamic_train_episodes,
            prefetch_factor=dataloader_prefetch_factor,
            sampler=train_sampler,
        )
        return train_dataset, train_loader, train_sampler

    train_dataset, train_loader, train_sampler = build_train_loader_for_epoch(0)
    val_protocol = str(eval_cfg.get("val_protocol", "fixed_query")).lower()
    val_user_split_name = str(eval_cfg.get("val_user_split", "val"))
    if val_user_split_name not in split_payload["splits"]:
        raise KeyError(f"Stage2 validation split '{val_user_split_name}' not found in split manifest")
    val_user_ids = [str(user_id) for user_id in split_payload["splits"][val_user_split_name]]
    if is_main and val_user_split_name != "val":
        print(
            f"[CABRIA] Stage2 validation user split override: "
            f"val_user_split={val_user_split_name}, users={len(val_user_ids)}"
        )
    val_episodes_are_episodic = val_protocol == "episodic"
    episodic_val_common_query = bool(eval_cfg.get("episodic_val_common_query", False))
    episodic_val_query_support_size = max(
        int(support_size),
        int(eval_cfg.get("episodic_val_query_support_size", eval_cfg.get("query_support_size", support_size))),
    )
    if val_protocol == "manifest":
        val_episode_specs = episodes["val"][support_size]
        val_specs_source = "manifest"
    elif val_protocol == "episodic":
        episodic_val_cfg = resolve_episodic_val_settings(eval_cfg)
        val_episodic_users = resolve_episodic_test_users(
            frame,
            user_source=str(episodic_val_cfg["user_source"]),
            split_users=val_user_ids,
            test_users_file=episodic_val_cfg["test_users_file"],
        )
        eligibility_support_size = episodic_val_query_support_size if episodic_val_common_query else int(support_size)
        val_episodic_eligible = filter_episodic_eligible_users(frame, val_episodic_users, eligibility_support_size)
        if episodic_val_common_query:
            val_episode_specs = build_episodic_episodes_common_query(
                frame,
                val_episodic_eligible,
                int(support_size),
                query_support_size=episodic_val_query_support_size,
                seed=int(episodic_val_cfg["seed"]),
                repeat_index=0,
            )
        else:
            val_episode_specs = build_episodic_episodes(
                frame,
                val_episodic_eligible,
                int(support_size),
                seed=int(episodic_val_cfg["seed"]),
                repeat_index=0,
            )
        val_specs_source = "episodic"
        if is_main:
            print(
                f"[CABRIA] val_protocol=episodic: user_source={episodic_val_cfg['user_source']}, "
                f"eligible={len(val_episodic_eligible)}/{len(val_episodic_users)}, "
                f"episodes={len(val_episode_specs)}, val_repeats={episodic_val_cfg['repeats']}, "
                f"seed={episodic_val_cfg['seed']}, common_query={episodic_val_common_query}, "
                f"query_support_size={episodic_val_query_support_size}"
            )
    else:
        seed_override = int(fixed_episode_seed) if fixed_episode_seed is not None else None
        built = build_fixed_query_val_episodes(
            frame,
            val_user_ids,
            int(support_size),
            eval_cfg=eval_cfg,
            support_seed=seed_override,
        )
        if built is None:
            if is_main:
                print(
                    "[CABRIA] val_protocol=fixed_query produced zero val episodes "
                    "(try lowering evaluation.fixed_query_min_images or val_protocol: manifest); "
                    "falling back to manifest val."
                )
            val_episode_specs = episodes["val"][support_size]
            val_specs_source = "manifest_fallback"
        else:
            val_episode_specs = built
            val_specs_source = "fixed_query"
    val_episodes_are_fixed_query = val_specs_source == "fixed_query"
    if is_main:
        print(f"[CABRIA] Stage2 val episode source={val_specs_source}, count={len(val_episode_specs)}")
    val_dataset_full = make_dataset(
        val_episode_specs,
        train=False,
        max_query_size=config["data"].get("eval_query_size"),
    )
    val_dataset = _rank_subset(val_dataset_full, distributed=distributed, rank=rank, world_size=world_size)
    val_loader = build_episode_dataloader(
        val_dataset,
        shuffle=False,
        num_workers=eval_num_workers,
        persistent_workers=dataloader_persistent_workers,
        prefetch_factor=dataloader_prefetch_factor,
    )
    print(
        "[CABRIA] Stage2 dataloaders ready: "
        f"train_episodes={len(train_dataset)}, "
        f"val_episodes={len(val_dataset_full)}, "
        f"train_num_workers={config['data']['num_workers']}, "
        f"eval_num_workers={eval_num_workers}, "
        f"prefetch_factor={dataloader_prefetch_factor}, "
        f"cache_resized_images={bool(config['data'].get('cache_resized_images', False))}, "
        f"image_cache_size={int(config['data'].get('image_cache_size', 0))}, "
        f"val_rank_episodes={len(val_dataset)}, "
        f"dynamic_train_episodes={dynamic_train_episodes}, "
        f"train_episodes_per_user={train_episodes_per_user}"
    )

    def _average_metric_rows(metric_rows: list[dict[str, float]]) -> dict[str, float]:
        keys = sorted({key for row in metric_rows for key in row.keys()})
        averaged: dict[str, float] = {}
        for key in keys:
            values = [float(row[key]) for row in metric_rows if key in row and row[key] == row[key]]
            if values:
                averaged[key] = sum(values) / len(values)
        return averaged

    def gather_predictions(local_predictions: pd.DataFrame) -> pd.DataFrame:
        if not distributed:
            return local_predictions
        gathered = [None for _ in range(world_size)] if is_main else None
        dist.gather_object(local_predictions, gathered, dst=0)
        if not is_main:
            return pd.DataFrame()
        frames = [frame for frame in gathered if frame is not None and not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    selection_eval_config = config
    if "selection_tta_enabled" in eval_cfg:
        selection_eval_config = copy.deepcopy(config)
        selection_eval_config.setdefault("test_time_adaptation", {})["enabled"] = bool(
            eval_cfg["selection_tta_enabled"]
        )
        if is_main:
            state = "enabled" if selection_eval_config["test_time_adaptation"]["enabled"] else "disabled"
            print(f"[CABRIA] Stage2 validation checkpoint selection TTA {state}.")

    def evaluate_validation(epoch: int) -> tuple[dict[str, float], pd.DataFrame]:
        acc_threshold = eval_cfg.get("acc_threshold")
        val_users_list = val_user_ids
        repeats_eff = 1 if fixed_episode_seed is not None else int(selection_repeats)

        if val_episodes_are_episodic:
            episodic_val_cfg = resolve_episodic_val_settings(eval_cfg)
            repeats_eff = int(episodic_val_cfg["repeats"])
            val_users = resolve_episodic_test_users(
                frame,
                user_source=str(episodic_val_cfg["user_source"]),
                split_users=val_users_list,
                test_users_file=episodic_val_cfg["test_users_file"],
            )
            eligibility_support_size = episodic_val_query_support_size if episodic_val_common_query else int(support_size)
            eligible = filter_episodic_eligible_users(frame, val_users, eligibility_support_size)
            if not eligible:
                raise RuntimeError(
                    f"val_protocol=episodic produced zero eligible users for support_size={support_size}."
                )
            metric_rows: list[dict[str, float]] = []
            prediction_frames: list[pd.DataFrame] = []
            seed_base = int(episodic_val_cfg["seed"])
            for repeat in range(repeats_eff):
                if episodic_val_common_query:
                    repeat_episodes = build_episodic_episodes_common_query(
                        frame,
                        eligible,
                        int(support_size),
                        query_support_size=episodic_val_query_support_size,
                        seed=seed_base,
                        repeat_index=repeat,
                    )
                else:
                    repeat_episodes = build_episodic_episodes(
                        frame,
                        eligible,
                        int(support_size),
                        seed=seed_base,
                        repeat_index=repeat,
                    )
                if diagnostics_max_val_users > 0:
                    repeat_episodes = repeat_episodes[:diagnostics_max_val_users]
                if not repeat_episodes:
                    continue
                repeat_dataset_full = make_dataset(
                    repeat_episodes,
                    train=False,
                    max_query_size=config["data"].get("eval_query_size"),
                )
                repeat_dataset = _rank_subset(
                    repeat_dataset_full,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                )
                repeat_loader = build_episode_dataloader(
                    repeat_dataset,
                    shuffle=False,
                    num_workers=eval_num_workers,
                    persistent_workers=False,
                    prefetch_factor=dataloader_prefetch_factor,
                )
                _repeat_metrics, local_predictions = evaluate_stage2_decomposed(
                    model_core,
                    repeat_loader,
                    device,
                    acc_threshold=acc_threshold,
                    config=selection_eval_config,
                )
                repeat_predictions = gather_predictions(local_predictions)
                if not is_main:
                    continue
                repeat_metrics = _metrics_from_predictions(repeat_predictions, acc_threshold=acc_threshold)
                metric_rows.append(repeat_metrics)
                repeat_predictions = repeat_predictions.copy()
                repeat_predictions["repeat"] = repeat
                repeat_predictions["seed"] = seed_base + repeat
                prediction_frames.append(repeat_predictions)
            if not is_main:
                return {}, pd.DataFrame()
            if not metric_rows:
                raise RuntimeError("val_protocol=episodic produced no metric rows.")
            averaged_metrics = _average_metric_rows(metric_rows)
            averaged_metrics["selection_repeats"] = float(repeats_eff)
            averaged_metrics["episodic_val_users"] = float(len(eligible))
            return averaged_metrics, pd.concat(prediction_frames, ignore_index=True)

        if repeats_eff <= 1:
            _local_metrics, local_predictions = evaluate_stage2_decomposed(
                model_core,
                val_loader,
                device,
                acc_threshold=acc_threshold,
                config=selection_eval_config,
            )
            predictions = gather_predictions(local_predictions)
            if not is_main:
                return {}, pd.DataFrame()
            return _metrics_from_predictions(predictions, acc_threshold=acc_threshold), predictions

        metric_rows: list[dict[str, float]] = []
        prediction_frames: list[pd.DataFrame] = []
        fq = resolve_fixed_query_settings(eval_cfg)
        users_v = list(val_users_list)
        if val_episodes_are_fixed_query and fq.same_user_pool:
            user_pool_support_size = (
                max(int(support_size), int(fq.user_pool_support_size))
                if fq.user_pool_support_size is not None
                else int(support_size)
            )
            users_v = filter_users_by_min_row_count(frame, users_v, fq.min_query_images + user_pool_support_size)
        for repeat in range(repeats_eff):
            repeat_seed = base_seed + 2003 + repeat
            if val_episodes_are_fixed_query:
                repeat_episodes = build_fixed_query_episodes(
                    frame,
                    users_v,
                    int(support_size),
                    holdout_seed=fq.holdout_seed,
                    support_seed=repeat_seed,
                    min_query_images=fq.min_query_images,
                )
                if not repeat_episodes:
                    repeat_episodes = build_episode_specs(
                        frame,
                        val_users_list,
                        [int(support_size)],
                        seed=repeat_seed,
                        episodes_per_user=1,
                    )[support_size]
            else:
                repeat_episodes = build_episode_specs(
                    frame,
                    val_users_list,
                    [int(support_size)],
                    seed=repeat_seed,
                    episodes_per_user=1,
                )[support_size]
            repeat_dataset_full = make_dataset(
                repeat_episodes,
                train=False,
                max_query_size=config["data"].get("eval_query_size"),
            )
            repeat_dataset = _rank_subset(
                repeat_dataset_full,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
            )
            repeat_loader = build_episode_dataloader(
                repeat_dataset,
                shuffle=False,
                num_workers=eval_num_workers,
                persistent_workers=False,
                prefetch_factor=dataloader_prefetch_factor,
            )
            _repeat_metrics, local_predictions = evaluate_stage2_decomposed(
                model_core,
                repeat_loader,
                device,
                acc_threshold=acc_threshold,
                config=selection_eval_config,
            )
            repeat_predictions = gather_predictions(local_predictions)
            if not is_main:
                continue
            repeat_metrics = _metrics_from_predictions(repeat_predictions, acc_threshold=acc_threshold)
            metric_rows.append(repeat_metrics)
            repeat_predictions = repeat_predictions.copy()
            repeat_predictions["repeat"] = repeat
            repeat_predictions["seed"] = repeat_seed
            prediction_frames.append(repeat_predictions)
        if not is_main:
            return {}, pd.DataFrame()
        averaged_metrics = _average_metric_rows(metric_rows)
        averaged_metrics["selection_repeats"] = float(repeats_eff)
        return averaged_metrics, pd.concat(prediction_frames, ignore_index=True)

    def selection_score(metrics: dict[str, float]) -> float:
        score = float(metrics.get("macro_srcc", float("-inf")))
        gain = float(metrics.get("gain_vs_base_macro_srcc", 0.0))
        if selection_gain_weight > 0.0:
            score += selection_gain_weight * gain
        if selection_same_image_weight > 0.0:
            score += selection_same_image_weight * float(metrics.get("same_image_srcc", 0.0))
        if selection_residual_corr_weight > 0.0:
            residual_corr = float(metrics.get(selection_residual_corr_metric, 0.0))
            if residual_corr == residual_corr:
                score += selection_residual_corr_weight * residual_corr
        if selection_base_penalty > 0.0:
            score += selection_base_penalty * min(0.0, gain)
        return score

    regression_loss = GeneralRegressionLoss(loss_name=config["loss"].get("regression", "smooth_l1"))
    dist_loss = None
    if "lambda_distribution" in config["loss"]:
        dist_cfg = config["loss"]
        dist_mode = str(dist_cfg.get("dist_mode", "softmax")).lower()
        score_values = sorted(frame["score"].astype(float).unique().tolist())
        max_discrete_bins = int(dist_cfg.get("dist_max_discrete_bins", 20))
        if dist_mode == "softmax":
            dist_loss = DistributionAlignmentLoss(
                mode="softmax",
                temperature=float(dist_cfg.get("dist_temperature", 1.0)),
            )
            print(f"[CABRIA] Stage2 distribution mode: softmax")
            print(f"[CABRIA] Stage2 distribution temperature: {dist_loss.temperature}")
        elif len(score_values) <= max_discrete_bins:
            dist_loss = DistributionAlignmentLoss(
                mode="histogram",
                score_values=score_values,
                sigma=float(dist_cfg.get("dist_bin_sigma", 0.5)),
            )
            print(f"[CABRIA] Stage2 distribution mode: histogram")
            print(f"[CABRIA] Stage2 distribution bins: {dist_loss.describe_bins()}")
            print(f"[CABRIA] Stage2 distribution sigma: {dist_loss.sigma}")
        else:
            dist_loss = DistributionAlignmentLoss(
                mode="histogram",
                score_min=float(dist_cfg.get("dist_score_min", min(score_values))),
                score_max=float(dist_cfg.get("dist_score_max", max(score_values))),
                num_bins=int(dist_cfg.get("dist_num_bins", 11)),
                sigma=float(dist_cfg.get("dist_bin_sigma", 0.5)),
            )
            print(f"[CABRIA] Stage2 distribution mode: histogram")
            print(f"[CABRIA] Stage2 distribution bins: {dist_loss.describe_bins()}")
            print(f"[CABRIA] Stage2 distribution sigma: {dist_loss.sigma}")
    else:
        print("[CABRIA] Stage2 distribution loss disabled (lambda_distribution omitted).")
    trainable_patterns = list(config["optimization"].get("trainable_patterns", []))
    if trainable_patterns:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        matched_names: list[str] = []
        for name, parameter in model.named_parameters():
            if any(pattern in name for pattern in trainable_patterns):
                parameter.requires_grad_(True)
                matched_names.append(name)
        if not matched_names:
            raise RuntimeError(f"No Stage2 trainable parameters matched patterns={trainable_patterns!r}.")
        if is_main:
            print(
                "[CABRIA] Stage2 trainable pattern override: "
                f"patterns={trainable_patterns}, tensors={len(matched_names)}"
            )
    base_lr = float(config["optimization"]["lr"])
    weight_decay = float(config["optimization"]["weight_decay"])
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=base_lr, weight_decay=weight_decay)
    scheduler = EpochLRScheduler(
        optimizer,
        total_epochs=int(config["optimization"]["epochs"]),
        config=config["optimization"].get("scheduler"),
    )
    early_stopping = EarlyStopping.from_config(config["optimization"].get("early_stopping"))
    amp_cfg = config.get("amp", {})
    use_amp = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(amp_cfg.get("dtype", "bfloat16")).lower() == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    log_every_steps = int(config.get("tracking", {}).get("log_every_steps", 20))
    enable_residual_target = "lambda_residual_target" in config["loss"]
    retrieved_residual_target_weight = float(config["loss"].get("lambda_retrieved_residual_target", 0.0))
    enable_retrieved_residual_target = retrieved_residual_target_weight > 0.0
    support_estimator_residual_target_weight = float(
        config["loss"].get("lambda_support_estimator_residual_target", 0.0)
    )
    support_estimator_residual_ranking_weight = float(
        config["loss"].get("lambda_support_estimator_residual_ranking", 0.0)
    )
    enable_support_estimator_residual_target = support_estimator_residual_target_weight > 0.0
    enable_support_estimator_residual_ranking = support_estimator_residual_ranking_weight > 0.0
    enable_same_image_order = "lambda_same_image_user_order" in config["loss"]
    enable_support_ranking = "lambda_support_ranking" in config["loss"]
    support_ranking_weight = float(config["loss"].get("lambda_support_ranking", 0.0))
    support_reconstruction_weight = float(config["loss"].get("lambda_support_reconstruction", 0.0))
    support_loo_max_items = int(config["loss"].get("support_loo_max_items", 0))
    support_anchor_weight = float(config["loss"].get("lambda_support_anchor_ranking", 0.0))
    enable_support_anchor_ranking = support_anchor_weight > 0.0
    support_anchor_source = str(config["loss"].get("support_anchor_source", "labels")).lower()
    support_query_alignment_weight = float(config["loss"].get("lambda_support_query_residual_alignment", 0.0))
    enable_support_query_alignment = support_query_alignment_weight > 0.0
    support_query_alignment_feature_temperature = float(
        config["loss"].get("support_query_alignment_feature_temperature", 0.07)
    )
    support_query_alignment_residual_temperature = float(
        config["loss"].get("support_query_alignment_residual_temperature", 0.25)
    )
    query_ranking_mode = str(config["loss"].get("query_ranking_mode", "softplus_margin"))
    support_ranking_mode = str(config["loss"].get("support_ranking_mode", query_ranking_mode))
    query_ranking_max_pairs = int(config["loss"].get("query_ranking_max_pairs", 0))
    support_ranking_max_pairs = int(config["loss"].get("support_ranking_max_pairs", 0))
    rank_improvement_weight = float(config["loss"].get("lambda_rank_improvement", 0.0))
    rank_improvement_margin = float(config["loss"].get("rank_improvement_margin", 0.0))
    enable_rank_improvement = rank_improvement_weight > 0.0
    query_residual_ranking_weight = float(config["loss"].get("lambda_query_residual_ranking", 0.0))
    support_residual_ranking_weight = float(config["loss"].get("lambda_support_residual_ranking", 0.0))
    enable_query_residual_ranking = query_residual_ranking_weight > 0.0
    enable_support_residual_ranking = support_residual_ranking_weight > 0.0
    enable_support_loo = (
        support_reconstruction_weight > 0.0
        or support_ranking_weight > 0.0
        or support_residual_ranking_weight > 0.0
        or (enable_support_anchor_ranking and support_anchor_source == "loo")
    )
    output_dir = ensure_dir(config["experiment"]["output_dir"])
    best_metric = float("-inf")
    checkpoint_label = _stage2_checkpoint_label(config, support_size)
    best_checkpoint = output_dir / f"best_stage2_{checkpoint_label}.pt"
    latest_checkpoint = output_dir / f"checkpoint_latest_{checkpoint_label}.pt"
    save_epoch_checkpoints = bool(config.get("experiment", {}).get("save_epoch_checkpoints", False))
    save_latest_checkpoint = bool(config.get("experiment", {}).get("save_latest_checkpoint", True))
    global_step = 0
    start_epoch = 1
    resume_checkpoint = _stage2_resume_path(config, output_dir, checkpoint_label)
    if resume_checkpoint is not None:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Stage2 resume checkpoint not found: {resume_checkpoint}")
        # Load on CPU so the checkpoint payload does not duplicate the model and
        # optimizer state in VRAM for the lifetime of this training function.
        payload = torch.load(resume_checkpoint, map_location="cpu")
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model_core.load_state_dict(state, strict=True)
        if isinstance(payload, dict):
            if "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            elif bool(config.get("experiment", {}).get("resume_strict_optimizer", False)):
                raise KeyError(f"Resume checkpoint has no optimizer state: {resume_checkpoint}")
            extra = payload.get("extra", {}) or {}
            scaler_state = extra.get("scaler")
            if scaler_state:
                scaler.load_state_dict(scaler_state)
            metrics = payload.get("metrics", {}) or {}
            best_metric = float(extra.get("best_metric", selection_score(metrics) if metrics else best_metric))
            global_step = int(extra.get("global_step", global_step))
            early_state = extra.get("early_stopping", {}) or {}
            early_stopping.best_metric = float(early_state.get("best_metric", early_stopping.best_metric))
            early_stopping.bad_epochs = int(early_state.get("bad_epochs", early_stopping.bad_epochs))
            early_stopping.should_stop = bool(early_state.get("should_stop", False))
            last_epoch = int(payload.get("epoch") or 0)
            start_epoch = last_epoch + 1
        del state, payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if is_main:
            print(
                "[CABRIA] Stage2 resumed checkpoint: "
                f"path={resume_checkpoint.as_posix()}, start_epoch={start_epoch}, "
                f"best_metric={best_metric:.4f}, global_step={global_step}"
            )
    print(
        "[CABRIA] Stage2 optimization: "
        f"epochs={config['optimization']['epochs']}, "
        f"lr={config['optimization']['lr']}, "
        f"use_amp={use_amp}, "
        f"output_dir={output_dir.as_posix()}, "
        f"checkpoint_label={checkpoint_label}"
    )
    tta_cfg = config.get("test_time_adaptation", {})
    use_train_tta = tta_enabled(config) and bool(tta_cfg.get("train_enabled", False))
    train_tta_every_n_steps = max(int(tta_cfg.get("train_every_n_steps", 1)), 1)
    train_tta_max_episodes_per_batch = int(tta_cfg.get("train_max_episodes_per_batch", 0))
    if tta_enabled(config):
        print(f"[CABRIA] Stage2 test-time adaptation enabled: {tta_cfg}")
    if use_train_tta:
        episode_limit = train_tta_max_episodes_per_batch if train_tta_max_episodes_per_batch > 0 else "all"
        print(
            "[CABRIA] Stage2 train-time TTA enabled: "
            f"every_n_steps={train_tta_every_n_steps}, "
            f"episodes_per_batch={episode_limit}."
        )
    else:
        print("[CABRIA] Stage2 train-time TTA disabled; TTA is applied only for validation/test.")
    backward_per_episode = bool(config.get("data", {}).get("backward_per_episode", False))
    if backward_per_episode and enable_same_image_order:
        print(
            "[CABRIA] Stage2 backward_per_episode disabled: incompatible with "
            "lambda_same_image_user_order (batch-coupled loss)."
        )
        backward_per_episode = False
    if backward_per_episode:
        print(
            "[CABRIA] Stage2 backward_per_episode enabled: sequential backward(loss / batch_size) "
            "per episode to reduce activation VRAM (gradients match joint batch-mean loss)."
        )
    print(
        "[CABRIA] Stage2 ranking: "
        f"query_mode={query_ranking_mode}, support_mode={support_ranking_mode}, "
        f"query_max_pairs={query_ranking_max_pairs}, support_max_pairs={support_ranking_max_pairs}"
    )
    print("[CABRIA] Stage2 entering training loop.")

    def _support_estimator_residual(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the support-derived residual components as one trainable estimator."""
        residual_score = outputs["residual_score"]
        estimator = outputs.get("residual_offset_score", residual_score.new_zeros(residual_score.shape))
        for key in (
            "bridge_residual_score",
            "bridge_basis_residual_score",
            "retrieved_residual_score",
            "kernel_residual_score",
            "user_state_residual_score",
        ):
            estimator = estimator + outputs.get(key, residual_score.new_zeros(residual_score.shape))
        return estimator

    def _weighted_scalar_loss_one_episode(
        episode: dict,
        episode_index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        """Forward one episode and build the same scalar loss as the joint path, without same-image term."""
        targets = episode["query_scores"].to(device, non_blocking=True)
        support_targets = episode["support_scores"].to(device, non_blocking=True)
        support_prior_scores = episode.get("support_prior_scores")
        query_prior_scores = episode.get("query_prior_scores")
        use_episode_train_tta = (
            use_train_tta
            and global_step % train_tta_every_n_steps == 0
            and (
                train_tta_max_episodes_per_batch <= 0
                or episode_index < train_tta_max_episodes_per_batch
            )
        )
        user_adaptation, _ = adapt_user_on_support(
            model=model_core,
            episode=episode,
            config=config,
            device=device,
            regression_loss=regression_loss,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            mode="train",
        ) if use_episode_train_tta else (None, {"tta_loss": 0.0})
        outputs = model(
            support_images={key: value.to(device, non_blocking=True) for key, value in episode["support_batch"].items()},
            support_scores=support_targets,
            query_images={key: value.to(device, non_blocking=True) for key, value in episode["query_batch"].items()},
            support_prior_scores=support_prior_scores.to(device, non_blocking=True) if support_prior_scores is not None else None,
            query_prior_scores=query_prior_scores.to(device, non_blocking=True) if query_prior_scores is not None else None,
            user_adaptation=user_adaptation,
            compute_support_loo=enable_support_loo,
            support_loo_indices=(
                torch.linspace(
                    0,
                    support_targets.numel() - 1,
                    steps=support_loo_max_items,
                    device=device,
                ).long()
                if 0 < support_loo_max_items < support_targets.numel()
                else None
            ),
        )
        reg_i = regression_loss(outputs["score"], targets)
        dist_i = dist_loss(outputs["score"], targets) if dist_loss is not None else None
        query_rank_i = pairwise_ranking_loss(
            outputs["score"],
            targets,
            mode=query_ranking_mode,
            margin=float(config["loss"].get("query_ranking_margin", 0.5)),
            max_pairs=query_ranking_max_pairs,
        )
        if enable_rank_improvement:
            base_rank_i = pairwise_ranking_loss(
                outputs["generic_score"].detach(),
                targets,
                mode=query_ranking_mode,
                margin=float(config["loss"].get("query_ranking_margin", 0.5)),
                max_pairs=query_ranking_max_pairs,
            )
            rank_improvement_i = F.relu(
                query_rank_i - base_rank_i + query_rank_i.new_tensor(rank_improvement_margin)
            )
        else:
            base_rank_i = query_rank_i.detach()
            rank_improvement_i = query_rank_i.new_zeros(())
        residual_l2_i = outputs["raw_residual_score"].pow(2).mean()
        target_residual = targets - outputs["generic_score"].detach()
        if enable_residual_target:
            residual_target_i = regression_loss(outputs["residual_score"], target_residual)
        else:
            residual_target_i = reg_i.new_zeros(())
        if enable_query_residual_ranking:
            query_residual_rank_i = pairwise_ranking_loss(
                outputs["residual_score"],
                target_residual,
                mode=query_ranking_mode,
                margin=float(
                    config["loss"].get(
                        "query_residual_ranking_margin",
                        config["loss"].get("query_ranking_margin", 0.5),
                    )
                ),
                max_pairs=query_ranking_max_pairs,
            )
        else:
            query_residual_rank_i = reg_i.new_zeros(())
        if enable_retrieved_residual_target:
            retrieved_residual = outputs.get("residual_offset_score", outputs["residual_score"].new_zeros(outputs["residual_score"].shape))
            retrieved_residual = retrieved_residual + outputs["retrieved_residual_score"]
            retrieved_residual_target_i = regression_loss(retrieved_residual, target_residual)
        else:
            retrieved_residual_target_i = reg_i.new_zeros(())
        support_estimator_residual_i = (
            _support_estimator_residual(outputs)
            if (enable_support_estimator_residual_target or enable_support_estimator_residual_ranking)
            else outputs["residual_score"].new_zeros(outputs["residual_score"].shape)
        )
        if enable_support_estimator_residual_target:
            support_estimator_residual_target_i = regression_loss(support_estimator_residual_i, target_residual)
        else:
            support_estimator_residual_target_i = reg_i.new_zeros(())
        if enable_support_estimator_residual_ranking:
            support_estimator_residual_rank_i = pairwise_ranking_loss(
                support_estimator_residual_i,
                target_residual,
                mode=query_ranking_mode,
                margin=float(
                    config["loss"].get(
                        "support_estimator_residual_ranking_margin",
                        config["loss"].get(
                            "query_residual_ranking_margin",
                            config["loss"].get("query_ranking_margin", 0.5),
                        ),
                    )
                ),
                max_pairs=query_ranking_max_pairs,
            )
        else:
            support_estimator_residual_rank_i = reg_i.new_zeros(())
        support_loo_targets = (
            support_targets.index_select(0, outputs["support_loo_indices"].long())
            if "support_loo_indices" in outputs
            else support_targets
        )
        support_reconstruction_i = regression_loss(outputs["support_loo_score"], support_loo_targets)
        support_loo_base = (
            outputs["support_generic_score"].index_select(0, outputs["support_loo_indices"].long())
            if "support_loo_indices" in outputs
            else outputs["support_generic_score"]
        )
        support_loo_residual_targets = support_loo_targets - support_loo_base.detach()
        if enable_support_residual_ranking:
            support_residual_rank_i = pairwise_ranking_loss(
                outputs["support_loo_raw_residual_score"],
                support_loo_residual_targets,
                mode=support_ranking_mode,
                margin=float(
                    config["loss"].get(
                        "support_residual_ranking_margin",
                        config["loss"].get(
                            "support_ranking_margin",
                            config["loss"].get("query_ranking_margin", 0.5),
                        ),
                    )
                ),
                max_pairs=support_ranking_max_pairs,
            )
        else:
            support_residual_rank_i = reg_i.new_zeros(())
        calibration_l2_i = (
            outputs["calibration_stage1_delta_adjust"].pow(2).mean()
            + (outputs["calibration_residual_gate"] - 1.0).pow(2).mean()
            + outputs["calibration_bias"].pow(2).mean()
        )
        if enable_support_ranking and support_ranking_weight > 0.0:
            support_rank_i = pairwise_ranking_loss(
                outputs["support_loo_score"],
                support_loo_targets,
                mode=support_ranking_mode,
                margin=float(config["loss"].get("support_ranking_margin", config["loss"].get("query_ranking_margin", 0.5))),
                max_pairs=support_ranking_max_pairs,
            )
        else:
            support_rank_i = reg_i.new_zeros(())
        if enable_support_anchor_ranking:
            anchor_source = str(config["loss"].get("support_anchor_source", "labels")).lower()
            if anchor_source == "labels":
                support_anchor_scores = support_targets.detach()
            elif anchor_source == "support_score":
                support_anchor_scores = outputs["support_score"].detach()
            elif anchor_source == "loo":
                support_anchor_scores = outputs["support_loo_score"].detach()
                support_anchor_targets = support_loo_targets
            else:
                raise ValueError(f"Unknown support_anchor_source: {anchor_source}")
            if anchor_source != "loo":
                support_anchor_targets = support_targets
            support_anchor_i = _support_anchored_query_ranking_loss(
                outputs["score"],
                targets,
                support_anchor_scores,
                support_anchor_targets,
                min_delta=float(config["loss"].get("support_anchor_min_delta", 0.15)),
                max_pairs=int(config["loss"].get("support_anchor_max_pairs", 4096)),
            )
        else:
            support_anchor_i = reg_i.new_zeros(())
        if enable_support_query_alignment:
            support_query_alignment_i = _support_query_residual_alignment_loss(
                query_embeddings=outputs["personalized_representation"],
                support_embeddings=outputs["support_personalized_representation"],
                query_targets=targets,
                support_targets=support_targets,
                query_base=outputs["generic_score"],
                support_base=outputs["support_generic_score"],
                feature_temperature=support_query_alignment_feature_temperature,
                residual_temperature=support_query_alignment_residual_temperature,
            )
        else:
            support_query_alignment_i = reg_i.new_zeros(())

        loss_i = (
            reg_i * float(config["loss"]["lambda_regression"])
            + query_rank_i * float(config["loss"].get("lambda_query_ranking", 0.0))
            + residual_l2_i * float(config["loss"].get("lambda_residual_l2", 0.0))
            + support_reconstruction_i * support_reconstruction_weight
            + calibration_l2_i * float(config["loss"].get("lambda_calibration_l2", 0.0))
        )
        if dist_i is not None:
            loss_i = loss_i + dist_i * float(config["loss"]["lambda_distribution"])
        if enable_residual_target:
            loss_i = loss_i + residual_target_i * float(config["loss"]["lambda_residual_target"])
        if enable_retrieved_residual_target:
            loss_i = loss_i + retrieved_residual_target_i * retrieved_residual_target_weight
        if enable_support_estimator_residual_target:
            loss_i = loss_i + support_estimator_residual_target_i * support_estimator_residual_target_weight
        if enable_support_estimator_residual_ranking:
            loss_i = loss_i + support_estimator_residual_rank_i * support_estimator_residual_ranking_weight
        if enable_support_ranking:
            loss_i = loss_i + support_rank_i * support_ranking_weight
        if enable_support_anchor_ranking:
            loss_i = loss_i + support_anchor_i * support_anchor_weight
        if enable_support_query_alignment:
            loss_i = loss_i + support_query_alignment_i * support_query_alignment_weight
        if enable_rank_improvement:
            loss_i = loss_i + rank_improvement_i * rank_improvement_weight
        if enable_query_residual_ranking:
            loss_i = loss_i + query_residual_rank_i * query_residual_ranking_weight
        if enable_support_residual_ranking:
            loss_i = loss_i + support_residual_rank_i * support_residual_ranking_weight
        parts = {
            "reg": float(reg_i.detach().item()),
            "dist": float(dist_i.detach().item()) if dist_i is not None else 0.0,
            "query_rank": float(query_rank_i.detach().item()),
            "rank_improvement": float(rank_improvement_i.detach().item()),
            "same_image": 0.0,
            "residual_l2": float(residual_l2_i.detach().item()),
            "residual_target": float(residual_target_i.detach().item()),
            "query_residual_rank": float(query_residual_rank_i.detach().item()),
            "retrieved_residual_target": float(retrieved_residual_target_i.detach().item()),
            "support_estimator_residual_target": float(support_estimator_residual_target_i.detach().item()),
            "support_estimator_residual_rank": float(support_estimator_residual_rank_i.detach().item()),
            "support_reconstruction": float(support_reconstruction_i.detach().item()),
            "support_rank": float(support_rank_i.detach().item()),
            "support_residual_rank": float(support_residual_rank_i.detach().item()),
            "support_anchor_rank": float(support_anchor_i.detach().item()),
            "support_query_alignment": float(support_query_alignment_i.detach().item()),
            "calibration_l2": float(calibration_l2_i.detach().item()),
        }
        return loss_i, outputs, parts

    for epoch in range(start_epoch, int(config["optimization"]["epochs"]) + 1):
        current_lr = scheduler.step(epoch)
        if dynamic_train_episodes:
            train_dataset, train_loader, train_sampler = build_train_loader_for_epoch(epoch)
            tqdm.write(
                f"[CABRIA] Stage2 resampled train episodes for epoch {epoch}: "
                f"episodes={len(train_dataset)}, episodes_per_user={train_episodes_per_user}"
            )
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        progress = tqdm(train_loader, desc=f"Stage2 Epoch {epoch}", leave=False, disable=not is_main)
        epoch_loss = 0.0
        epoch_reg = 0.0
        epoch_dist = 0.0
        epoch_query_rank = 0.0
        epoch_same_image_order = 0.0
        epoch_residual_l2 = 0.0
        epoch_residual_target = 0.0
        epoch_retrieved_residual_target = 0.0
        epoch_support_estimator_residual_target = 0.0
        epoch_support_estimator_residual_rank = 0.0
        epoch_query_residual_rank = 0.0
        epoch_support_reconstruction = 0.0
        epoch_support_rank = 0.0
        epoch_support_residual_rank = 0.0
        epoch_support_anchor_rank = 0.0
        epoch_calibration_l2 = 0.0
        epoch_support_query_alignment = 0.0
        num_batches = 0
        for batch in progress:
            episode_batch = batch if isinstance(batch, list) else [batch]
            optimizer.zero_grad(set_to_none=True)
            n_episodes = max(len(episode_batch), 1)
            if backward_per_episode:
                mean_parts: dict[str, float] = {
                    "reg": 0.0,
                    "dist": 0.0,
                    "query_rank": 0.0,
                    "rank_improvement": 0.0,
                    "residual_l2": 0.0,
                    "residual_target": 0.0,
                    "query_residual_rank": 0.0,
                    "retrieved_residual_target": 0.0,
                    "support_estimator_residual_target": 0.0,
                    "support_estimator_residual_rank": 0.0,
                    "support_reconstruction": 0.0,
                    "support_rank": 0.0,
                    "support_residual_rank": 0.0,
                    "support_anchor_rank": 0.0,
                    "support_query_alignment": 0.0,
                    "calibration_l2": 0.0,
                }
                total_loss_float = 0.0
                last_outputs: dict[str, torch.Tensor] | None = None
                for episode_index, episode in enumerate(episode_batch):
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        loss_i, outputs, parts = _weighted_scalar_loss_one_episode(
                            episode,
                            episode_index,
                        )
                    last_outputs = outputs
                    total_loss_float += float(loss_i.detach().item())
                    for key in mean_parts:
                        mean_parts[key] += float(parts.get(key, 0.0))
                    scaled = loss_i / float(n_episodes)
                    if scaler.is_enabled():
                        scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                inv_n = 1.0 / float(n_episodes)
                reg_value = torch.tensor(mean_parts["reg"] * inv_n, device=device, dtype=torch.float32)
                dist_value = torch.tensor(mean_parts["dist"] * inv_n, device=device, dtype=torch.float32)
                query_rank_value = torch.tensor(mean_parts["query_rank"] * inv_n, device=device, dtype=torch.float32)
                rank_improvement_value = torch.tensor(mean_parts["rank_improvement"] * inv_n, device=device, dtype=torch.float32)
                residual_l2_value = torch.tensor(mean_parts["residual_l2"] * inv_n, device=device, dtype=torch.float32)
                residual_target_value = torch.tensor(mean_parts["residual_target"] * inv_n, device=device, dtype=torch.float32)
                retrieved_residual_target_value = torch.tensor(mean_parts["retrieved_residual_target"] * inv_n, device=device, dtype=torch.float32)
                support_estimator_residual_target_value = torch.tensor(
                    mean_parts["support_estimator_residual_target"] * inv_n,
                    device=device,
                    dtype=torch.float32,
                )
                support_estimator_residual_rank_value = torch.tensor(
                    mean_parts["support_estimator_residual_rank"] * inv_n,
                    device=device,
                    dtype=torch.float32,
                )
                query_residual_rank_value = torch.tensor(mean_parts["query_residual_rank"] * inv_n, device=device, dtype=torch.float32)
                support_reconstruction_value = torch.tensor(mean_parts["support_reconstruction"] * inv_n, device=device, dtype=torch.float32)
                calibration_l2_value = torch.tensor(mean_parts["calibration_l2"] * inv_n, device=device, dtype=torch.float32)
                support_rank_value = torch.tensor(mean_parts["support_rank"] * inv_n, device=device, dtype=torch.float32)
                support_residual_rank_value = torch.tensor(mean_parts["support_residual_rank"] * inv_n, device=device, dtype=torch.float32)
                support_anchor_rank_value = torch.tensor(mean_parts["support_anchor_rank"] * inv_n, device=device, dtype=torch.float32)
                support_query_alignment_value = torch.tensor(mean_parts["support_query_alignment"] * inv_n, device=device, dtype=torch.float32)
                same_image_order_value = torch.tensor(0.0, device=device, dtype=torch.float32)
                total_loss = torch.tensor(total_loss_float * inv_n, device=device, dtype=torch.float32)
                outputs = last_outputs if last_outputs is not None else {}
            else:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    reg_values = []
                    dist_values = []
                    query_rank_values = []
                    residual_l2_values = []
                    rank_improvement_values = []
                    residual_target_values = []
                    query_residual_rank_values = []
                    retrieved_residual_target_values = []
                    support_estimator_residual_target_values = []
                    support_estimator_residual_rank_values = []
                    support_reconstruction_values = []
                    support_rank_values = []
                    support_residual_rank_values = []
                    support_anchor_rank_values = []
                    support_query_alignment_values = []
                    calibration_l2_values = []
                    all_predictions = []
                    all_targets = []
                    all_image_ids: list[str] = []
                    all_user_ids: list[str] = []
                    for episode_index, episode in enumerate(episode_batch):
                        targets = episode["query_scores"].to(device, non_blocking=True)
                        support_targets = episode["support_scores"].to(device, non_blocking=True)
                        support_prior_scores = episode.get("support_prior_scores")
                        query_prior_scores = episode.get("query_prior_scores")
                        use_episode_train_tta = (
                            use_train_tta
                            and global_step % train_tta_every_n_steps == 0
                            and (
                                train_tta_max_episodes_per_batch <= 0
                                or episode_index < train_tta_max_episodes_per_batch
                            )
                        )
                        user_adaptation, _ = adapt_user_on_support(
                            model=model_core,
                            episode=episode,
                            config=config,
                            device=device,
                            regression_loss=regression_loss,
                            use_amp=use_amp,
                            amp_dtype=amp_dtype,
                            mode="train",
                        ) if use_episode_train_tta else (None, {"tta_loss": 0.0})
                        outputs = model(
                            support_images={key: value.to(device, non_blocking=True) for key, value in episode["support_batch"].items()},
                            support_scores=support_targets,
                            query_images={key: value.to(device, non_blocking=True) for key, value in episode["query_batch"].items()},
                            support_prior_scores=support_prior_scores.to(device, non_blocking=True) if support_prior_scores is not None else None,
                            query_prior_scores=query_prior_scores.to(device, non_blocking=True) if query_prior_scores is not None else None,
                            user_adaptation=user_adaptation,
                            compute_support_loo=enable_support_loo,
                            support_loo_indices=(
                                torch.linspace(
                                    0,
                                    support_targets.numel() - 1,
                                    steps=support_loo_max_items,
                                    device=device,
                                ).long()
                                if 0 < support_loo_max_items < support_targets.numel()
                                else None
                            ),
                        )
                        reg_values.append(regression_loss(outputs["score"], targets))
                        if dist_loss is not None:
                            dist_values.append(dist_loss(outputs["score"], targets))
                        query_rank_value_i = pairwise_ranking_loss(
                            outputs["score"],
                            targets,
                            mode=query_ranking_mode,
                            margin=float(config["loss"].get("query_ranking_margin", 0.5)),
                            max_pairs=query_ranking_max_pairs,
                        )
                        query_rank_values.append(query_rank_value_i)
                        if enable_rank_improvement:
                            base_rank_value_i = pairwise_ranking_loss(
                                outputs["generic_score"].detach(),
                                targets,
                                mode=query_ranking_mode,
                                margin=float(config["loss"].get("query_ranking_margin", 0.5)),
                                max_pairs=query_ranking_max_pairs,
                            )
                            rank_improvement_values.append(
                                F.relu(
                                    query_rank_value_i
                                    - base_rank_value_i
                                    + query_rank_value_i.new_tensor(rank_improvement_margin)
                                )
                            )
                        residual_l2_values.append(outputs["raw_residual_score"].pow(2).mean())
                        target_residual = targets - outputs["generic_score"].detach()
                        if enable_residual_target:
                            residual_target_values.append(regression_loss(outputs["residual_score"], target_residual))
                        if enable_query_residual_ranking:
                            query_residual_rank_values.append(
                                pairwise_ranking_loss(
                                    outputs["residual_score"],
                                    target_residual,
                                    mode=query_ranking_mode,
                                    margin=float(
                                        config["loss"].get(
                                            "query_residual_ranking_margin",
                                            config["loss"].get("query_ranking_margin", 0.5),
                                        )
                                    ),
                                    max_pairs=query_ranking_max_pairs,
                                )
                            )
                        if enable_retrieved_residual_target:
                            retrieved_residual = outputs.get("residual_offset_score", outputs["residual_score"].new_zeros(outputs["residual_score"].shape))
                            retrieved_residual = retrieved_residual + outputs["retrieved_residual_score"]
                            retrieved_residual_target_values.append(regression_loss(retrieved_residual, target_residual))
                        support_estimator_residual = (
                            _support_estimator_residual(outputs)
                            if (
                                enable_support_estimator_residual_target
                                or enable_support_estimator_residual_ranking
                            )
                            else outputs["residual_score"].new_zeros(outputs["residual_score"].shape)
                        )
                        if enable_support_estimator_residual_target:
                            support_estimator_residual_target_values.append(
                                regression_loss(support_estimator_residual, target_residual)
                            )
                        if enable_support_estimator_residual_ranking:
                            support_estimator_residual_rank_values.append(
                                pairwise_ranking_loss(
                                    support_estimator_residual,
                                    target_residual,
                                    mode=query_ranking_mode,
                                    margin=float(
                                        config["loss"].get(
                                            "support_estimator_residual_ranking_margin",
                                            config["loss"].get(
                                                "query_residual_ranking_margin",
                                                config["loss"].get("query_ranking_margin", 0.5),
                                            ),
                                        )
                                    ),
                                    max_pairs=query_ranking_max_pairs,
                                )
                            )
                        support_loo_targets = (
                            support_targets.index_select(0, outputs["support_loo_indices"].long())
                            if "support_loo_indices" in outputs
                            else support_targets
                        )
                        support_reconstruction_values.append(regression_loss(outputs["support_loo_score"], support_loo_targets))
                        support_loo_base = (
                            outputs["support_generic_score"].index_select(0, outputs["support_loo_indices"].long())
                            if "support_loo_indices" in outputs
                            else outputs["support_generic_score"]
                        )
                        support_loo_residual_targets = support_loo_targets - support_loo_base.detach()
                        if enable_support_residual_ranking:
                            support_residual_rank_values.append(
                                pairwise_ranking_loss(
                                    outputs["support_loo_raw_residual_score"],
                                    support_loo_residual_targets,
                                    mode=support_ranking_mode,
                                    margin=float(
                                        config["loss"].get(
                                            "support_residual_ranking_margin",
                                            config["loss"].get(
                                                "support_ranking_margin",
                                                config["loss"].get("query_ranking_margin", 0.5),
                                            ),
                                        )
                                    ),
                                    max_pairs=support_ranking_max_pairs,
                                )
                            )
                        calibration_l2_values.append(
                            outputs["calibration_stage1_delta_adjust"].pow(2).mean()
                            + (outputs["calibration_residual_gate"] - 1.0).pow(2).mean()
                            + outputs["calibration_bias"].pow(2).mean()
                        )
                        if enable_support_ranking and support_ranking_weight > 0.0:
                            support_rank_values.append(
                                pairwise_ranking_loss(
                                    outputs["support_loo_score"],
                                    support_loo_targets,
                                    mode=support_ranking_mode,
                                    margin=float(config["loss"].get("support_ranking_margin", config["loss"].get("query_ranking_margin", 0.5))),
                                    max_pairs=support_ranking_max_pairs,
                                )
                            )
                        if enable_support_anchor_ranking:
                            anchor_source = str(config["loss"].get("support_anchor_source", "labels")).lower()
                            if anchor_source == "labels":
                                support_anchor_scores = support_targets.detach()
                            elif anchor_source == "support_score":
                                support_anchor_scores = outputs["support_score"].detach()
                            elif anchor_source == "loo":
                                support_anchor_scores = outputs["support_loo_score"].detach()
                                support_anchor_targets = support_loo_targets
                            else:
                                raise ValueError(f"Unknown support_anchor_source: {anchor_source}")
                            if anchor_source != "loo":
                                support_anchor_targets = support_targets
                            support_anchor_rank_values.append(
                                _support_anchored_query_ranking_loss(
                                    outputs["score"],
                                    targets,
                                    support_anchor_scores,
                                    support_anchor_targets,
                                    min_delta=float(config["loss"].get("support_anchor_min_delta", 0.15)),
                                    max_pairs=int(config["loss"].get("support_anchor_max_pairs", 4096)),
                                )
                            )
                        if enable_support_query_alignment:
                            support_query_alignment_values.append(
                                _support_query_residual_alignment_loss(
                                    query_embeddings=outputs["personalized_representation"],
                                    support_embeddings=outputs["support_personalized_representation"],
                                    query_targets=targets,
                                    support_targets=support_targets,
                                    query_base=outputs["generic_score"],
                                    support_base=outputs["support_generic_score"],
                                    feature_temperature=support_query_alignment_feature_temperature,
                                    residual_temperature=support_query_alignment_residual_temperature,
                                )
                            )
                        if enable_same_image_order:
                            all_predictions.append(outputs["score"])
                            all_targets.append(targets)
                            all_image_ids.extend([str(image_id) for image_id in episode["query_image_ids"]])
                            all_user_ids.extend([str(episode["user_id"])] * len(episode["query_image_ids"]))

                    reg_value = torch.stack(reg_values).mean()
                    dist_value = torch.stack(dist_values).mean() if dist_values else reg_value.new_tensor(0.0)
                    query_rank_value = torch.stack(query_rank_values).mean()
                    rank_improvement_value = (
                        torch.stack(rank_improvement_values).mean()
                        if rank_improvement_values
                        else query_rank_value.new_tensor(0.0)
                    )
                    residual_l2_value = torch.stack(residual_l2_values).mean()
                    residual_target_value = (
                        torch.stack(residual_target_values).mean()
                        if residual_target_values
                        else reg_value.new_tensor(0.0)
                    )
                    retrieved_residual_target_value = (
                        torch.stack(retrieved_residual_target_values).mean()
                        if retrieved_residual_target_values
                        else reg_value.new_tensor(0.0)
                    )
                    support_estimator_residual_target_value = (
                        torch.stack(support_estimator_residual_target_values).mean()
                        if support_estimator_residual_target_values
                        else reg_value.new_tensor(0.0)
                    )
                    support_estimator_residual_rank_value = (
                        torch.stack(support_estimator_residual_rank_values).mean()
                        if support_estimator_residual_rank_values
                        else reg_value.new_tensor(0.0)
                    )
                    query_residual_rank_value = (
                        torch.stack(query_residual_rank_values).mean()
                        if query_residual_rank_values
                        else reg_value.new_tensor(0.0)
                    )
                    support_reconstruction_value = torch.stack(support_reconstruction_values).mean()
                    calibration_l2_value = torch.stack(calibration_l2_values).mean()
                    support_rank_value = (
                        torch.stack(support_rank_values).mean()
                        if support_rank_values
                        else reg_value.new_tensor(0.0)
                    )
                    support_residual_rank_value = (
                        torch.stack(support_residual_rank_values).mean()
                        if support_residual_rank_values
                        else reg_value.new_tensor(0.0)
                    )
                    support_anchor_rank_value = (
                        torch.stack(support_anchor_rank_values).mean()
                        if support_anchor_rank_values
                        else reg_value.new_tensor(0.0)
                    )
                    support_query_alignment_value = (
                        torch.stack(support_query_alignment_values).mean()
                        if support_query_alignment_values
                        else reg_value.new_tensor(0.0)
                    )
                    same_image_order_value = reg_value.new_tensor(0.0)
                    if enable_same_image_order:
                        flat_predictions = torch.cat(all_predictions, dim=0)
                        flat_targets = torch.cat(all_targets, dim=0)
                        same_image_order_value = same_image_user_order_loss(
                            predictions=flat_predictions,
                            targets=flat_targets,
                            image_ids=all_image_ids,
                            user_ids=all_user_ids,
                        )
                    total_loss = (
                        reg_value * float(config["loss"]["lambda_regression"])
                        + query_rank_value * float(config["loss"].get("lambda_query_ranking", 0.0))
                        + residual_l2_value * float(config["loss"].get("lambda_residual_l2", 0.0))
                        + support_reconstruction_value * support_reconstruction_weight
                        + calibration_l2_value * float(config["loss"].get("lambda_calibration_l2", 0.0))
                    )
                    if dist_loss is not None:
                        total_loss = total_loss + dist_value * float(config["loss"]["lambda_distribution"])
                    if enable_same_image_order:
                        total_loss = total_loss + same_image_order_value * float(config["loss"]["lambda_same_image_user_order"])
                    if enable_residual_target:
                        total_loss = total_loss + residual_target_value * float(config["loss"]["lambda_residual_target"])
                    if enable_retrieved_residual_target:
                        total_loss = total_loss + retrieved_residual_target_value * retrieved_residual_target_weight
                    if enable_support_estimator_residual_target:
                        total_loss = (
                            total_loss
                            + support_estimator_residual_target_value
                            * support_estimator_residual_target_weight
                        )
                    if enable_support_estimator_residual_ranking:
                        total_loss = (
                            total_loss
                            + support_estimator_residual_rank_value
                            * support_estimator_residual_ranking_weight
                        )
                    if enable_support_ranking:
                        total_loss = total_loss + support_rank_value * support_ranking_weight
                    if enable_support_anchor_ranking:
                        total_loss = total_loss + support_anchor_rank_value * support_anchor_weight
                    if enable_support_query_alignment:
                        total_loss = total_loss + support_query_alignment_value * support_query_alignment_weight
                    if enable_rank_improvement:
                        total_loss = total_loss + rank_improvement_value * rank_improvement_weight
                    if enable_query_residual_ranking:
                        total_loss = total_loss + query_residual_rank_value * query_residual_ranking_weight
                    if enable_support_residual_ranking:
                        total_loss = total_loss + support_residual_rank_value * support_residual_ranking_weight

                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()
            global_step += 1
            epoch_loss += float(total_loss.item())
            epoch_reg += float(reg_value.item())
            epoch_dist += float(dist_value.item())
            epoch_query_rank += float(query_rank_value.item())
            epoch_same_image_order += float(same_image_order_value.item())
            epoch_residual_l2 += float(residual_l2_value.item())
            epoch_residual_target += float(residual_target_value.item())
            epoch_retrieved_residual_target += float(retrieved_residual_target_value.item())
            epoch_support_estimator_residual_target += float(support_estimator_residual_target_value.item())
            epoch_support_estimator_residual_rank += float(support_estimator_residual_rank_value.item())
            epoch_query_residual_rank += float(query_residual_rank_value.item())
            epoch_support_reconstruction += float(support_reconstruction_value.item())
            epoch_support_rank += float(support_rank_value.item())
            epoch_support_residual_rank += float(support_residual_rank_value.item())
            epoch_support_anchor_rank += float(support_anchor_rank_value.item())
            epoch_support_query_alignment += float(support_query_alignment_value.item())
            epoch_calibration_l2 += float(calibration_l2_value.item())
            num_batches += 1

            if tracker is not None and global_step % max(log_every_steps, 1) == 0:
                tracker.log(
                    {
                        "train/global_step": global_step,
                        "train/loss": float(total_loss.item()),
                        "train/regression_loss": float(reg_value.item()),
                        "train/distribution_loss": float(dist_value.item()),
                        "train/query_ranking_loss": float(query_rank_value.item()),
                        "train/rank_improvement_loss": float(rank_improvement_value.item()),
                        "train/same_image_user_order_loss": float(same_image_order_value.item()),
                        "train/residual_l2_loss": float(residual_l2_value.item()),
                        "train/residual_target_loss": float(residual_target_value.item()),
                        "train/query_residual_ranking_loss": float(query_residual_rank_value.item()),
                        "train/retrieved_residual_target_loss": float(retrieved_residual_target_value.item()),
                        "train/support_estimator_residual_target_loss": float(
                            support_estimator_residual_target_value.item()
                        ),
                        "train/support_estimator_residual_ranking_loss": float(
                            support_estimator_residual_rank_value.item()
                        ),
                        "train/support_reconstruction_loss": float(support_reconstruction_value.item()),
                        "train/support_ranking_loss": float(support_rank_value.item()),
                        "train/support_residual_ranking_loss": float(support_residual_rank_value.item()),
                        "train/support_anchor_ranking_loss": float(support_anchor_rank_value.item()),
                        "train/support_query_residual_alignment_loss": float(support_query_alignment_value.item()),
                        "train/calibration_l2_loss": float(calibration_l2_value.item()),
                        "train/calibration_stage1_delta_adjust": float(outputs["calibration_stage1_delta_adjust"].mean().item()),
                        "train/calibration_residual_gate": float(outputs["calibration_residual_gate"].mean().item()),
                        "train/calibration_bias": float(outputs["calibration_bias"].mean().item()),
                        "train/support_size": int(support_size),
                        "optimization/lr": current_lr,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            progress.set_postfix(
                loss=f"{total_loss.item():.4f}",
                reg=f"{reg_value.item():.4f}",
                rank=f"{query_rank_value.item():.4f}",
                res_l2=f"{residual_l2_value.item():.4f}",
            )
            if diagnostics_max_train_batches > 0 and num_batches >= diagnostics_max_train_batches:
                tqdm.write(
                    "[CABRIA][PROFILE] Stopping training epoch after "
                    f"{num_batches} diagnostic batch(es)."
                )
                break

        epoch_totals = torch.tensor(
            [
                epoch_loss,
                epoch_reg,
                epoch_dist,
                epoch_query_rank,
                epoch_same_image_order,
                epoch_residual_l2,
                epoch_residual_target,
                epoch_retrieved_residual_target,
                epoch_support_estimator_residual_target,
                epoch_support_estimator_residual_rank,
                epoch_support_reconstruction,
                epoch_support_rank,
                epoch_support_anchor_rank,
                epoch_support_query_alignment,
                epoch_calibration_l2,
                float(num_batches),
            ],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(epoch_totals, op=dist.ReduceOp.SUM)
        denom = max(float(epoch_totals[-1].item()), 1.0)
        train_metrics = {
            "train/loss": float(epoch_totals[0].item()) / denom,
            "train/regression_loss": float(epoch_totals[1].item()) / denom,
            "train/distribution_loss": float(epoch_totals[2].item()) / denom,
            "train/query_ranking_loss": float(epoch_totals[3].item()) / denom,
            "train/same_image_user_order_loss": float(epoch_totals[4].item()) / denom,
            "train/residual_l2_loss": float(epoch_totals[5].item()) / denom,
            "train/residual_target_loss": float(epoch_totals[6].item()) / denom,
            "train/retrieved_residual_target_loss": float(epoch_totals[7].item()) / denom,
            "train/support_estimator_residual_target_loss": float(epoch_totals[8].item()) / denom,
            "train/support_estimator_residual_ranking_loss": float(epoch_totals[9].item()) / denom,
            "train/support_reconstruction_loss": float(epoch_totals[10].item()) / denom,
            "train/support_ranking_loss": float(epoch_totals[11].item()) / denom,
            "train/support_anchor_ranking_loss": float(epoch_totals[12].item()) / denom,
            "train/support_query_residual_alignment_loss": float(epoch_totals[13].item()) / denom,
            "train/calibration_l2_loss": float(epoch_totals[14].item()) / denom,
            "train/support_size": int(support_size),
            "train/use_amp": float(use_amp),
            "optimization/lr": current_lr,
        }

        # The next epoch starts with zero_grad(set_to_none=True), so releasing
        # the final training gradients and unused CUDA blocks here is
        # numerically equivalent and prevents WDDM paging during validation.
        progress.close()
        del progress
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        metrics, predictions = evaluate_validation(epoch)
        stop_now = False
        if is_main:
            eval_metrics = {f"val/{key}": value for key, value in metrics.items()}
            tqdm.write(
                f"[CABRIA] Stage2 Epoch {epoch}: "
                f"train_loss={train_metrics['train/loss']:.4f}, "
                f"macro_srcc={metrics.get('macro_srcc', float('nan')):.4f}, "
                f"macro_plcc={metrics.get('macro_plcc', float('nan')):.4f}, "
                f"same_image_srcc={metrics.get('same_image_srcc', float('nan')):.4f}, "
                f"base_macro_srcc={metrics.get('base_macro_srcc', float('nan')):.4f}, "
                f"retrieved_residual_corr={metrics.get('retrieved_residual_macro_spearman', float('nan')):.4f}, "
                f"res_l2={train_metrics['train/residual_l2_loss']:.4f}, "
                f"selection={selection_score(metrics):.4f}"
            )
            if tracker is not None:
                tracker.log(
                    {
                        **train_metrics,
                        **eval_metrics,
                        "optimization/early_stopping_bad_epochs": float(early_stopping.bad_epochs),
                        "optimization/early_stopping_patience": float(early_stopping.patience),
                        "epoch": epoch,
                    },
                    step=epoch,
                )
            metric = selection_score(metrics)
            latest_best_metric = max(best_metric, metric)
            if save_latest_checkpoint:
                save_checkpoint(
                    latest_checkpoint,
                    model_core,
                    optimizer=optimizer,
                    scheduler=scheduler.state_dict(),
                    epoch=epoch,
                    metrics=metrics,
                    extra={
                        "best_metric": latest_best_metric,
                        "global_step": global_step,
                        "scaler": scaler.state_dict(),
                        "early_stopping": {
                            "best_metric": early_stopping.best_metric,
                            "bad_epochs": early_stopping.bad_epochs,
                            "should_stop": early_stopping.should_stop,
                        },
                    },
                )
                if tracker is not None:
                    tracker.log_summary({"artifacts/latest_stage2_checkpoint": latest_checkpoint.as_posix()})
            if save_epoch_checkpoints:
                epoch_checkpoint = output_dir / f"checkpoint_epoch_{epoch}_{checkpoint_label}.pt"
                save_checkpoint(
                    epoch_checkpoint,
                    model_core,
                    optimizer=optimizer,
                    scheduler=scheduler.state_dict(),
                    epoch=epoch,
                    metrics=metrics,
                    extra={
                        "best_metric": latest_best_metric,
                        "global_step": global_step,
                        "scaler": scaler.state_dict(),
                        "early_stopping": {
                            "best_metric": early_stopping.best_metric,
                            "bad_epochs": early_stopping.bad_epochs,
                            "should_stop": early_stopping.should_stop,
                        },
                    },
                )
            if metric > best_metric:
                best_metric = metric
                save_checkpoint(
                    best_checkpoint,
                    model_core,
                    optimizer=optimizer,
                    scheduler=scheduler.state_dict(),
                    epoch=epoch,
                    metrics=metrics,
                    extra={
                        "best_metric": best_metric,
                        "global_step": global_step,
                        "scaler": scaler.state_dict(),
                        "early_stopping": {
                            "best_metric": early_stopping.best_metric,
                            "bad_epochs": early_stopping.bad_epochs,
                            "should_stop": early_stopping.should_stop,
                        },
                    },
                )
                predictions.to_csv(output_dir / f"val_predictions_{checkpoint_label}.csv", index=False)
                tqdm.write(f"[CABRIA] Stage2 saved new best checkpoint: {best_checkpoint.as_posix()}")
                if tracker is not None:
                    tracker.log_summary(
                        {
                            "best/stage2_macro_srcc": metrics.get("macro_srcc"),
                            "best/stage2_macro_plcc": metrics.get("macro_plcc"),
                            "best/stage2_same_image_srcc": metrics.get("same_image_srcc"),
                            "best/stage2_base_macro_srcc": metrics.get("base_macro_srcc"),
                            "best/stage2_gain_vs_base_macro_srcc": metrics.get("gain_vs_base_macro_srcc"),
                            "best/stage2_selection_score": metric,
                            "best/stage2_selection_repeats": metrics.get("selection_repeats", 1.0),
                            "best/stage2_support_size": int(support_size),
                            "best/stage2_checkpoint_label": checkpoint_label,
                            "best/stage2_epoch": epoch,
                            "artifacts/best_stage2_checkpoint": best_checkpoint.as_posix(),
                            "artifacts/best_stage2_predictions": (output_dir / f"val_predictions_{checkpoint_label}.csv").as_posix(),
                        }
                    )
                    tracker.log_table("best_val_predictions", predictions.to_dict(orient="records"))
            early_stopping.update(metric)
            stop_now = bool(early_stopping.should_stop)
            if stop_now:
                tqdm.write(
                    f"[CABRIA] Stage2 early stopping triggered at epoch {epoch} "
                    f"(patience={early_stopping.patience}, best_selection={early_stopping.best_metric:.4f})."
                )
                if tracker is not None:
                    tracker.log_summary(
                        {
                            "optimization/stage2_early_stopped": 1.0,
                            "optimization/stage2_stop_epoch": epoch,
                        }
                    )
        if distributed:
            stop_tensor = torch.tensor([1 if stop_now else 0], dtype=torch.int64, device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_now = bool(stop_tensor.item())
        if stop_now:
            break

    if is_main and bool(eval_cfg.get("print_repeated_test_hint", False)):
        fq = resolve_fixed_query_settings(eval_cfg)
        if fixed_episode_seed is not None:
            tqdm.write(
                "[CABRIA] Protocol: deterministic test (fixed-query holdout; match val support seed) via:\n"
                f"  PYTHONPATH=src python scripts/evaluate_repeated_support.py \\\n"
                "    --config <stage2_yaml> --data-config <data_yaml> \\\n"
                f"    --checkpoint {best_checkpoint.as_posix()} \\\n"
                f"    --split test --support-size {int(support_size)} --repeats 1 --base-seed {int(fixed_episode_seed)} \\\n"
                f"    --holdout-seed {fq.holdout_seed} --min-query-images {fq.min_query_images} \\\n"
                f"    --output-dir {output_dir.as_posix()}"
            )
        else:
            tqdm.write(
                "[CABRIA] Protocol: report test macro_srcc mean±std under fixed-query + resampled support via:\n"
                f"  PYTHONPATH=src python scripts/evaluate_repeated_support.py \\\n"
                "    --config <stage2_yaml> --data-config <data_yaml> \\\n"
                f"    --checkpoint {best_checkpoint.as_posix()} \\\n"
                f"    --split test --support-size {int(support_size)} --repeats 10 --base-seed 1000 \\\n"
                f"    --holdout-seed {fq.holdout_seed} --min-query-images {fq.min_query_images} \\\n"
                f"    --output-dir {output_dir.as_posix()}"
            )

    return best_checkpoint
