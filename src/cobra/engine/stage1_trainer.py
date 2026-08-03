from __future__ import annotations

from pathlib import Path
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from cobra.data.general_flickr_dataset import FlickrGeneralDataset, GeneralAestheticDataset, load_general_aesthetic_frame
from cobra.data.image_batch import collate_general_samples
from cobra.engine.evaluator import evaluate_stage1
from cobra.evaluation.stage1_fixed_query import (
    evaluate_stage1_fixed_query_base,
    stage1_checkpoint_selection_score,
)
from cobra.losses.general_regression import GeneralRegressionLoss
from cobra.losses.query_ranking import query_pairwise_ranking_loss
from cobra.models.attribute_token_extractor import attribute_diversity_loss
from cobra.models.cobra_model import COBRAStage1Model
from cobra.utils.checkpoint import save_checkpoint
from cobra.utils.common import ensure_dir
from cobra.utils.factory import build_model_config
from cobra.utils.optimization import EarlyStopping, EpochLRScheduler
from cobra.utils.tracking import Tracker


def _distributed_state() -> tuple[bool, int, int, int]:
    if not (dist.is_available() and dist.is_initialized()):
        return False, 0, 1, 0
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    return True, rank, world_size, local_rank


def _skipped_bad_image_count(dataset: Dataset) -> int:
    if hasattr(dataset, "get_skipped_bad_image_count"):
        return int(dataset.get_skipped_bad_image_count())
    if isinstance(dataset, ConcatDataset):
        return sum(_skipped_bad_image_count(child) for child in dataset.datasets)
    return 0


def _build_general_dataset(
    dataset_cfg: dict,
    split_name: str,
    target_mode: str,
    train: bool,
    common_dataset_kwargs: dict,
    default_seed: int,
) -> Dataset:
    dataset_name = str(dataset_cfg.get("name", "flickr_aes")).lower()
    if dataset_name == "flickr_aes":
        return FlickrGeneralDataset(
            dataset_root=dataset_cfg["root"],
            split_file=dataset_cfg["split_file"],
            train=train,
            split_name=split_name,
            target_mode=target_mode,
            split_strategy=str(dataset_cfg.get("split_strategy", "image_random")),
            val_ratio=float(dataset_cfg.get("val_ratio", 0.2)),
            seed=int(dataset_cfg.get("split_seed", default_seed)),
            limit=dataset_cfg.get("train_limit") if train else dataset_cfg.get("val_limit"),
            **common_dataset_kwargs,
        )
    frame = load_general_aesthetic_frame(
        dataset_name=dataset_name,
        dataset_root=dataset_cfg["root"],
        split_name=split_name,
        target_mode=target_mode,
        split_file=dataset_cfg.get("split_file"),
        split_strategy=str(dataset_cfg.get("split_strategy", "image_random")),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.2)),
        seed=int(dataset_cfg.get("split_seed", default_seed)),
    )
    return GeneralAestheticDataset(
        frame=frame,
        train=train,
        source_name=dataset_name,
        limit=dataset_cfg.get("train_limit") if train else dataset_cfg.get("val_limit"),
        **common_dataset_kwargs,
    )


def _build_general_datasets(
    data_config: dict,
    common_dataset_kwargs: dict,
    default_seed: int,
) -> tuple[Dataset, Dataset, Dataset | None, str]:
    if "general_datasets" not in data_config:
        dataset_cfg = data_config["general_dataset"]
        train_dataset = _build_general_dataset(
            dataset_cfg,
            split_name=str(dataset_cfg["train_split"]),
            target_mode=str(dataset_cfg["train_target_mode"]),
            train=True,
            common_dataset_kwargs=common_dataset_kwargs,
            default_seed=default_seed,
        )
        val_dataset = _build_general_dataset(
            dataset_cfg,
            split_name=str(dataset_cfg["val_split"]),
            target_mode=str(dataset_cfg["val_target_mode"]),
            train=False,
            common_dataset_kwargs=common_dataset_kwargs,
            default_seed=default_seed,
        )
        aux_val_dataset = None
        aux_val_mode = dataset_cfg.get("aux_val_target_mode")
        if aux_val_mode:
            aux_val_cfg = dict(dataset_cfg)
            aux_val_cfg["val_limit"] = dataset_cfg.get("aux_val_limit", dataset_cfg.get("val_limit"))
            aux_val_dataset = _build_general_dataset(
                aux_val_cfg,
                split_name=str(dataset_cfg["val_split"]),
                target_mode=str(aux_val_mode),
                train=False,
                common_dataset_kwargs=common_dataset_kwargs,
                default_seed=default_seed,
            )
        return train_dataset, val_dataset, aux_val_dataset, str(aux_val_mode or "")

    train_datasets: list[Dataset] = []
    val_datasets: list[Dataset] = []
    for source_cfg in data_config["general_datasets"]:
        train_datasets.append(
            _build_general_dataset(
                source_cfg,
                split_name=str(source_cfg.get("train_split", "train")),
                target_mode=str(source_cfg.get("train_target_mode", "image_mean")),
                train=True,
                common_dataset_kwargs=common_dataset_kwargs,
                default_seed=default_seed,
            )
        )
        val_datasets.append(
            _build_general_dataset(
                source_cfg,
                split_name=str(source_cfg.get("val_split", "val")),
                target_mode=str(source_cfg.get("val_target_mode", source_cfg.get("train_target_mode", "image_mean"))),
                train=False,
                common_dataset_kwargs=common_dataset_kwargs,
                default_seed=default_seed,
            )
        )
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets), None, ""


def train_stage1(config: dict, data_config: dict, device: torch.device, tracker: Tracker | None = None) -> Path:
    distributed, rank, world_size, local_rank = _distributed_state()
    is_main = rank == 0

    model = COBRAStage1Model(build_model_config(config)).to(device)
    model_core = model
    dataset_cfg = data_config.get("general_dataset", {})
    backbone_cfg = config["backbone"]
    optimization_cfg = config["optimization"]
    loss_cfg = config["loss"]
    amp_cfg = config.get("amp", {})
    use_amp = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(amp_cfg.get("dtype", "bfloat16")).lower() == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    log_every_steps = int(config.get("tracking", {}).get("log_every_steps", 100))

    common_dataset_kwargs = {
        "image_size": int(backbone_cfg["image_size"]),
        "mean": backbone_cfg.get("mean"),
        "std": backbone_cfg.get("std"),
        "use_naflex": bool(backbone_cfg.get("use_naflex", False)),
        "preferred_long_side": backbone_cfg.get("preferred_long_side"),
        "patch_size": int(backbone_cfg["patch_size"]),
        "max_num_patches": backbone_cfg.get("max_num_patches"),
    }
    split_seed = int(dataset_cfg.get("split_seed", config["experiment"].get("seed", 42)))
    train_dataset, val_dataset, aux_val_dataset, aux_val_mode = _build_general_datasets(
        data_config=data_config,
        common_dataset_kwargs=common_dataset_kwargs,
        default_seed=split_seed,
    )
    source_names = [str(cfg.get("name", "flickr_aes")) for cfg in data_config.get("general_datasets", [dataset_cfg])]
    print(
        "[COBRA] Stage1 GIAA data: "
        f"sources={source_names}, "
        f"aux_val_target={aux_val_mode or 'none'}, "
        f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}, "
        f"aux_val_samples={len(aux_val_dataset) if aux_val_dataset is not None else 0}"
    )

    base_seed = int(config["experiment"].get("seed", 42))
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=base_seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        collate_fn=partial(
            collate_general_samples,
            use_naflex=bool(backbone_cfg.get("use_naflex", False)),
            patch_size=int(backbone_cfg["patch_size"]),
            max_num_patches=backbone_cfg.get("max_num_patches"),
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        collate_fn=partial(
            collate_general_samples,
            use_naflex=bool(backbone_cfg.get("use_naflex", False)),
            patch_size=int(backbone_cfg["patch_size"]),
            max_num_patches=backbone_cfg.get("max_num_patches"),
        ),
    )
    aux_val_loader = None
    if aux_val_dataset is not None:
        aux_val_loader = DataLoader(
            aux_val_dataset,
            batch_size=int(config["data"]["eval_batch_size"]),
            shuffle=False,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=True,
            collate_fn=partial(
                collate_general_samples,
                use_naflex=bool(backbone_cfg.get("use_naflex", False)),
                patch_size=int(backbone_cfg["patch_size"]),
                max_num_patches=backbone_cfg.get("max_num_patches"),
            ),
        )

    regression_loss = GeneralRegressionLoss(loss_name=loss_cfg.get("regression", "smooth_l1"))
    trainable_parameters = [parameter for parameter in model_core.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(optimization_cfg["lr"]), weight_decay=float(optimization_cfg["weight_decay"]))
    scheduler = EpochLRScheduler(optimizer, total_epochs=int(optimization_cfg["epochs"]), config=optimization_cfg.get("scheduler"))
    early_stopping = EarlyStopping.from_config(optimization_cfg.get("early_stopping"))
    eval_cfg = config.get("evaluation", {})
    checkpoint_metric = str(eval_cfg.get("checkpoint_metric", "srcc")).lower()
    use_fixed_query_proxy = bool(
        eval_cfg.get("use_fixed_query_proxy", False)
        or checkpoint_metric.startswith("fixed_query")
        or checkpoint_metric == "blend_macro_srcc"
    )
    fixed_query_split = str(eval_cfg.get("fixed_query_proxy_split", "val"))
    fixed_query_every = max(int(eval_cfg.get("fixed_query_eval_every_epochs", 1)), 1)

    if distributed:
        model = DDP(
            model_core,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=bool(config.get("distributed", {}).get("find_unused_parameters", True)),
            broadcast_buffers=False,
        )
        if is_main:
            find_unused = bool(config.get("distributed", {}).get("find_unused_parameters", False))
            print(f"[COBRA] Stage1 DDP enabled: world_size={world_size}, find_unused_parameters={find_unused}")

    output_dir = ensure_dir(config["experiment"]["output_dir"])
    best_metric = float("-inf")
    best_checkpoint = output_dir / "best_stage1.pt"
    global_step = 0

    for epoch in range(1, int(optimization_cfg["epochs"]) + 1):
        current_lr = scheduler.step(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        progress = tqdm(train_loader, desc=f"Stage1 Epoch {epoch}", leave=False, disable=not is_main)
        train_bad_images_before = _skipped_bad_image_count(train_dataset)
        epoch_loss = 0.0
        epoch_reg = 0.0
        epoch_diversity = 0.0
        epoch_attribute_reg = 0.0
        epoch_ranking = 0.0
        num_batches = 0
        for batch in progress:
            images = {key: value.to(device, non_blocking=True) for key, value in batch["image_batch"].items()}
            targets = batch["score"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(images)
                reg_loss = regression_loss(outputs["score"], targets)
                attribute_reg_loss = regression_loss(outputs["attribute_score"], targets)
                ranking_loss = query_pairwise_ranking_loss(
                    outputs["score"],
                    targets,
                    margin=float(loss_cfg.get("ranking_margin", 0.0)),
                )
                diversity_loss = attribute_diversity_loss(outputs["attribute_tokens"])
                total_loss = (
                    reg_loss
                    + float(loss_cfg.get("lambda_attribute_regression", 0.0)) * attribute_reg_loss
                    + float(loss_cfg.get("lambda_ranking", 0.0)) * ranking_loss
                    + float(loss_cfg.get("lambda_attribute_diversity", 0.0)) * diversity_loss
                )

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            global_step += 1
            epoch_loss += float(total_loss.item())
            epoch_reg += float(reg_loss.item())
            epoch_attribute_reg += float(attribute_reg_loss.item())
            epoch_ranking += float(ranking_loss.item())
            epoch_diversity += float(diversity_loss.item())
            num_batches += 1

            if tracker is not None and global_step % max(log_every_steps, 1) == 0:
                tracker.log(
                    {
                        "train/global_step": global_step,
                        "train/loss": float(total_loss.item()),
                        "train/regression_loss": float(reg_loss.item()),
                        "train/attribute_regression_loss": float(attribute_reg_loss.item()),
                        "train/ranking_loss": float(ranking_loss.item()),
                        "train/attribute_diversity_loss": float(diversity_loss.item()),
                        "optimization/lr": current_lr,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            progress.set_postfix(loss=f"{total_loss.item():.4f}", reg=f"{reg_loss.item():.4f}", rank=f"{ranking_loss.item():.4f}")

        train_bad_images_after = _skipped_bad_image_count(train_dataset)
        train_bad_images_epoch = train_bad_images_after - train_bad_images_before
        stop_now = False
        if is_main:
            val_bad_images_before = _skipped_bad_image_count(val_dataset)
            metrics = evaluate_stage1(model_core, val_loader, device)
            aux_metrics = evaluate_stage1(model_core, aux_val_loader, device) if aux_val_loader is not None else {}
            fixed_query_metrics: dict[str, float] = {}
            if use_fixed_query_proxy and (epoch % fixed_query_every == 0 or epoch == 1):
                fixed_query_metrics = evaluate_stage1_fixed_query_base(
                    model_core,
                    data_config,
                    backbone_cfg,
                    device,
                    eval_cfg,
                    split=fixed_query_split,
                )
            val_bad_images_after = _skipped_bad_image_count(val_dataset)
            val_bad_images_epoch = val_bad_images_after - val_bad_images_before
            train_metrics = {
                "train/loss": epoch_loss / max(num_batches, 1),
                "train/regression_loss": epoch_reg / max(num_batches, 1),
                "train/attribute_regression_loss": epoch_attribute_reg / max(num_batches, 1),
                "train/ranking_loss": epoch_ranking / max(num_batches, 1),
                "train/attribute_diversity_loss": epoch_diversity / max(num_batches, 1),
                "train/use_amp": float(use_amp),
                "optimization/lr": current_lr,
                "data/train_bad_images_skipped_epoch": float(train_bad_images_epoch),
                "data/train_bad_images_skipped_total": float(train_bad_images_after),
                "data/val_bad_images_skipped_epoch": float(val_bad_images_epoch),
                "data/val_bad_images_skipped_total": float(val_bad_images_after),
            }
            eval_metrics = {f"val/{key}": value for key, value in metrics.items()}
            if aux_metrics:
                eval_metrics.update({f"val_aux_{aux_val_mode}/{key}": value for key, value in aux_metrics.items()})
            if fixed_query_metrics:
                eval_metrics.update({f"val/{key}": value for key, value in fixed_query_metrics.items()})
            selection_score, selection_key = stage1_checkpoint_selection_score(
                metrics,
                fixed_query_metrics,
                eval_cfg,
                checkpoint_metric,
            )
            tqdm.write(
                f"[COBRA] Stage1 Epoch {epoch}: "
                f"train_bad_images_skipped={train_bad_images_epoch} "
                f"(total={train_bad_images_after}), "
                f"val_bad_images_skipped={val_bad_images_epoch} "
                f"(total={val_bad_images_after}), "
                f"val_macro_srcc={metrics.get('macro_srcc', float('nan')):.4f}, "
                f"fixed_query_macro_srcc={fixed_query_metrics.get('fixed_query_macro_srcc', float('nan')):.4f}, "
                f"selection({selection_key})={selection_score:.4f}"
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
            metric = selection_score
            if metric != metric:
                metric = float("-inf")
            checkpoint_metrics = {**metrics, **fixed_query_metrics, "selection_score": metric, "selection_key": selection_key}
            if metric > best_metric:
                best_metric = metric
                save_checkpoint(
                    best_checkpoint,
                    model_core,
                    optimizer=optimizer,
                    scheduler=scheduler.state_dict(),
                    epoch=epoch,
                    metrics=checkpoint_metrics,
                )
                if tracker is not None:
                    tracker.log_summary(
                        {
                            "best/stage1_srcc": metrics["srcc"],
                            "best/stage1_plcc": metrics["plcc"],
                            "best/stage1_macro_srcc": metrics.get("macro_srcc"),
                            "best/stage1_macro_plcc": metrics.get("macro_plcc"),
                            "best/stage1_fixed_query_macro_srcc": fixed_query_metrics.get("fixed_query_macro_srcc"),
                            "best/stage1_checkpoint_metric": checkpoint_metric,
                            "best/stage1_selection_key": selection_key,
                            "best/stage1_selection_value": metric,
                            **({f"best/stage1_aux_{aux_val_mode}_srcc": aux_metrics.get("srcc"),
                                f"best/stage1_aux_{aux_val_mode}_macro_srcc": aux_metrics.get("macro_srcc")} if aux_metrics else {}),
                            "best/stage1_epoch": epoch,
                            "artifacts/best_stage1_checkpoint": best_checkpoint.as_posix(),
                            "data/train_bad_images_skipped_total": float(train_bad_images_after),
                            "data/val_bad_images_skipped_total": float(val_bad_images_after),
                        }
                    )
            early_stopping.update(metric)
            stop_now = bool(early_stopping.should_stop)
            if stop_now:
                tqdm.write(
                    f"[COBRA] Stage1 early stopping triggered at epoch {epoch} "
                    f"(patience={early_stopping.patience}, best_srcc={early_stopping.best_metric:.4f})."
                )
                if tracker is not None:
                    tracker.log_summary(
                        {
                            "optimization/stage1_early_stopped": 1.0,
                            "optimization/stage1_stop_epoch": epoch,
                        }
                    )
        if distributed:
            stop_tensor = torch.tensor([1 if stop_now else 0], dtype=torch.int64, device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_now = bool(stop_tensor.item())
        if stop_now:
            break

    return best_checkpoint
