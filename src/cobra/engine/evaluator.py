from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from cobra.engine.stage2_tta import adapt_model_parameters_on_support, adapt_user_on_support, tta_enabled
from cobra.losses.general_regression import GeneralRegressionLoss
from cobra.utils.metrics import macro_user_correlations, same_image_rank_correlation, threshold_accuracy


_QUERY_OUTPUT_KEYS = {
    "score",
    "generic_score",
    "stage1_generic_score",
    "residual_score",
    "bridge_residual_score",
    "bridge_basis_residual_score",
    "retrieved_residual_score",
    "kernel_residual_score",
    "user_state_residual_score",
    "basis_residual_score",
    "local_residual_score",
    "residual_offset_score",
    "query_prior_score",
}


def _slice_image_batch(images: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    return {key: value[start:end] for key, value in images.items()}


def _forward_stage2_query_chunks(
    model: torch.nn.Module,
    *,
    support_images: dict[str, torch.Tensor],
    support_scores: torch.Tensor,
    query_images: dict[str, torch.Tensor],
    support_prior_scores: torch.Tensor | None,
    query_prior_scores: torch.Tensor | None,
    user_adaptation: dict[str, torch.Tensor] | None,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    """Run independent query samples in smaller FP32 batches and restore order."""
    query_count = next(iter(query_images.values())).shape[0]
    if chunk_size <= 0 or query_count <= chunk_size:
        return model(
            support_images=support_images,
            support_scores=support_scores,
            query_images=query_images,
            support_prior_scores=support_prior_scores,
            query_prior_scores=query_prior_scores,
            user_adaptation=user_adaptation,
            compute_support_loo=False,
        )

    first_outputs: dict[str, torch.Tensor] | None = None
    query_outputs: dict[str, list[torch.Tensor]] = {}
    for start in range(0, query_count, chunk_size):
        end = min(start + chunk_size, query_count)
        chunk_outputs = model(
            support_images=support_images,
            support_scores=support_scores,
            query_images=_slice_image_batch(query_images, start, end),
            support_prior_scores=support_prior_scores,
            query_prior_scores=(query_prior_scores[start:end] if query_prior_scores is not None else None),
            user_adaptation=user_adaptation,
            compute_support_loo=False,
        )
        if first_outputs is None:
            first_outputs = dict(chunk_outputs)
        for key in _QUERY_OUTPUT_KEYS:
            value = chunk_outputs.get(key)
            if isinstance(value, torch.Tensor):
                query_outputs.setdefault(key, []).append(value)

    if first_outputs is None:
        raise RuntimeError("Stage2 query chunking received an empty query batch.")
    for key, values in query_outputs.items():
        first_outputs[key] = torch.cat(values, dim=0)
    return first_outputs


@contextmanager
def temporary_model_runtime_overrides(
    model: torch.nn.Module,
    config: dict[str, Any] | None,
):
    """Temporarily apply eval-only scalar/runtime knobs to a Stage2 model."""
    if config is None:
        yield
        return
    eval_cfg = config.get("evaluation", {})
    overrides = (
        eval_cfg.get("runtime_model_overrides")
        or eval_cfg.get("model_runtime_overrides")
        or {}
    )
    if not overrides:
        yield
        return
    target = model.module if hasattr(model, "module") else model
    previous: dict[str, Any] = {}
    for name, value in overrides.items():
        if hasattr(target, name):
            previous[name] = getattr(target, name)
            setattr(target, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(target, name, value)


def _disable_eval_progress() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_rank() != 0


@torch.no_grad()
def evaluate_stage1(model: torch.nn.Module, data_loader, device: torch.device) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in tqdm(data_loader, desc="Stage1 Eval", leave=False, disable=_disable_eval_progress()):
        images = {key: value.to(device, non_blocking=True) for key, value in batch["image_batch"].items()}
        scores = batch["score"].to(device, non_blocking=True)
        outputs = model(images)
        predictions = outputs["score"].detach().cpu().view(-1).tolist()
        targets = scores.detach().cpu().view(-1).tolist()
        user_ids = batch.get("user_id", ["global"] * len(targets))
        image_ids = batch.get("image_id", list(range(len(targets))))
        for user_id, image_id, truth, pred in zip(user_ids, image_ids, targets, predictions):
            rows.append(
                {
                    "user_id": str(user_id),
                    "image_id": str(image_id),
                    "score": float(truth),
                    "prediction": float(pred),
                }
            )
    frame = pd.DataFrame(rows)
    global_frame = frame.copy()
    global_frame["user_id"] = "global"
    global_metrics = macro_user_correlations(global_frame)
    macro_metrics = macro_user_correlations(frame)
    return {
        "srcc": global_metrics["macro_srcc"],
        "plcc": global_metrics["macro_plcc"],
        "macro_srcc": macro_metrics["macro_srcc"],
        "macro_plcc": macro_metrics["macro_plcc"],
    }


@torch.no_grad()
def evaluate_stage2(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    acc_threshold: float | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in tqdm(data_loader, desc="Stage2 Eval", leave=False, disable=_disable_eval_progress()):
        tta_metrics = {"tta_loss": 0.0, "tta_steps": 0.0}
        support_prior_scores = batch.get("support_prior_scores")
        query_prior_scores = batch.get("query_prior_scores")
        outputs = model(
            support_images={key: value.to(device, non_blocking=True) for key, value in batch["support_batch"].items()},
            support_scores=batch["support_scores"].to(device, non_blocking=True),
            query_images={key: value.to(device, non_blocking=True) for key, value in batch["query_batch"].items()},
            support_prior_scores=support_prior_scores.to(device, non_blocking=True) if support_prior_scores is not None else None,
            query_prior_scores=query_prior_scores.to(device, non_blocking=True) if query_prior_scores is not None else None,
        )
        predictions = outputs["score"].detach().cpu().tolist()
        for image_id, truth, pred in zip(batch["query_image_ids"], batch["query_scores"].tolist(), predictions):
            rows.append(
                {
                    "user_id": batch["user_id"],
                    "image_id": image_id,
                    "score": float(truth),
                    "prediction": float(pred),
                    "support_size": int(batch["support_size"]),
                    "tta_loss": float(tta_metrics.get("tta_loss", 0.0)),
                    "tta_steps": float(tta_metrics.get("tta_steps", 0.0)),
                }
            )
    frame = pd.DataFrame(rows)
    metrics = macro_user_correlations(frame)
    metrics["same_image_srcc"] = same_image_rank_correlation(frame)
    if acc_threshold is not None:
        metrics["acc"] = threshold_accuracy(frame, threshold=acc_threshold)
    return metrics, frame


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


def evaluate_stage2_decomposed(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    acc_threshold: float | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    use_tta = config is not None and tta_enabled(config)
    regression_loss = GeneralRegressionLoss(loss_name=config.get("loss", {}).get("regression", "smooth_l1")) if use_tta else None
    amp_cfg = config.get("amp", {}) if config is not None else {}
    use_amp = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(amp_cfg.get("dtype", "bfloat16")).lower() == "bfloat16" else torch.float16
    query_forward_chunk_size = int((config or {}).get("evaluation", {}).get("query_forward_chunk_size", 0))
    profile_eval_phases = bool((config or {}).get("diagnostics", {}).get("profile_eval_phases", False))

    def profile_now() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    with temporary_model_runtime_overrides(model, config):
        for batch in tqdm(data_loader, desc="Stage2 Eval", leave=False, disable=_disable_eval_progress()):
            phase_started = profile_now() if profile_eval_phases else 0.0
            user_adaptation = None
            tta_metrics = {"tta_loss": 0.0, "tta_steps": 0.0}
            if use_tta and regression_loss is not None:
                user_adaptation, tta_metrics = adapt_user_on_support(
                    model=model,
                    episode=batch,
                    config=config,
                    device=device,
                    regression_loss=regression_loss,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                user_tta_finished = profile_now() if profile_eval_phases else 0.0
                restore_model_tta, param_tta_metrics = adapt_model_parameters_on_support(
                    model=model,
                    episode=batch,
                    config=config,
                    device=device,
                    regression_loss=regression_loss,
                    user_adaptation=user_adaptation,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                parameter_tta_finished = profile_now() if profile_eval_phases else 0.0
                tta_metrics.update(param_tta_metrics)
            else:
                restore_model_tta = lambda: None
                user_tta_finished = phase_started
                parameter_tta_finished = phase_started
            support_prior_scores = batch.get("support_prior_scores")
            query_prior_scores = batch.get("query_prior_scores")
            try:
                with torch.no_grad():
                    outputs = _forward_stage2_query_chunks(
                        model,
                        support_images={key: value.to(device, non_blocking=True) for key, value in batch["support_batch"].items()},
                        support_scores=batch["support_scores"].to(device, non_blocking=True),
                        query_images={key: value.to(device, non_blocking=True) for key, value in batch["query_batch"].items()},
                        support_prior_scores=support_prior_scores.to(device, non_blocking=True) if support_prior_scores is not None else None,
                        query_prior_scores=query_prior_scores.to(device, non_blocking=True) if query_prior_scores is not None else None,
                        user_adaptation=user_adaptation,
                        chunk_size=query_forward_chunk_size,
                    )
                query_finished = profile_now() if profile_eval_phases else 0.0
            finally:
                restore_model_tta()
            phase_finished = profile_now() if profile_eval_phases else 0.0
            if profile_eval_phases:
                memory = ""
                if device.type == "cuda":
                    memory = (
                        f" allocated_gb={torch.cuda.memory_allocated(device) / (1024 ** 3):.3f}"
                        f" reserved_gb={torch.cuda.memory_reserved(device) / (1024 ** 3):.3f}"
                    )
                print(
                    "[COBRA][PROFILE] Stage2 eval phases: "
                    f"user_id={batch['user_id']} "
                    f"user_tta={user_tta_finished - phase_started:.3f}s "
                    f"parameter_tta={parameter_tta_finished - user_tta_finished:.3f}s "
                    f"query={query_finished - parameter_tta_finished:.3f}s "
                    f"restore={phase_finished - query_finished:.3f}s "
                    f"total={phase_finished - phase_started:.3f}s{memory}",
                    flush=True,
                )
            predictions = outputs["score"].detach().cpu().tolist()
            base_predictions = outputs["generic_score"].detach().cpu().tolist()
            stage1_predictions = outputs["stage1_generic_score"].detach().cpu().tolist()
            residual_predictions = outputs.get("residual_score", outputs["score"] - outputs["generic_score"]).detach().cpu().tolist()
            bridge_residual_predictions = outputs.get("bridge_residual_score", outputs["score"].new_zeros(outputs["score"].shape)).detach().cpu().tolist()
            bridge_basis_residual_predictions = outputs.get("bridge_basis_residual_score", outputs["score"].new_zeros(outputs["score"].shape)).detach().cpu().tolist()
            retrieved_residual_predictions = outputs.get("retrieved_residual_score", outputs.get("local_residual_score", outputs["score"].new_zeros(outputs["score"].shape))).detach().cpu().tolist()
            kernel_residual_predictions = outputs.get("kernel_residual_score", outputs["score"].new_zeros(outputs["score"].shape)).detach().cpu().tolist()
            user_state_residual_predictions = outputs.get("user_state_residual_score", outputs["score"].new_zeros(outputs["score"].shape)).detach().cpu().tolist()
            basis_residual_predictions = outputs.get("basis_residual_score", outputs["score"].new_zeros(outputs["score"].shape)).detach().cpu().tolist()
            local_residual_predictions = outputs.get("local_residual_score", outputs["score"].new_zeros(outputs["score"].shape)).detach().cpu().tolist()
            residual_offset_predictions = outputs.get("residual_offset_score", outputs["score"].new_zeros(outputs["score"].shape)).detach().cpu().tolist()
            prior_tensor = outputs.get("query_prior_score")
            prior_predictions = (
                prior_tensor.detach().cpu().tolist()
                if prior_tensor is not None
                else [float("nan")] * len(predictions)
            )
            for (
                image_id,
                truth,
                pred,
                base_pred,
                stage1_pred,
                prior_pred,
                residual_pred,
                bridge_residual_pred,
                bridge_basis_residual_pred,
                retrieved_residual_pred,
                kernel_residual_pred,
                user_state_residual_pred,
                basis_residual_pred,
                local_residual_pred,
                residual_offset_pred,
            ) in zip(
                batch["query_image_ids"],
                batch["query_scores"].tolist(),
                predictions,
                base_predictions,
                stage1_predictions,
                prior_predictions,
                residual_predictions,
                bridge_residual_predictions,
                bridge_basis_residual_predictions,
                retrieved_residual_predictions,
                kernel_residual_predictions,
                user_state_residual_predictions,
                basis_residual_predictions,
                local_residual_predictions,
                residual_offset_predictions,
            ):
                support_estimator_residual_pred = (
                    float(residual_offset_pred)
                    + float(bridge_residual_pred)
                    + float(bridge_basis_residual_pred)
                    + float(retrieved_residual_pred)
                    + float(kernel_residual_pred)
                    + float(user_state_residual_pred)
                )
                rows.append(
                    {
                        "user_id": batch["user_id"],
                        "image_id": image_id,
                        "score": float(truth),
                        "prediction": float(pred),
                        "base_prediction": float(base_pred),
                        "stage1_prediction": float(stage1_pred),
                        "prior_prediction": float(prior_pred),
                        "target_residual": float(truth) - float(base_pred),
                        "residual_score": float(residual_pred),
                        "bridge_residual_score": float(bridge_residual_pred) + float(residual_offset_pred),
                        "bridge_basis_residual_score": float(bridge_basis_residual_pred) + float(residual_offset_pred),
                        "retrieved_residual_score": float(retrieved_residual_pred) + float(residual_offset_pred),
                        "kernel_residual_score": float(kernel_residual_pred) + float(residual_offset_pred),
                        "user_state_residual_score": float(user_state_residual_pred) + float(residual_offset_pred),
                        "support_estimator_residual_score": support_estimator_residual_pred,
                        "basis_residual_score": float(basis_residual_pred) + float(residual_offset_pred),
                        "local_residual_score": float(local_residual_pred) + float(residual_offset_pred),
                        "residual_offset_score": float(residual_offset_pred),
                        "support_size": int(batch["support_size"]),
                        "tta_loss": float(tta_metrics.get("tta_loss", 0.0)),
                        "tta_steps": float(tta_metrics.get("tta_steps", 0.0)),
                        "tta_support_images": float(tta_metrics.get("tta_support_images", 0.0)),
                        "tta_support_chunks": float(tta_metrics.get("tta_support_chunks", 0.0)),
                        "tta_validation_images": float(tta_metrics.get("tta_validation_images", 0.0)),
                        "tta_support_leave_one_out": float(
                            tta_metrics.get("tta_support_leave_one_out", 0.0)
                        ),
                        "tta_rank_gate_applied": float(tta_metrics.get("tta_rank_gate_applied", 0.0)),
                        "tta_lambda_support_regression": float(
                            tta_metrics.get("tta_lambda_support_regression", 0.0)
                        ),
                        "tta_param_loss": float(tta_metrics.get("tta_param_loss", 0.0)),
                        "tta_param_steps": float(tta_metrics.get("tta_param_steps", 0.0)),
                        "tta_param_count": float(tta_metrics.get("tta_param_count", 0.0)),
                        "tta_param_tensors": float(tta_metrics.get("tta_param_tensors", 0.0)),
                        "tta_param_support_images": float(tta_metrics.get("tta_param_support_images", 0.0)),
                        "tta_param_support_chunks": float(tta_metrics.get("tta_param_support_chunks", 0.0)),
                        "tta_param_validation_images": float(
                            tta_metrics.get("tta_param_validation_images", 0.0)
                        ),
                        "tta_param_validation_gate_applied": float(
                            tta_metrics.get("tta_param_validation_gate_applied", 0.0)
                        ),
                        "tta_param_validation_base_rank": float(
                            tta_metrics.get("tta_param_validation_base_rank", 0.0)
                        ),
                        "tta_param_validation_adapted_rank": float(
                            tta_metrics.get("tta_param_validation_adapted_rank", 0.0)
                        ),
                        "tta_param_support_leave_one_out": float(
                            tta_metrics.get("tta_param_support_leave_one_out", 0.0)
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
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
    return metrics, frame
