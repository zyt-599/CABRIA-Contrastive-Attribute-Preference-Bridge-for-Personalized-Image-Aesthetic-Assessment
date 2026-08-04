from __future__ import annotations

from typing import Any, Callable

import torch

from cabria.losses.general_regression import GeneralRegressionLoss
from cabria.losses.query_ranking import pairwise_ranking_loss
from cabria.models.cabria_model import CABRIAStage2Model
from cabria.utils.metrics import srcc


def _move_image_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _slice_image_batch(batch: dict[str, torch.Tensor], stop: int) -> dict[str, torch.Tensor]:
    return {key: value[:stop] for key, value in batch.items()}


def _index_image_batch(batch: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value.index_select(0, indices) for key, value in batch.items()}


def _score_stratified_indices(scores: torch.Tensor, max_images: int) -> torch.Tensor | None:
    if max_images <= 0 or scores.shape[0] <= max_images:
        return None
    order = torch.argsort(scores.detach())
    positions = torch.linspace(0, scores.shape[0] - 1, steps=max_images, device=scores.device).round().long()
    return order.index_select(0, positions)


def _chunk_indices(indices: torch.Tensor, chunk_size: int) -> list[torch.Tensor]:
    chunk_size = max(int(chunk_size), 1)
    return [indices[start : start + chunk_size] for start in range(0, indices.numel(), chunk_size)]


def _support_without_chunk(
    support_images: dict[str, torch.Tensor],
    support_scores: torch.Tensor,
    support_prior_scores: torch.Tensor | None,
    chunk: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    keep_mask = torch.ones(support_scores.numel(), dtype=torch.bool, device=support_scores.device)
    keep_mask.index_fill_(0, chunk, False)
    fit_indices = torch.nonzero(keep_mask, as_tuple=False).view(-1)
    fit_images = _index_image_batch(support_images, fit_indices)
    fit_scores = support_scores.index_select(0, fit_indices)
    fit_prior_scores = (
        support_prior_scores.index_select(0, fit_indices)
        if support_prior_scores is not None
        else None
    )
    return fit_images, fit_scores, fit_prior_scores


def _apply_support_srcc_tta_init(
    model: CABRIAStage2Model,
    user_state: dict[str, torch.Tensor],
    support_images: dict[str, torch.Tensor] | torch.Tensor,
    support_scores: torch.Tensor,
    support_prior_scores: torch.Tensor | None,
    *,
    config: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Task-vector style init: map support Spearman (baseline vs labels) to residual_log_scale in [min, max]."""
    tta_cfg = config.get("test_time_adaptation", {})
    meta = {"tta_support_srcc_baseline": float("nan"), "tta_support_srcc_init_applied": 0.0}
    if not bool(tta_cfg.get("support_srcc_init", False)):
        return meta, user_state
    core = model.module if hasattr(model, "module") else model
    scale_key = "residual_log_scale"
    if scale_key not in user_state or int(support_scores.numel()) < 3:
        return meta, user_state

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(
                support_images=support_images,
                support_scores=support_scores,
                query_images=support_images,
                support_prior_scores=support_prior_scores,
                query_prior_scores=support_prior_scores,
                user_adaptation=None,
                compute_support_loo=False,
            )
        preds = out["score"].detach().float().view(-1).cpu().tolist()
        targets = support_scores.detach().float().view(-1).cpu().tolist()
        rho = srcc(targets, preds)
        meta["tta_support_srcc_baseline"] = float(rho)
        if rho != rho:
            return meta, user_state
        t = max(0.0, min(1.0, (float(rho) + 1.0) * 0.5))
        lo = float(core.tta_residual_log_scale_min)
        hi = float(core.tta_residual_log_scale_max)
        if hi <= lo + 1e-9:
            return meta, user_state
        value = lo + t * (hi - lo)
        ref = user_state[scale_key]
        new_tensor = torch.full_like(ref, float(value), device=ref.device, dtype=ref.dtype)
        new_tensor.requires_grad_(True)
        user_state[scale_key] = new_tensor
        meta["tta_support_srcc_init_applied"] = 1.0
        return meta, user_state
    finally:
        if was_training:
            model.train()


def tta_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("test_time_adaptation", {}).get("enabled", False))


def _mode_value(tta_cfg: dict[str, Any], key: str, mode: str, default: Any) -> Any:
    mode_key = f"{mode}_{key}"
    if mode_key in tta_cfg:
        return tta_cfg[mode_key]
    return tta_cfg.get(key, default)


def _mode_list(tta_cfg: dict[str, Any], key: str, mode: str, default: list[str] | None = None) -> list[str]:
    value = _mode_value(tta_cfg, key, mode, default or [])
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _select_named_parameters(
    model: torch.nn.Module,
    patterns: list[str],
) -> list[tuple[str, torch.nn.Parameter]]:
    if not patterns:
        return []
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            selected.append((name, parameter))
    return selected


def adapt_model_parameters_on_support(
    model: CABRIAStage2Model,
    episode: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    regression_loss: GeneralRegressionLoss,
    user_adaptation: dict[str, torch.Tensor] | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    mode: str = "eval",
) -> tuple[Callable[[], None], dict[str, float]]:
    """Temporarily fine-tune selected Stage2 parameters on support LOO loss.

    This applies per-user support adaptation while confining updates to
    configured Stage2 bridge/residual modules and restores the checkpoint after
    the episode query forward.
    """
    tta_cfg = config.get("test_time_adaptation", {})
    if not bool(_mode_value(tta_cfg, "model_parameter_adaptation", mode, False)):
        return (lambda: None), {"tta_param_steps": 0.0, "tta_param_count": 0.0}

    patterns = _mode_list(
        tta_cfg,
        "model_parameter_patterns",
        mode,
        ["bridge.", "residual_head."],
    )
    selected = _select_named_parameters(model, patterns)
    if not selected:
        return (lambda: None), {"tta_param_steps": 0.0, "tta_param_count": 0.0}

    loss_cfg = config.get("loss", {})
    query_ranking_mode = str(loss_cfg.get("query_ranking_mode", "softplus_margin"))
    support_ranking_mode = str(loss_cfg.get("support_ranking_mode", query_ranking_mode))
    tta_support_ranking_mode = str(_mode_value(tta_cfg, "model_parameter_support_ranking_mode", mode, support_ranking_mode))
    support_ranking_max_pairs = int(loss_cfg.get("support_ranking_max_pairs", 0))
    tta_support_ranking_max_pairs = int(
        _mode_value(tta_cfg, "model_parameter_support_ranking_max_pairs", mode, support_ranking_max_pairs)
    )
    steps = int(_mode_value(tta_cfg, "model_parameter_steps", mode, 1))
    lr = float(_mode_value(tta_cfg, "model_parameter_lr", mode, 1e-5))
    weight_decay = float(_mode_value(tta_cfg, "model_parameter_weight_decay", mode, 0.0))
    lambda_regression = float(_mode_value(tta_cfg, "model_parameter_lambda_support_regression", mode, 0.2))
    lambda_rank = float(_mode_value(tta_cfg, "model_parameter_lambda_support_ranking", mode, 1.0))
    margin = float(
        _mode_value(
            tta_cfg,
            "model_parameter_support_ranking_margin",
            mode,
            config.get("loss", {}).get("query_ranking_margin", 0.5),
        )
    )
    chunk_size = int(_mode_value(tta_cfg, "model_parameter_support_chunk_size", mode, 16))
    max_support_images = int(_mode_value(tta_cfg, "model_parameter_max_support_images", mode, 0))
    support_leave_one_out = bool(_mode_value(tta_cfg, "model_parameter_support_leave_one_out", mode, False))

    full_support_scores = episode["support_scores"].to(device, non_blocking=True)
    full_support_prior_scores = episode.get("support_prior_scores")
    full_support_prior_scores = full_support_prior_scores.to(device, non_blocking=True) if full_support_prior_scores is not None else None
    full_support_images = _move_image_batch(episode["support_batch"], device)
    selected_indices = _score_stratified_indices(full_support_scores, max_images=max_support_images)
    if selected_indices is None:
        support_scores = full_support_scores
        support_prior_scores = full_support_prior_scores
        support_images = full_support_images
    else:
        support_scores = full_support_scores.index_select(0, selected_indices)
        support_prior_scores = (
            full_support_prior_scores.index_select(0, selected_indices)
            if full_support_prior_scores is not None
            else None
        )
        support_images = _index_image_batch(full_support_images, selected_indices)

    validation_scores = None
    validation_images = None
    validation_prior_scores = None
    validation_holdout_count = 0
    validation_images_count = int(_mode_value(tta_cfg, "model_parameter_validation_images", mode, 0))
    validation_holdout = bool(_mode_value(tta_cfg, "model_parameter_validation_holdout", mode, True))
    if validation_images_count > 0 and support_scores.numel() > validation_images_count + 1:
        val_count = min(validation_images_count, support_scores.numel() - 2)
        order = torch.argsort(support_scores.detach())
        positions = torch.linspace(0, support_scores.numel() - 1, steps=val_count, device=device).round().long().unique()
        validation_local = order.index_select(0, positions[:val_count])
        if validation_local.numel() >= 2:
            validation_scores = support_scores.index_select(0, validation_local)
            validation_images = _index_image_batch(support_images, validation_local)
            validation_prior_scores = (
                support_prior_scores.index_select(0, validation_local)
                if support_prior_scores is not None
                else None
            )
            if validation_holdout:
                keep_mask = torch.ones(support_scores.numel(), dtype=torch.bool, device=device)
                keep_mask.index_fill_(0, validation_local, False)
                adaptation_local = torch.nonzero(keep_mask, as_tuple=False).view(-1)
                if adaptation_local.numel() >= 2:
                    support_scores = support_scores.index_select(0, adaptation_local)
                    support_prior_scores = (
                        support_prior_scores.index_select(0, adaptation_local)
                        if support_prior_scores is not None
                        else None
                    )
                    support_images = _index_image_batch(support_images, adaptation_local)
                    validation_holdout_count = int(validation_local.numel())
                else:
                    validation_scores = None
                    validation_images = None
                    validation_prior_scores = None

    support_loo_indices = torch.arange(support_scores.numel(), device=device)
    chunks = _chunk_indices(support_loo_indices, chunk_size=chunk_size)

    all_parameters = tuple(model.parameters())
    original_requires_grad = tuple(parameter.requires_grad for parameter in all_parameters)
    originals = [(parameter, parameter.detach().clone()) for _, parameter in selected]

    selected_ids = {id(parameter) for _, parameter in selected}
    for parameter in all_parameters:
        parameter.requires_grad_(id(parameter) in selected_ids)
    optimizer = torch.optim.AdamW((parameter for _, parameter in selected), lr=lr, weight_decay=weight_decay)
    last_loss = support_scores.new_tensor(0.0)

    def _copy_selected_values(values: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
        with torch.no_grad():
            for parameter, value in values:
                parameter.copy_(value)

    def restore() -> None:
        _copy_selected_values(originals)
        for parameter, requires_grad in zip(all_parameters, original_requires_grad):
            parameter.requires_grad_(requires_grad)

    try:
        for _ in range(max(steps, 0)):
            optimizer.zero_grad(set_to_none=True)
            step_loss = support_scores.new_tensor(0.0)
            for chunk in chunks:
                chunk_scores = support_scores.index_select(0, chunk)
                query_images_for_forward = _index_image_batch(support_images, chunk)
                query_prior_for_forward = (
                    support_prior_scores.index_select(0, chunk)
                    if support_prior_scores is not None
                    else None
                )
                fit_support_images = support_images
                fit_support_scores = support_scores
                fit_support_prior_scores = support_prior_scores
                if support_leave_one_out and support_scores.numel() > chunk.numel() + 1:
                    fit_support_images, fit_support_scores, fit_support_prior_scores = _support_without_chunk(
                        support_images,
                        support_scores,
                        support_prior_scores,
                        chunk,
                    )
                with torch.enable_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(
                        support_images=fit_support_images,
                        support_scores=fit_support_scores,
                        query_images=query_images_for_forward,
                        support_prior_scores=fit_support_prior_scores,
                        query_prior_scores=query_prior_for_forward,
                        user_adaptation=user_adaptation,
                        compute_support_loo=False,
                    )
                    chunk_loss = support_scores.new_tensor(0.0)
                    if lambda_regression > 0.0:
                        chunk_loss = chunk_loss + lambda_regression * regression_loss(
                            outputs["score"],
                            chunk_scores,
                        )
                    if lambda_rank > 0.0:
                        chunk_loss = chunk_loss + lambda_rank * pairwise_ranking_loss(
                            outputs["score"],
                            chunk_scores,
                            mode=tta_support_ranking_mode,
                            margin=margin,
                            max_pairs=tta_support_ranking_max_pairs,
                        )
                    chunk_loss = chunk_loss / max(len(chunks), 1)
                if chunk_loss.requires_grad:
                    chunk_loss.backward()
                step_loss = step_loss + chunk_loss.detach()
            optimizer.step()
            last_loss = step_loss.detach()
    except Exception:
        restore()
        raise

    validation_gate_applied = 0.0
    validation_base_rank = float("nan")
    validation_adapted_rank = float("nan")
    if validation_scores is not None and validation_images is not None:
        adapted_values = [(parameter, parameter.detach().clone()) for _, parameter in selected]
        min_improvement = float(_mode_value(tta_cfg, "model_parameter_validation_min_rank_improvement", mode, 0.0))
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            adapted_outputs = model(
                support_images=support_images,
                support_scores=support_scores,
                query_images=validation_images,
                support_prior_scores=support_prior_scores,
                query_prior_scores=validation_prior_scores,
                user_adaptation=user_adaptation,
                compute_support_loo=False,
            )
            adapted_rank = pairwise_ranking_loss(
                adapted_outputs["score"],
                validation_scores,
                mode=tta_support_ranking_mode,
                margin=margin,
                max_pairs=tta_support_ranking_max_pairs,
            )
        _copy_selected_values(originals)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            base_outputs = model(
                support_images=support_images,
                support_scores=support_scores,
                query_images=validation_images,
                support_prior_scores=support_prior_scores,
                query_prior_scores=validation_prior_scores,
                user_adaptation=user_adaptation,
                compute_support_loo=False,
            )
            base_rank = pairwise_ranking_loss(
                base_outputs["score"],
                validation_scores,
                mode=tta_support_ranking_mode,
                margin=margin,
                max_pairs=tta_support_ranking_max_pairs,
            )
        validation_base_rank = float(base_rank.detach().cpu().item())
        validation_adapted_rank = float(adapted_rank.detach().cpu().item())
        if validation_adapted_rank <= validation_base_rank - min_improvement:
            _copy_selected_values(adapted_values)
        else:
            validation_gate_applied = 1.0

    return restore, {
        "tta_param_loss": float(last_loss.detach().cpu().item()),
        "tta_param_steps": float(max(steps, 0)),
        "tta_param_count": float(sum(parameter.numel() for _, parameter in selected)),
        "tta_param_tensors": float(len(selected)),
        "tta_param_support_images": float(support_scores.numel()),
        "tta_param_support_chunks": float(len(chunks)),
        "tta_param_validation_images": float(validation_holdout_count),
        "tta_param_validation_gate_applied": float(validation_gate_applied),
        "tta_param_validation_base_rank": float(validation_base_rank),
        "tta_param_validation_adapted_rank": float(validation_adapted_rank),
        "tta_param_support_leave_one_out": 1.0 if support_leave_one_out else 0.0,
    }


def adapt_user_on_support(
    model: CABRIAStage2Model,
    episode: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    regression_loss: GeneralRegressionLoss,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    mode: str = "eval",
) -> tuple[dict[str, torch.Tensor] | None, dict[str, float]]:
    tta_cfg = config.get("test_time_adaptation", {})
    loss_cfg = config.get("loss", {})
    query_ranking_mode = str(loss_cfg.get("query_ranking_mode", "softplus_margin"))
    support_ranking_mode = str(loss_cfg.get("support_ranking_mode", query_ranking_mode))
    tta_support_ranking_mode = str(_mode_value(tta_cfg, "tta_support_ranking_mode", mode, support_ranking_mode))
    query_ranking_max_pairs = int(loss_cfg.get("query_ranking_max_pairs", 0))
    support_ranking_max_pairs = int(loss_cfg.get("support_ranking_max_pairs", 0))
    tta_support_ranking_max_pairs = int(
        _mode_value(tta_cfg, "tta_support_ranking_max_pairs", mode, support_ranking_max_pairs)
    )
    if not bool(tta_cfg.get("enabled", False)):
        return None, {"tta_loss": 0.0, "tta_steps": 0.0}
    user_state = model.make_user_adaptation(device=device, requires_grad=True)
    if user_state is None:
        return None, {"tta_loss": 0.0, "tta_steps": 0.0}
    freeze_keys = set(_mode_list(tta_cfg, "freeze_keys", mode))
    if bool(_mode_value(tta_cfg, "freeze_scalar_adaptation", mode, False)):
        freeze_keys.update({"bias", "stage1_delta_adjust"})
    for key in freeze_keys:
        user_state.pop(key, None)
    if not user_state:
        return None, {"tta_loss": 0.0, "tta_steps": 0.0}

    full_support_scores = episode["support_scores"].to(device, non_blocking=True)
    full_support_prior_scores = episode.get("support_prior_scores")
    full_support_prior_scores = full_support_prior_scores.to(device, non_blocking=True) if full_support_prior_scores is not None else None
    full_support_images = _move_image_batch(episode["support_batch"], device)
    steps = int(_mode_value(tta_cfg, "steps", mode, 3))
    lr = float(_mode_value(tta_cfg, "lr", mode, 0.05))
    lambda_regression = float(_mode_value(tta_cfg, "lambda_support_regression", mode, 1.0))
    lambda_rank = float(_mode_value(tta_cfg, "lambda_support_ranking", mode, 0.1))
    lambda_l2 = float(_mode_value(tta_cfg, "lambda_l2", mode, 0.001))
    margin = float(_mode_value(tta_cfg, "support_ranking_margin", mode, config.get("loss", {}).get("query_ranking_margin", 0.5)))
    chunk_size = int(_mode_value(tta_cfg, "support_chunk_size", mode, 16))
    max_support_images = int(_mode_value(tta_cfg, "max_support_images", mode, 0))
    support_leave_one_out = bool(_mode_value(tta_cfg, "support_leave_one_out", mode, False))
    selected_indices = _score_stratified_indices(full_support_scores, max_images=max_support_images)
    if selected_indices is None:
        selected_indices = torch.arange(full_support_scores.numel(), device=device)
        support_scores = full_support_scores
        support_prior_scores = full_support_prior_scores
        support_images = full_support_images
    else:
        support_scores = full_support_scores.index_select(0, selected_indices)
        support_prior_scores = (
            full_support_prior_scores.index_select(0, selected_indices)
            if full_support_prior_scores is not None
            else None
        )
        support_images = _index_image_batch(full_support_images, selected_indices)
    validation_scores = None
    validation_images = None
    validation_prior_scores = None
    validation_holdout_count = 0
    support_validation_images = int(_mode_value(tta_cfg, "support_validation_images", mode, 0))
    support_validation_holdout = bool(_mode_value(tta_cfg, "support_validation_holdout", mode, False))
    if support_validation_images > 0 and support_scores.numel() > support_validation_images + 1:
        val_count = min(support_validation_images, support_scores.numel() - 2)
        order = torch.argsort(support_scores.detach())
        positions = torch.linspace(0, support_scores.numel() - 1, steps=val_count, device=device).round().long().unique()
        validation_local = order.index_select(0, positions[:val_count])
        if validation_local.numel() >= 2:
            validation_scores = support_scores.index_select(0, validation_local)
            validation_images = _index_image_batch(support_images, validation_local)
            validation_prior_scores = (
                support_prior_scores.index_select(0, validation_local)
                if support_prior_scores is not None
                else None
            )
            if support_validation_holdout:
                keep_mask = torch.ones(support_scores.numel(), dtype=torch.bool, device=device)
                keep_mask.index_fill_(0, validation_local, False)
                adaptation_local = torch.nonzero(keep_mask, as_tuple=False).view(-1)
                if adaptation_local.numel() >= 2:
                    support_scores = support_scores.index_select(0, adaptation_local)
                    support_prior_scores = (
                        support_prior_scores.index_select(0, adaptation_local)
                        if support_prior_scores is not None
                        else None
                    )
                    support_images = _index_image_batch(support_images, adaptation_local)
                    validation_holdout_count = int(validation_local.numel())
                else:
                    validation_scores = None
                    validation_images = None
                    validation_prior_scores = None

    init_meta, user_state = _apply_support_srcc_tta_init(
        model=model,
        user_state=user_state,
        support_images=support_images,
        support_scores=support_scores,
        support_prior_scores=support_prior_scores,
        config=config,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    support_loo_indices = torch.arange(support_scores.numel(), device=device)
    chunks = _chunk_indices(support_loo_indices, chunk_size=chunk_size)
    last_loss = support_scores.new_tensor(0.0)
    rank_gate_applied = 0.0
    if lambda_regression <= 0.0 and lambda_rank <= 0.0 and lambda_l2 <= 0.0:
        return {key: value.detach() for key, value in user_state.items()}, {
            "tta_loss": 0.0,
            "tta_steps": 0.0,
            "tta_support_images": float(support_scores.numel()),
            "tta_support_chunks": float(len(chunks)),
            "tta_validation_images": float(validation_holdout_count),
            "tta_mode": 1.0 if mode == "train" else 0.0,
            "tta_lambda_support_regression": float(lambda_regression),
            "tta_support_leave_one_out": 1.0 if support_leave_one_out else 0.0,
            **init_meta,
        }

    model_parameters = tuple(model.parameters())
    original_requires_grad = tuple(parameter.requires_grad for parameter in model_parameters)
    for parameter in model_parameters:
        parameter.requires_grad_(False)

    try:
        for _ in range(max(steps, 0)):
            state_items = tuple(user_state.items())
            state_values = tuple(value for _, value in state_items)
            accumulated_grads = [torch.zeros_like(value) for value in state_values]
            step_loss_value = support_scores.new_tensor(0.0)
            for chunk in chunks:
                chunk_scores = support_scores.index_select(0, chunk)
                query_images_for_forward = _index_image_batch(support_images, chunk)
                query_prior_for_forward = (
                    support_prior_scores.index_select(0, chunk)
                    if support_prior_scores is not None
                    else None
                )
                fit_support_images = support_images
                fit_support_scores = support_scores
                fit_support_prior_scores = support_prior_scores
                if support_leave_one_out and support_scores.numel() > chunk.numel() + 1:
                    fit_support_images, fit_support_scores, fit_support_prior_scores = _support_without_chunk(
                        support_images,
                        support_scores,
                        support_prior_scores,
                        chunk,
                    )
                with torch.enable_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(
                        support_images=fit_support_images,
                        support_scores=fit_support_scores,
                        query_images=query_images_for_forward,
                        support_prior_scores=fit_support_prior_scores,
                        query_prior_scores=query_prior_for_forward,
                        user_adaptation=user_state,
                        compute_support_loo=False,
                    )
                    chunk_loss = support_scores.new_tensor(0.0)
                    if lambda_regression > 0.0:
                        chunk_loss = chunk_loss + lambda_regression * regression_loss(
                            outputs["score"],
                            chunk_scores,
                        )
                    if lambda_rank > 0.0:
                        chunk_loss = chunk_loss + lambda_rank * pairwise_ranking_loss(
                            outputs["score"],
                            chunk_scores,
                            mode=tta_support_ranking_mode,
                            margin=margin,
                            max_pairs=tta_support_ranking_max_pairs,
                        )
                    chunk_loss = chunk_loss / max(len(chunks), 1)
                if not chunk_loss.requires_grad:
                    step_loss_value = step_loss_value + chunk_loss.detach()
                    continue
                grads = torch.autograd.grad(
                    chunk_loss,
                    state_values,
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )
                for index, grad in enumerate(grads):
                    if grad is not None:
                        accumulated_grads[index] = accumulated_grads[index] + grad.detach()
                step_loss_value = step_loss_value + chunk_loss.detach()

            with torch.enable_grad():
                state_l2 = model.user_adaptation_l2(user_state)
                if state_l2 is not None and lambda_l2 > 0.0:
                    l2_loss = lambda_l2 * state_l2
                    grads = torch.autograd.grad(
                        l2_loss,
                        state_values,
                        allow_unused=True,
                        retain_graph=False,
                        create_graph=False,
                    )
                    for index, grad in enumerate(grads):
                        if grad is not None:
                            accumulated_grads[index] = accumulated_grads[index] + grad.detach()
                    step_loss_value = step_loss_value + l2_loss.detach()

            next_state: dict[str, torch.Tensor] = {}
            for (key, value), grad in zip(state_items, accumulated_grads):
                next_state[key] = (value - lr * grad).detach().requires_grad_(True)
            user_state = next_state
            last_loss = step_loss_value.detach()
    finally:
        for parameter, requires_grad in zip(model_parameters, original_requires_grad):
            parameter.requires_grad_(requires_grad)

    if validation_scores is not None and validation_images is not None:
        min_improvement = float(_mode_value(tta_cfg, "support_validation_min_rank_improvement", mode, 0.0))
        shrink = float(_mode_value(tta_cfg, "support_validation_shrink", mode, 0.35))
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            base_outputs = model(
                support_images=support_images,
                support_scores=support_scores,
                query_images=validation_images,
                support_prior_scores=support_prior_scores,
                query_prior_scores=validation_prior_scores,
                user_adaptation=None,
                compute_support_loo=False,
            )
            adapted_outputs = model(
                support_images=support_images,
                support_scores=support_scores,
                query_images=validation_images,
                support_prior_scores=support_prior_scores,
                query_prior_scores=validation_prior_scores,
                user_adaptation=user_state,
                compute_support_loo=False,
            )
            base_rank = pairwise_ranking_loss(
                base_outputs["score"],
                validation_scores,
                mode=tta_support_ranking_mode,
                margin=margin,
                max_pairs=tta_support_ranking_max_pairs,
            )
            adapted_rank = pairwise_ranking_loss(
                adapted_outputs["score"],
                validation_scores,
                mode=tta_support_ranking_mode,
                margin=margin,
                max_pairs=tta_support_ranking_max_pairs,
            )
        if float(adapted_rank.detach().cpu().item()) > float(base_rank.detach().cpu().item()) - min_improvement:
            user_state = {key: value * shrink for key, value in user_state.items()}
            rank_gate_applied = 1.0

    detached_state = {key: value.detach() for key, value in user_state.items()}
    return detached_state, {
        "tta_loss": float(last_loss.detach().cpu().item()),
        "tta_steps": float(max(steps, 0)),
        "tta_support_images": float(support_scores.numel()),
        "tta_support_chunks": float(len(chunks)),
        "tta_validation_images": float(validation_holdout_count),
        "tta_mode": 1.0 if mode == "train" else 0.0,
        "tta_lambda_support_regression": float(lambda_regression),
        "tta_rank_gate_applied": float(rank_gate_applied),
        "tta_support_leave_one_out": 1.0 if support_leave_one_out else 0.0,
        **init_meta,
    }
