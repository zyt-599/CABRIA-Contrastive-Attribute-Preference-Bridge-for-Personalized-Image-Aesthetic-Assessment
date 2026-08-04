from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from cabria.models.attribute_token_extractor import AttributeTokenExtractor
from cabria.models.cross_attention_bridge import CrossAttentionBridge
from cabria.models.preference_encoder import PreferenceEncoder
from cabria.models.score_heads import ProjectionHead, ScoreHead, TokenResidualHead, TokenScoreHead
from cabria.models.siglip_adapter import BackboneConfig, Siglip2AdapterBackbone
from cabria.utils.common import resolve_path


@dataclass
class CABRIAModelConfig:
    backbone: BackboneConfig
    num_attribute_tokens: int
    num_preference_tokens: int
    num_heads: int
    projection_dim: int
    dropout: float = 0.0
    residual_scale: float = 1.0
    bridge_residual_scale: float = 1.0
    stage2_residual_prior: bool = True
    cross_attention_bridge: bool = True
    bridge_support_fit: bool = False
    bridge_support_fit_ridge: float = 0.05
    bridge_support_fit_shrinkage: float = 20.0
    bridge_support_fit_max_abs_scale: float = 1.5
    bridge_support_fit_detach: bool = True
    bridge_basis_fit: bool = False
    bridge_basis_fit_ridge: float = 0.08
    bridge_basis_fit_shrinkage: float = 12.0
    bridge_basis_fit_scale: float = 0.0
    bridge_basis_fit_max_abs: float | None = None
    bridge_basis_fit_detach: bool = True
    support_residual_fusion: bool = False
    support_residual_fusion_components: tuple[str, ...] = (
        "bridge",
        "bridge_basis",
        "retrieved",
        "kernel",
        "user_state",
    )
    support_residual_fusion_ridge: float = 0.15
    support_residual_fusion_shrinkage: float = 12.0
    support_residual_fusion_max_abs_weight: float = 2.0
    support_residual_fusion_min_support: int = 50
    support_residual_fusion_validation_images: int = 20
    support_residual_fusion_validation_margin: float = 0.0
    support_residual_fusion_validation_metric: str = "residual_rank"
    support_residual_fusion_detach: bool = True
    support_residual_score_fit: bool = False
    support_residual_score_fit_ridge: float = 0.05
    support_residual_score_fit_shrinkage: float = 8.0
    support_residual_score_fit_max_abs_scale: float = 2.0
    support_residual_score_fit_min_support: int = 50
    support_residual_score_fit_validation_images: int = 0
    support_residual_score_fit_validation_margin: float = 0.0
    support_residual_score_fit_detach: bool = True
    support_residual_score_fit_candidate_scales: tuple[float, ...] = ()
    piaa_token_fusion: bool = False
    piaa_token_fusion_tokens: int = 4
    piaa_token_fusion_min_support: int = 50
    piaa_token_fusion_scale: float = 1.0
    piaa_token_fusion_detach: bool = True
    piaa_token_fusion_mode: str = "hard_topk"
    piaa_token_fusion_temperature: float = 0.7
    residual_max: float | None = None
    preference_mode: str = "hard"
    preference_temperature: float = 1.0
    score_bin_temperature: float = 0.7
    preference_memory_source: str = "pooled"
    preference_score_source: str = "target"
    base_score_source: str = "prior_stage1_delta"
    stage1_delta_scale: float = 0.0
    direct_support_memory: bool = False
    direct_support_memory_tokens: int = 0
    direct_support_memory_loo_tokens: int = 0
    direct_support_memory_scale: float = 1.0
    query_support_memory_topk: int = 0
    query_support_memory_scale: float = 1.0
    stage2_token_adapter: bool = False
    stage2_token_adapter_hidden_dim: int = 192
    stage2_token_adapter_dropout: float = 0.0
    stage2_token_adapter_scale: float = 0.5
    support_encode_chunk_size: int = 0
    support_calibration: bool = False
    calibration_hidden_dim: int = 128
    calibration_dropout: float = 0.0
    calibration_residual_gate_center: float = 1.0
    calibration_residual_gate_range: float = 0.5
    support_residual_retrieval: bool = False
    support_residual_similarity_source: str = "conditioned"
    support_retrieval_temperature: float = 0.2
    support_retrieval_scale: float = 0.0
    support_retrieval_detach: bool = True
    support_kernel_residual: bool = False
    support_kernel_residual_temperature: float = 0.03
    support_kernel_residual_ridge: float = 0.1
    support_kernel_residual_scale: float = 0.0
    support_kernel_residual_shrinkage: float = 0.0
    support_kernel_residual_max_abs: float | None = None
    support_kernel_residual_detach: bool = True
    linear_solve_device: str = "auto"
    transductive_retrieval: bool = False
    transductive_mode: str = "iterative"
    transductive_steps: int = 0
    transductive_alpha: float = 0.85
    transductive_ridge: float = 1.0e-4
    transductive_topk: int = 0
    tta_num_tokens: int = 0
    tta_init_std: float = 0.02
    tta_residual_log_scale_min: float = -1.5
    tta_residual_log_scale_max: float = 1.5
    tta_level_residual_log_scale_min: float = -0.5
    tta_level_residual_log_scale_max: float = 0.5
    tta_level_residual_bias_range: float = 0.0
    residual_memory_levels: int = 0
    residual_memory_temperature: float = 0.7
    residual_memory_confidence_tau: float = 4.0
    residual_memory_min_confidence_scale: float = 0.5
    semantic_prompt_memory: bool = False
    semantic_prompts: tuple[str, ...] = ()
    semantic_memory_tokens: int = 0
    semantic_memory_temperature: float = 0.07
    semantic_memory_scale: float = 0.2
    user_state_enabled: bool = False
    user_state_dim: int = 32
    user_state_ridge: float = 0.1
    user_state_scale: float = 0.2
    user_state_detach_support_residual: bool = True


class SupportCalibrationHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        residual_gate_center: float = 1.0,
        residual_gate_range: float = 0.5,
    ) -> None:
        super().__init__()
        self.residual_gate_center = float(residual_gate_center)
        self.residual_gate_range = max(float(residual_gate_range), 0.0)
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim * 2 + 6),
            nn.Linear(embed_dim * 2 + 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        positive_memory: torch.Tensor,
        negative_memory: torch.Tensor,
        support_stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        positive_summary = positive_memory.mean(dim=1)
        negative_summary = negative_memory.mean(dim=1)
        raw = self.net(torch.cat([positive_summary, negative_summary, support_stats], dim=1))
        return {
            "stage1_delta_adjust": 0.25 * torch.tanh(raw[:, 0]),
            "residual_gate": self.residual_gate_center + self.residual_gate_range * torch.tanh(raw[:, 1]),
            "bias": 0.5 * torch.tanh(raw[:, 2]),
            "raw": raw,
        }


class CABRIAStage1Model(nn.Module):
    def __init__(self, config: CABRIAModelConfig) -> None:
        super().__init__()
        self.backbone = Siglip2AdapterBackbone(config.backbone)
        self.attribute_extractor = AttributeTokenExtractor(
            embed_dim=config.backbone.embed_dim,
            num_attribute_tokens=config.num_attribute_tokens,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.general_head = ScoreHead(config.backbone.embed_dim, dropout=config.dropout)
        self.attribute_head = TokenScoreHead(config.backbone.embed_dim, dropout=config.dropout)
        self.projection_head = ProjectionHead(config.backbone.embed_dim, config.projection_dim, dropout=config.dropout)

    def forward(self, images: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self._encode(images)
        attribute_outputs = self.attribute_extractor(features["patch_tokens"])
        attribute_tokens = attribute_outputs["attribute_tokens"]
        attribute_score_outputs = self.attribute_head(attribute_tokens)
        return {
            "score": self.general_head(features["pooled_output"]),
            "patch_tokens": features["patch_tokens"],
            "pooled_output": features["pooled_output"],
            "attribute_tokens": attribute_tokens,
            "attribute_score": attribute_score_outputs["score"],
            "attribute_pooled": attribute_score_outputs["pooled_tokens"],
            "attribute_pool_weights": attribute_score_outputs["token_weights"],
            "projected_tokens": self.projection_head(attribute_tokens),
            "attribute_attention": attribute_outputs["attention_weights"],
        }

    def _encode(self, images: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if isinstance(images, dict):
            return self.backbone(
                pixel_values=images["pixel_values"],
                pixel_attention_mask=images.get("pixel_attention_mask"),
                spatial_shapes=images.get("spatial_shapes"),
            )
        return self.backbone(pixel_values=images)


class CABRIAStage2Model(nn.Module):
    def __init__(
        self,
        config: CABRIAModelConfig,
        stage1_checkpoint: str | None = None,
        contrast_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        self.stage1 = CABRIAStage1Model(config)
        if stage1_checkpoint:
            checkpoint = torch.load(resolve_path(stage1_checkpoint), map_location="cpu")
            self.stage1.load_state_dict(checkpoint["model"], strict=True)
        self.contrast_stage1 = None
        if contrast_checkpoint:
            self.contrast_stage1 = CABRIAStage1Model(config)
            checkpoint = torch.load(resolve_path(contrast_checkpoint), map_location="cpu")
            self.contrast_stage1.load_state_dict(checkpoint["model"], strict=True)
        self.residual_scale = config.residual_scale
        self.bridge_residual_scale = float(config.bridge_residual_scale)
        self.stage2_residual_prior = bool(config.stage2_residual_prior)
        self.cross_attention_bridge = bool(config.cross_attention_bridge)
        self.residual_max = config.residual_max
        self.preference_memory_source = config.preference_memory_source.lower()
        self.preference_score_source = config.preference_score_source.lower()
        self.base_score_source = config.base_score_source.lower()
        self.stage1_delta_scale = float(config.stage1_delta_scale)
        self.bridge_support_fit = bool(config.bridge_support_fit)
        self.bridge_support_fit_ridge = max(float(config.bridge_support_fit_ridge), 1e-6)
        self.bridge_support_fit_shrinkage = max(float(config.bridge_support_fit_shrinkage), 0.0)
        self.bridge_support_fit_max_abs_scale = max(float(config.bridge_support_fit_max_abs_scale), 0.0)
        self.bridge_support_fit_detach = bool(config.bridge_support_fit_detach)
        self.bridge_basis_fit = bool(config.bridge_basis_fit)
        self.bridge_basis_fit_ridge = max(float(config.bridge_basis_fit_ridge), 1e-6)
        self.bridge_basis_fit_shrinkage = max(float(config.bridge_basis_fit_shrinkage), 0.0)
        self.bridge_basis_fit_scale = float(config.bridge_basis_fit_scale)
        self.bridge_basis_fit_max_abs = config.bridge_basis_fit_max_abs
        self.bridge_basis_fit_detach = bool(config.bridge_basis_fit_detach)
        self.support_residual_fusion = bool(config.support_residual_fusion)
        self.support_residual_fusion_components = tuple(
            str(component).lower()
            for component in config.support_residual_fusion_components
        )
        self.support_residual_fusion_ridge = max(float(config.support_residual_fusion_ridge), 1e-6)
        self.support_residual_fusion_shrinkage = max(float(config.support_residual_fusion_shrinkage), 0.0)
        self.support_residual_fusion_max_abs_weight = max(
            float(config.support_residual_fusion_max_abs_weight),
            0.0,
        )
        self.support_residual_fusion_min_support = max(int(config.support_residual_fusion_min_support), 2)
        self.support_residual_fusion_validation_images = max(
            int(config.support_residual_fusion_validation_images),
            0,
        )
        self.support_residual_fusion_validation_margin = max(
            float(config.support_residual_fusion_validation_margin),
            0.0,
        )
        self.support_residual_fusion_validation_metric = str(
            config.support_residual_fusion_validation_metric
        ).lower()
        self.support_residual_fusion_detach = bool(config.support_residual_fusion_detach)
        self.support_residual_score_fit = bool(config.support_residual_score_fit)
        self.support_residual_score_fit_ridge = max(float(config.support_residual_score_fit_ridge), 1e-6)
        self.support_residual_score_fit_shrinkage = max(float(config.support_residual_score_fit_shrinkage), 0.0)
        self.support_residual_score_fit_max_abs_scale = max(
            float(config.support_residual_score_fit_max_abs_scale),
            0.0,
        )
        self.support_residual_score_fit_min_support = max(int(config.support_residual_score_fit_min_support), 2)
        self.support_residual_score_fit_validation_images = max(
            int(config.support_residual_score_fit_validation_images),
            0,
        )
        self.support_residual_score_fit_validation_margin = max(
            float(config.support_residual_score_fit_validation_margin),
            0.0,
        )
        self.support_residual_score_fit_detach = bool(config.support_residual_score_fit_detach)
        self.support_residual_score_fit_candidate_scales = tuple(
            float(scale) for scale in config.support_residual_score_fit_candidate_scales
        )
        self.piaa_token_fusion = bool(config.piaa_token_fusion)
        self.piaa_token_fusion_tokens = max(int(config.piaa_token_fusion_tokens), 0)
        self.piaa_token_fusion_min_support = max(int(config.piaa_token_fusion_min_support), 2)
        self.piaa_token_fusion_scale = max(float(config.piaa_token_fusion_scale), 0.0)
        self.piaa_token_fusion_detach = bool(config.piaa_token_fusion_detach)
        self.piaa_token_fusion_mode = str(config.piaa_token_fusion_mode).lower()
        self.piaa_token_fusion_temperature = max(float(config.piaa_token_fusion_temperature), 1.0e-6)
        self.direct_support_memory = bool(config.direct_support_memory)
        self.direct_support_memory_tokens = max(int(config.direct_support_memory_tokens), 0)
        self.direct_support_memory_loo_tokens = max(int(config.direct_support_memory_loo_tokens), 0)
        self.direct_support_memory_scale = float(config.direct_support_memory_scale)
        self.query_support_memory_topk = max(int(config.query_support_memory_topk), 0)
        self.query_support_memory_scale = float(config.query_support_memory_scale)
        self.support_encode_chunk_size = max(int(config.support_encode_chunk_size), 0)
        self.stage2_token_adapter_scale = float(config.stage2_token_adapter_scale)
        self.stage2_token_adapter = self._build_stage2_token_adapter(config)
        self.preference_encoder = PreferenceEncoder(
            config.backbone.embed_dim,
            config.num_preference_tokens,
            config.num_heads,
            dropout=config.dropout,
            mode=config.preference_mode,
            temperature=config.preference_temperature,
            score_bin_temperature=config.score_bin_temperature,
        )
        self.bridge = CrossAttentionBridge(config.backbone.embed_dim, config.num_heads, dropout=config.dropout)
        self.residual_head = TokenResidualHead(config.backbone.embed_dim, dropout=config.dropout)
        self.support_calibration = config.support_calibration
        self.support_residual_retrieval = bool(config.support_residual_retrieval)
        self.support_residual_similarity_source = str(config.support_residual_similarity_source).lower()
        self.support_retrieval_temperature = max(float(config.support_retrieval_temperature), 1e-6)
        self.support_retrieval_detach = bool(config.support_retrieval_detach)
        self.transductive_retrieval = bool(config.transductive_retrieval)
        self.transductive_mode = config.transductive_mode.lower()
        self.transductive_steps = max(int(config.transductive_steps), 0)
        self.transductive_alpha = min(max(float(config.transductive_alpha), 0.0), 0.99)
        self.transductive_ridge = max(float(config.transductive_ridge), 0.0)
        self.transductive_topk = max(int(config.transductive_topk), 0)
        self.support_retrieval_gate = (
            nn.Parameter(torch.tensor(float(config.support_retrieval_scale)))
            if self.support_residual_retrieval
            else None
        )
        self.support_kernel_residual = bool(config.support_kernel_residual)
        self.support_kernel_residual_temperature = max(float(config.support_kernel_residual_temperature), 1e-6)
        self.support_kernel_residual_ridge = max(float(config.support_kernel_residual_ridge), 1e-6)
        self.support_kernel_residual_scale = float(config.support_kernel_residual_scale)
        self.support_kernel_residual_shrinkage = max(float(config.support_kernel_residual_shrinkage), 0.0)
        self.support_kernel_residual_max_abs = config.support_kernel_residual_max_abs
        self.support_kernel_residual_detach = bool(config.support_kernel_residual_detach)
        self.linear_solve_device = str(config.linear_solve_device).lower()
        if self.linear_solve_device not in {"auto", "cuda", "cpu"}:
            raise ValueError("model.linear_solve_device must be one of: auto, cuda, cpu.")
        self.user_state_enabled = bool(config.user_state_enabled)
        self.user_state_ridge = max(float(config.user_state_ridge), 1e-6)
        self.user_state_detach_support_residual = bool(config.user_state_detach_support_residual)
        if self.user_state_enabled:
            user_state_dim = max(int(config.user_state_dim), 1)
            self.user_state_projection = nn.Sequential(
                nn.LayerNorm(config.backbone.embed_dim),
                nn.Linear(config.backbone.embed_dim, user_state_dim, bias=False),
            )
            self.user_state_gate = nn.Parameter(torch.tensor(float(config.user_state_scale)))
        else:
            self.user_state_projection = None
            self.user_state_gate = None
        self.tta_num_tokens = int(config.tta_num_tokens)
        self.tta_init_std = float(config.tta_init_std)
        self.tta_residual_log_scale_min = float(config.tta_residual_log_scale_min)
        self.tta_residual_log_scale_max = float(config.tta_residual_log_scale_max)
        self.tta_level_residual_log_scale_min = float(config.tta_level_residual_log_scale_min)
        self.tta_level_residual_log_scale_max = float(config.tta_level_residual_log_scale_max)
        self.tta_level_residual_bias_range = max(float(config.tta_level_residual_bias_range), 0.0)
        self.residual_memory_levels = max(int(config.residual_memory_levels), 0)
        self.residual_memory_temperature = max(float(config.residual_memory_temperature), 1e-6)
        self.residual_memory_confidence_tau = max(float(config.residual_memory_confidence_tau), 1e-6)
        self.residual_memory_min_confidence_scale = min(
            max(float(config.residual_memory_min_confidence_scale), 0.0), 1.0
        )
        self.semantic_prompt_memory = bool(config.semantic_prompt_memory)
        self.semantic_prompts = tuple(config.semantic_prompts)
        self.semantic_memory_tokens = max(int(config.semantic_memory_tokens), 0)
        self.semantic_memory_temperature = max(float(config.semantic_memory_temperature), 1e-6)
        self.semantic_memory_enabled = (
            self.semantic_prompt_memory
            and self.semantic_memory_tokens > 0
            and len(self.semantic_prompts) > 0
        )
        if self.semantic_memory_enabled:
            semantic_features = self._load_semantic_text_features(config)
            self.register_buffer("semantic_text_features", semantic_features, persistent=False)
            self.semantic_score_embed = nn.Sequential(
                nn.Linear(1, config.backbone.embed_dim),
                nn.GELU(),
                nn.Linear(config.backbone.embed_dim, config.backbone.embed_dim),
            )
            self.semantic_memory_norm = nn.LayerNorm(config.backbone.embed_dim)
            self.semantic_memory_gate = nn.Parameter(torch.tensor(float(config.semantic_memory_scale)))
        else:
            self.register_buffer(
                "semantic_text_features",
                torch.empty(0, config.backbone.embed_dim),
                persistent=False,
            )
            self.semantic_score_embed = None
            self.semantic_memory_norm = None
            self.semantic_memory_gate = None

        if self.tta_num_tokens > 0:
            self.tta_positive_template = nn.Parameter(torch.zeros(1, self.tta_num_tokens, config.backbone.embed_dim))
            self.tta_negative_template = nn.Parameter(torch.zeros(1, self.tta_num_tokens, config.backbone.embed_dim))
            self.tta_residual_log_scale_template = nn.Parameter(torch.zeros(1))
            self.tta_retrieval_log_scale_template = nn.Parameter(torch.zeros(1))
            self.tta_stage1_delta_adjust_template = nn.Parameter(torch.zeros(1))
            self.tta_bias_template = nn.Parameter(torch.zeros(1))
            self.tta_positive_memory_log_weights_template = nn.Parameter(torch.zeros(1, config.num_preference_tokens))
            self.tta_negative_memory_log_weights_template = nn.Parameter(torch.zeros(1, config.num_preference_tokens))
            self.tta_level_memory_log_weights_template = nn.Parameter(torch.zeros(1, max(self.residual_memory_levels, 1)))
            self.tta_level_residual_log_scales_template = nn.Parameter(torch.zeros(1, max(self.residual_memory_levels, 1)))
            self.tta_level_residual_biases_template = nn.Parameter(torch.zeros(1, max(self.residual_memory_levels, 1)))
            self.tta_semantic_memory_log_weights_template = nn.Parameter(torch.zeros(1, max(self.semantic_memory_tokens, 1)))
            nn.init.normal_(self.tta_positive_template, mean=0.0, std=self.tta_init_std)
            nn.init.normal_(self.tta_negative_template, mean=0.0, std=self.tta_init_std)
        else:
            self.tta_positive_template = None
            self.tta_negative_template = None
            self.tta_residual_log_scale_template = None
            self.tta_retrieval_log_scale_template = None
            self.tta_stage1_delta_adjust_template = None
            self.tta_bias_template = None
            self.tta_positive_memory_log_weights_template = None
            self.tta_negative_memory_log_weights_template = None
            self.tta_level_memory_log_weights_template = None
            self.tta_level_residual_log_scales_template = None
            self.tta_level_residual_biases_template = None
            self.tta_semantic_memory_log_weights_template = None
        self.calibration_head = (
            SupportCalibrationHead(
                embed_dim=config.backbone.embed_dim,
                hidden_dim=config.calibration_hidden_dim,
                dropout=config.calibration_dropout,
                residual_gate_center=config.calibration_residual_gate_center,
                residual_gate_range=config.calibration_residual_gate_range,
            )
            if config.support_calibration
            else None
        )

    def _build_stage2_token_adapter(self, config: CABRIAModelConfig) -> nn.Module | None:
        if not bool(config.stage2_token_adapter):
            return None
        embed_dim = int(config.backbone.embed_dim)
        hidden_dim = max(int(config.stage2_token_adapter_hidden_dim), 1)
        adapter = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(config.stage2_token_adapter_dropout)),
            nn.Linear(hidden_dim, embed_dim),
        )
        final_layer = adapter[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        return adapter

    def _adapt_attribute_tokens(self, attribute_tokens: torch.Tensor) -> torch.Tensor:
        if self.stage2_token_adapter is None:
            return attribute_tokens
        delta = self.stage2_token_adapter(attribute_tokens)
        return attribute_tokens + self.stage2_token_adapter_scale * delta

    def _base_score(
        self,
        stage1_score: torch.Tensor,
        prior_scores: torch.Tensor | None,
    ) -> torch.Tensor:
        if prior_scores is None or self.base_score_source == "stage1":
            return stage1_score.detach()
        if self.base_score_source == "prior":
            return prior_scores
        if self.base_score_source in {"prior_stage1_delta", "prior_delta", "blend"}:
            return prior_scores + self.stage1_delta_scale * (stage1_score.detach() - prior_scores)
        raise ValueError(f"Unsupported base score source: {self.base_score_source}")

    def _slice_images(
        self,
        images: torch.Tensor | dict[str, torch.Tensor],
        start: int,
        end: int,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(images, dict):
            return {key: value[start:end] for key, value in images.items()}
        return images[start:end]

    def _image_batch_size(self, images: torch.Tensor | dict[str, torch.Tensor]) -> int:
        if isinstance(images, dict):
            first_value = next(iter(images.values()))
            return int(first_value.shape[0])
        return int(images.shape[0])

    def _encode_stage1_chunked(
        self,
        stage1_model: CABRIAStage1Model,
        images: torch.Tensor | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch_size = self._image_batch_size(images)
        chunk_size = self.support_encode_chunk_size
        if chunk_size <= 0 or batch_size <= chunk_size:
            return stage1_model._encode(images)
        chunks: list[dict[str, torch.Tensor]] = []
        for start in range(0, batch_size, chunk_size):
            chunks.append(stage1_model._encode(self._slice_images(images, start, min(start + chunk_size, batch_size))))
        return {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in chunks[0]
        }

    def _forward_stage1_chunked(
        self,
        stage1_model: CABRIAStage1Model,
        images: torch.Tensor | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch_size = self._image_batch_size(images)
        chunk_size = self.support_encode_chunk_size
        if chunk_size <= 0 or batch_size <= chunk_size:
            return stage1_model(images)
        chunks: list[dict[str, torch.Tensor]] = []
        for start in range(0, batch_size, chunk_size):
            chunks.append(stage1_model(self._slice_images(images, start, min(start + chunk_size, batch_size))))
        return {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in chunks[0]
            if torch.is_tensor(chunks[0][key])
        }

    def _scale_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if self.residual_max is not None:
            residual = torch.tanh(residual) * float(self.residual_max)
        return residual * float(self.residual_scale)

    def _load_semantic_text_features(self, config: CABRIAModelConfig) -> torch.Tensor:
        try:
            from transformers import AutoTokenizer, Siglip2Model
        except ImportError as exc:
            raise ImportError("transformers is required for semantic prompt memory.") from exc
        model_source = config.backbone.model_dir or config.backbone.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=bool(config.backbone.model_dir))
        text_model = Siglip2Model.from_pretrained(model_source, local_files_only=bool(config.backbone.model_dir))
        text_model.eval()
        with torch.no_grad():
            inputs = tokenizer(list(config.semantic_prompts), padding=True, return_tensors="pt")
            features = text_model.get_text_features(**inputs).float()
            features = F.normalize(features, dim=-1)
        del text_model
        return features

    def _semantic_prompt_scores(self, pooled_output: torch.Tensor) -> torch.Tensor | None:
        if not self.semantic_memory_enabled or self.semantic_text_features.numel() == 0:
            return None
        image_features = F.normalize(pooled_output.float(), dim=-1)
        text_features = self.semantic_text_features.to(device=image_features.device, dtype=image_features.dtype)
        return image_features @ text_features.transpose(0, 1)

    def make_user_adaptation(
        self,
        device: torch.device,
        requires_grad: bool = True,
    ) -> dict[str, torch.Tensor] | None:
        if self.tta_num_tokens <= 0 or self.tta_positive_template is None or self.tta_negative_template is None:
            return None
        state = {
            "positive_tokens": self.tta_positive_template.detach().clone().to(device),
            "negative_tokens": self.tta_negative_template.detach().clone().to(device),
            "residual_log_scale": self.tta_residual_log_scale_template.detach().clone().to(device),
            "retrieval_log_scale": self.tta_retrieval_log_scale_template.detach().clone().to(device),
            "stage1_delta_adjust": self.tta_stage1_delta_adjust_template.detach().clone().to(device),
            "bias": self.tta_bias_template.detach().clone().to(device),
            "positive_memory_log_weights": self.tta_positive_memory_log_weights_template.detach().clone().to(device),
            "negative_memory_log_weights": self.tta_negative_memory_log_weights_template.detach().clone().to(device),
        }
        if self.residual_memory_levels > 0 and self.tta_level_memory_log_weights_template is not None:
            state["level_memory_log_weights"] = (
                self.tta_level_memory_log_weights_template[:, : self.residual_memory_levels]
                .detach()
                .clone()
                .to(device)
            )
        if self.residual_memory_levels > 0 and self.tta_level_residual_log_scales_template is not None:
            state["level_residual_log_scales"] = (
                self.tta_level_residual_log_scales_template[:, : self.residual_memory_levels]
                .detach()
                .clone()
                .to(device)
            )
        if self.residual_memory_levels > 0 and self.tta_level_residual_biases_template is not None:
            state["level_residual_biases"] = (
                self.tta_level_residual_biases_template[:, : self.residual_memory_levels]
                .detach()
                .clone()
                .to(device)
            )
        if self.semantic_memory_enabled and self.tta_semantic_memory_log_weights_template is not None:
            state["semantic_memory_log_weights"] = (
                self.tta_semantic_memory_log_weights_template[:, : self.semantic_memory_tokens]
                .detach()
                .clone()
                .to(device)
            )
        if requires_grad:
            state = {key: value.requires_grad_(True) for key, value in state.items()}
        return state

    @staticmethod
    def user_adaptation_l2(user_adaptation: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
        if not user_adaptation:
            return None
        values = [value.pow(2).mean() for value in user_adaptation.values() if value.numel() > 0]
        if not values:
            return None
        return torch.stack(values).mean()

    @staticmethod
    def _append_user_memory(
        memory: torch.Tensor,
        user_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if user_tokens is None:
            return memory
        if user_tokens.dim() == 2:
            user_tokens = user_tokens.unsqueeze(0)
        return torch.cat([memory, user_tokens.expand(memory.size(0), -1, -1)], dim=1)

    @staticmethod
    def _apply_user_memory_weights(
        memory: torch.Tensor,
        user_adaptation: dict[str, torch.Tensor] | None,
        key: str,
    ) -> torch.Tensor:
        if not user_adaptation or key not in user_adaptation:
            return memory
        log_weights = user_adaptation[key]
        if log_weights.dim() == 1:
            log_weights = log_weights.unsqueeze(0)
        usable = min(memory.size(1), log_weights.size(1))
        if usable <= 0:
            return memory
        weights = torch.exp(torch.clamp(log_weights[:, :usable], min=-1.5, max=1.5)).unsqueeze(-1)
        weighted_prefix = memory[:, :usable, :] * weights.to(dtype=memory.dtype)
        if usable == memory.size(1):
            return weighted_prefix
        return torch.cat([weighted_prefix, memory[:, usable:, :]], dim=1)

    def _apply_user_residual_adaptation(
        self,
        residual_score: torch.Tensor,
        user_adaptation: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not user_adaptation:
            return residual_score
        log_scale = user_adaptation.get("residual_log_scale")
        bias = user_adaptation.get("bias")
        if log_scale is not None:
            scale = torch.exp(
                torch.clamp(
                    log_scale,
                    min=self.tta_residual_log_scale_min,
                    max=self.tta_residual_log_scale_max,
                )
            ).expand_as(residual_score)
            residual_score = residual_score * scale
        if bias is not None:
            residual_score = residual_score + bias.expand_as(residual_score)
        return residual_score

    def _residual_level_assignments(self, residual: torch.Tensor) -> torch.Tensor | None:
        if self.residual_memory_levels <= 0 or residual.numel() == 0:
            return None
        values = residual.detach().float().view(-1)
        value_std = values.std(unbiased=False).clamp_min(1e-6)
        normalized = ((values - values.mean()) / value_std).clamp(min=-2.0, max=2.0)
        centers = torch.linspace(
            -2.0,
            2.0,
            steps=self.residual_memory_levels,
            device=residual.device,
            dtype=torch.float32,
        )
        logits = -torch.abs(normalized.view(-1, 1) - centers.view(1, -1)) / self.residual_memory_temperature
        return torch.softmax(logits, dim=1).to(dtype=residual.dtype)

    def _apply_level_residual_adaptation(
        self,
        support_residual: torch.Tensor,
        user_adaptation: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not user_adaptation or self.residual_memory_levels <= 0:
            return support_residual
        assignments = self._residual_level_assignments(support_residual)
        if assignments is None:
            return support_residual
        adjusted = support_residual
        level_log_scales = user_adaptation.get("level_residual_log_scales")
        if level_log_scales is not None:
            if level_log_scales.dim() == 2:
                level_log_scales = level_log_scales.squeeze(0)
            usable = min(assignments.size(1), level_log_scales.numel())
            if usable > 0:
                level_scales = torch.exp(
                    torch.clamp(
                        level_log_scales[:usable],
                        min=self.tta_level_residual_log_scale_min,
                        max=self.tta_level_residual_log_scale_max,
                    )
                ).to(dtype=support_residual.dtype)
                adjusted = adjusted * torch.matmul(assignments[:, :usable], level_scales)
        level_biases = user_adaptation.get("level_residual_biases")
        if level_biases is not None and self.tta_level_residual_bias_range > 0.0:
            if level_biases.dim() == 2:
                level_biases = level_biases.squeeze(0)
            usable = min(assignments.size(1), level_biases.numel())
            if usable > 0:
                bounded_biases = self.tta_level_residual_bias_range * torch.tanh(level_biases[:usable])
                adjusted = adjusted + torch.matmul(assignments[:, :usable], bounded_biases.to(dtype=support_residual.dtype))
        return adjusted

    def _preference_memory_inputs(
        self,
        support_attribute_tokens: torch.Tensor,
        support_image_embeddings: torch.Tensor,
        memory_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.preference_memory_source == "pooled":
            return support_image_embeddings, memory_scores
        if self.preference_memory_source == "attribute_tokens":
            num_support, num_tokens, embed_dim = support_attribute_tokens.shape
            memory_embeddings = support_attribute_tokens.reshape(1, num_support * num_tokens, embed_dim)
            expanded_scores = memory_scores.repeat_interleave(num_tokens).unsqueeze(0)
            return memory_embeddings, expanded_scores
        raise ValueError(f"Unsupported preference memory source: {self.preference_memory_source}")

    def _piaa_token_fusion_memory(
        self,
        support_attribute_tokens: torch.Tensor,
        support_scores: torch.Tensor,
        support_base_score: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            not self.piaa_token_fusion
            or self.piaa_token_fusion_tokens <= 0
            or support_scores.numel() < self.piaa_token_fusion_min_support
        ):
            return None, None
        tokens = support_attribute_tokens
        residual = support_scores - support_base_score
        if self.piaa_token_fusion_detach:
            tokens = tokens.detach()
            residual = residual.detach()
        residual = residual.float().view(-1)
        residual = residual - residual.mean()
        num_support, num_tokens, embed_dim = tokens.shape
        count = min(self.piaa_token_fusion_tokens, num_support * num_tokens)
        if count <= 0:
            return None, None
        if self.piaa_token_fusion_mode in {"soft", "softmax", "prototype", "residual_softmax", "hybrid"}:
            center_by_token = tokens.mean(dim=0)
            normalized = residual / residual.std(unbiased=False).clamp_min(self.preference_encoder.eps)
            normalized = normalized.clamp(min=-3.0, max=3.0)
            temperature = self.piaa_token_fusion_temperature
            positive_weights = torch.softmax(normalized / temperature, dim=0).to(dtype=tokens.dtype)
            negative_weights = torch.softmax(-normalized / temperature, dim=0).to(dtype=tokens.dtype)
            soft_positive = torch.einsum("n,ntd->td", positive_weights, tokens)
            soft_negative = torch.einsum("n,ntd->td", negative_weights, tokens)
            soft_count = min(count, num_tokens)
            soft_center = center_by_token[:soft_count]
            soft_positive = soft_positive[:soft_count]
            soft_negative = soft_negative[:soft_count]
            if self.piaa_token_fusion_mode == "hybrid":
                flat_tokens = tokens.reshape(num_support * num_tokens, embed_dim)
                token_residual = residual.repeat_interleave(num_tokens)
                hard_count = max(1, min(count, flat_tokens.size(0)) // 2)
                positive_indices = torch.topk(token_residual, k=hard_count, largest=True).indices
                negative_indices = torch.topk(token_residual, k=hard_count, largest=False).indices
                hard_center = flat_tokens.mean(dim=0, keepdim=True).expand(hard_count, -1)
                positive_tokens = torch.cat([soft_positive, flat_tokens.index_select(0, positive_indices)], dim=0)
                negative_tokens = torch.cat([soft_negative, flat_tokens.index_select(0, negative_indices)], dim=0)
                center = torch.cat([soft_center, hard_center], dim=0)
            else:
                center = soft_center
                positive_tokens = soft_positive
                negative_tokens = soft_negative
        elif self.piaa_token_fusion_mode in {"hard", "hard_topk", "topk"}:
            flat_tokens = tokens.reshape(num_support * num_tokens, embed_dim)
            token_residual = residual.repeat_interleave(num_tokens)
            positive_indices = torch.topk(token_residual, k=count, largest=True).indices
            negative_indices = torch.topk(token_residual, k=count, largest=False).indices
            center = flat_tokens.mean(dim=0, keepdim=True)
            positive_tokens = flat_tokens.index_select(0, positive_indices)
            negative_tokens = flat_tokens.index_select(0, negative_indices)
        else:
            raise ValueError(f"Unsupported piaa_token_fusion_mode: {self.piaa_token_fusion_mode}")
        scale = float(self.piaa_token_fusion_scale)
        positive_tokens = center + scale * (positive_tokens - center)
        negative_tokens = center + scale * (negative_tokens - center)
        return positive_tokens.unsqueeze(0), negative_tokens.unsqueeze(0)

    @staticmethod
    def _append_piaa_token_fusion_memory(
        positive_memory: torch.Tensor,
        negative_memory: torch.Tensor,
        piaa_positive_memory: torch.Tensor | None,
        piaa_negative_memory: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if piaa_positive_memory is not None:
            positive_memory = torch.cat([positive_memory, piaa_positive_memory.to(dtype=positive_memory.dtype)], dim=1)
        if piaa_negative_memory is not None:
            negative_memory = torch.cat([negative_memory, piaa_negative_memory.to(dtype=negative_memory.dtype)], dim=1)
        return positive_memory, negative_memory

    def _append_direct_support_memory(
        self,
        positive_memory: torch.Tensor,
        negative_memory: torch.Tensor,
        memory_embeddings: torch.Tensor,
        memory_scores: torch.Tensor,
        max_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_limit = self.direct_support_memory_tokens if max_tokens is None else max(int(max_tokens), 0)
        if (
            not self.direct_support_memory
            or token_limit <= 0
            or memory_embeddings.numel() == 0
            or memory_scores.numel() == 0
        ):
            return positive_memory, negative_memory
        if memory_embeddings.dim() == 2:
            memory_embeddings = memory_embeddings.unsqueeze(0)
        if memory_scores.dim() == 1:
            memory_scores = memory_scores.unsqueeze(0)
        num_memory = memory_embeddings.size(1)
        k = min(token_limit, num_memory)
        if k <= 0:
            return positive_memory, negative_memory

        centered_scores = memory_scores - memory_scores.mean(dim=1, keepdim=True)
        normalized_scores = centered_scores / (memory_scores.std(dim=1, keepdim=True, unbiased=False) + self.preference_encoder.eps)
        normalized_support = self.preference_encoder.support_norm(memory_embeddings)

        def select_tokens(descending: bool) -> tuple[torch.Tensor, torch.Tensor]:
            indices = torch.argsort(normalized_scores, dim=1, descending=descending)[:, :k]
            gather_indices = indices.unsqueeze(-1).expand(-1, -1, normalized_support.size(-1))
            selected_embeddings = torch.gather(normalized_support, 1, gather_indices)
            selected_scores = torch.gather(normalized_scores, 1, indices)
            return selected_embeddings, selected_scores

        positive_embeddings, positive_scores = select_tokens(descending=True)
        negative_embeddings, negative_scores = select_tokens(descending=False)
        positive_direct = self.preference_encoder.output_norm(
            positive_embeddings + self.preference_encoder.positive_score_embed(torch.relu(positive_scores).unsqueeze(-1))
        ) * self.direct_support_memory_scale
        negative_direct = self.preference_encoder.output_norm(
            negative_embeddings + self.preference_encoder.negative_score_embed(torch.relu(-negative_scores).unsqueeze(-1))
        ) * self.direct_support_memory_scale
        return (
            torch.cat([positive_memory, positive_direct], dim=1),
            torch.cat([negative_memory, negative_direct], dim=1),
        )

    def _query_direct_support_memory(
        self,
        query_attribute_tokens: torch.Tensor,
        support_attribute_tokens: torch.Tensor,
        support_image_embeddings: torch.Tensor,
        memory_scores: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            self.query_support_memory_topk <= 0
            or query_attribute_tokens.numel() == 0
            or support_attribute_tokens.numel() == 0
            or support_image_embeddings.numel() == 0
            or memory_scores.numel() == 0
        ):
            return None, None
        num_support, num_tokens, embed_dim = support_attribute_tokens.shape
        topk = min(self.query_support_memory_topk, num_support)
        if topk <= 0:
            return None, None

        query_summary = F.normalize(query_attribute_tokens.mean(dim=1).float(), dim=-1)
        support_summary = F.normalize(support_image_embeddings.float(), dim=-1)
        similarity = torch.matmul(query_summary, support_summary.transpose(0, 1))
        indices = torch.topk(similarity, k=topk, dim=1).indices

        flat_indices = indices.reshape(-1)
        selected_tokens = support_attribute_tokens.index_select(0, flat_indices)
        selected_tokens = selected_tokens.reshape(query_attribute_tokens.size(0), topk * num_tokens, embed_dim)

        centered_scores = memory_scores - memory_scores.mean()
        normalized_scores = centered_scores / (memory_scores.std(unbiased=False) + self.preference_encoder.eps)
        selected_scores = normalized_scores.index_select(0, flat_indices)
        selected_scores = selected_scores.reshape(query_attribute_tokens.size(0), topk)
        selected_scores = selected_scores.repeat_interleave(num_tokens, dim=1)

        normalized_support = self.preference_encoder.support_norm(selected_tokens)
        positive_direct = self.preference_encoder.output_norm(
            normalized_support + self.preference_encoder.positive_score_embed(torch.relu(selected_scores).unsqueeze(-1))
        ) * self.query_support_memory_scale
        negative_direct = self.preference_encoder.output_norm(
            normalized_support + self.preference_encoder.negative_score_embed(torch.relu(-selected_scores).unsqueeze(-1))
        ) * self.query_support_memory_scale
        return positive_direct, negative_direct

    def _preference_memory_scores(
        self,
        support_scores: torch.Tensor,
        support_base_score: torch.Tensor,
    ) -> torch.Tensor:
        if self.preference_score_source == "target":
            return support_scores.detach()
        if self.preference_score_source == "residual":
            return (support_scores - support_base_score).detach()
        raise ValueError(f"Unsupported preference score source: {self.preference_score_source}")

    def _residual_level_memory(
        self,
        support_embeddings: torch.Tensor,
        memory_scores: torch.Tensor,
        user_adaptation: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if self.residual_memory_levels <= 0 or support_embeddings.numel() == 0:
            return None
        scores = memory_scores.detach().view(-1)
        score_std = scores.std(unbiased=False).clamp_min(1e-6)
        normalized_scores = ((scores - scores.mean()) / score_std).clamp(min=-2.0, max=2.0)
        centers = torch.linspace(
            -2.0,
            2.0,
            steps=self.residual_memory_levels,
            device=support_embeddings.device,
            dtype=support_embeddings.dtype,
        )
        logits = -torch.abs(normalized_scores.view(-1, 1) - centers.view(1, -1)) / self.residual_memory_temperature
        assignments = torch.softmax(logits.float(), dim=1)
        level_mass = assignments.sum(dim=0).clamp_min(1e-6)
        weights = (assignments / level_mass.view(1, -1)).to(dtype=support_embeddings.dtype)
        level_memory = torch.matmul(weights.transpose(0, 1), support_embeddings.detach()).unsqueeze(0)
        confidence = level_mass / (level_mass + self.residual_memory_confidence_tau)
        confidence_scale = self.residual_memory_min_confidence_scale + (
            1.0 - self.residual_memory_min_confidence_scale
        ) * confidence
        level_memory = level_memory * confidence_scale.view(1, -1, 1).to(
            device=level_memory.device, dtype=level_memory.dtype
        )
        if user_adaptation and "level_memory_log_weights" in user_adaptation:
            log_weights = user_adaptation["level_memory_log_weights"]
            if log_weights.dim() == 1:
                log_weights = log_weights.unsqueeze(0)
            usable = min(level_memory.size(1), log_weights.size(1))
            if usable > 0:
                scale = torch.exp(torch.clamp(log_weights[:, :usable], min=-1.5, max=1.5)).unsqueeze(-1)
                level_memory = torch.cat(
                    [level_memory[:, :usable, :] * scale.to(dtype=level_memory.dtype), level_memory[:, usable:, :]],
                    dim=1,
                )
        return level_memory

    def _append_residual_level_memory(
        self,
        positive_memory: torch.Tensor,
        negative_memory: torch.Tensor,
        support_embeddings: torch.Tensor,
        memory_scores: torch.Tensor,
        user_adaptation: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        level_memory = self._residual_level_memory(
            support_embeddings=support_embeddings,
            memory_scores=memory_scores,
            user_adaptation=user_adaptation,
        )
        if level_memory is None:
            return positive_memory, negative_memory
        positive_levels = torch.flip(level_memory, dims=[1])
        negative_levels = level_memory
        return (
            torch.cat([positive_memory, positive_levels], dim=1),
            torch.cat([negative_memory, negative_levels], dim=1),
        )

    def _semantic_preference_memory(
        self,
        support_semantic_scores: torch.Tensor | None,
        memory_scores: torch.Tensor,
        user_adaptation: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not self.semantic_memory_enabled
            or support_semantic_scores is None
            or support_semantic_scores.numel() == 0
            or self.semantic_score_embed is None
            or self.semantic_memory_norm is None
            or self.semantic_memory_gate is None
        ):
            return None
        semantic_scores = support_semantic_scores.float()
        residual = memory_scores.detach().float().view(-1)
        if semantic_scores.size(0) != residual.size(0):
            return None
        semantic_z = semantic_scores - semantic_scores.mean(dim=0, keepdim=True)
        semantic_z = semantic_z / semantic_z.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        residual_z = residual - residual.mean()
        residual_z = residual_z / residual_z.std(unbiased=False).clamp_min(1e-6)
        preference = (semantic_z * residual_z.view(-1, 1)).mean(dim=0)
        k = min(self.semantic_memory_tokens, preference.numel())
        if k <= 0:
            return None
        positive_values, positive_indices = torch.topk(preference, k=k, largest=True)
        negative_values, negative_indices = torch.topk(preference, k=k, largest=False)
        text_features = self.semantic_text_features.to(device=semantic_scores.device, dtype=semantic_scores.dtype)
        positive_tokens = text_features.index_select(0, positive_indices)
        negative_tokens = text_features.index_select(0, negative_indices)
        positive_values = positive_values / self.semantic_memory_temperature
        negative_values = negative_values / self.semantic_memory_temperature
        positive_tokens = positive_tokens + self.semantic_score_embed(positive_values.to(positive_tokens.dtype).unsqueeze(-1))
        negative_tokens = negative_tokens + self.semantic_score_embed(negative_values.to(negative_tokens.dtype).unsqueeze(-1))
        gate = torch.tanh(self.semantic_memory_gate).clamp(min=0.0, max=1.0).to(dtype=positive_tokens.dtype)
        positive_tokens = self.semantic_memory_norm(positive_tokens * gate).unsqueeze(0)
        negative_tokens = self.semantic_memory_norm(negative_tokens * gate).unsqueeze(0)
        if user_adaptation and "semantic_memory_log_weights" in user_adaptation:
            log_weights = user_adaptation["semantic_memory_log_weights"]
            if log_weights.dim() == 1:
                log_weights = log_weights.unsqueeze(0)
            usable = min(k, log_weights.size(1))
            if usable > 0:
                weights = torch.exp(torch.clamp(log_weights[:, :usable], min=-1.5, max=1.5)).unsqueeze(-1)
                positive_tokens = torch.cat([positive_tokens[:, :usable] * weights, positive_tokens[:, usable:]], dim=1)
                negative_tokens = torch.cat([negative_tokens[:, :usable] * weights, negative_tokens[:, usable:]], dim=1)
        return positive_tokens, negative_tokens

    def _append_semantic_prompt_memory(
        self,
        positive_memory: torch.Tensor,
        negative_memory: torch.Tensor,
        support_semantic_scores: torch.Tensor | None,
        memory_scores: torch.Tensor,
        user_adaptation: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_memory = self._semantic_preference_memory(
            support_semantic_scores=support_semantic_scores,
            memory_scores=memory_scores,
            user_adaptation=user_adaptation,
        )
        if semantic_memory is None:
            return positive_memory, negative_memory
        semantic_positive, semantic_negative = semantic_memory
        return (
            torch.cat([positive_memory, semantic_positive], dim=1),
            torch.cat([negative_memory, semantic_negative], dim=1),
        )

    def _retrieve_support_residual(
        self,
        query_embeddings: torch.Tensor,
        support_embeddings: torch.Tensor,
        support_scores: torch.Tensor,
        support_base_score: torch.Tensor,
        user_adaptation: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.support_retrieval_gate is None or support_embeddings.numel() == 0:
            return query_embeddings.new_zeros(query_embeddings.size(0))
        support_residual = support_scores - support_base_score
        if self.support_retrieval_detach:
            support_residual = support_residual.detach()
        support_residual = self._apply_level_residual_adaptation(
            support_residual=support_residual,
            user_adaptation=user_adaptation,
        )
        query_norm = F.normalize(query_embeddings, dim=-1)
        support_norm = F.normalize(support_embeddings, dim=-1)
        if self.transductive_retrieval and self.transductive_steps > 0 and query_norm.size(0) > 1:
            all_embeddings = torch.cat([support_norm, query_norm], dim=0)
            logits = torch.matmul(all_embeddings, all_embeddings.transpose(0, 1)) / self.support_retrieval_temperature
            self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(self_mask, float("-inf"))
            if self.transductive_topk > 0 and self.transductive_topk < logits.size(1) - 1:
                topk_values, topk_indices = torch.topk(logits, k=self.transductive_topk, dim=1)
                sparse_logits = logits.new_full(logits.shape, float("-inf"))
                logits = sparse_logits.scatter(1, topk_indices, topk_values)
            weights = torch.softmax(logits, dim=-1)
            num_support = support_norm.size(0)
            if self.transductive_mode == "closed_form":
                query_to_support = weights[num_support:, :num_support]
                query_to_query = weights[num_support:, num_support:]
                identity = torch.eye(
                    query_to_query.size(0),
                    device=query_to_query.device,
                    dtype=query_to_query.dtype,
                )
                system = identity - self.transductive_alpha * query_to_query
                if self.transductive_ridge > 0.0:
                    system = system + self.transductive_ridge * identity
                rhs = self.transductive_alpha * torch.matmul(
                    query_to_support,
                    support_residual.view(-1, 1),
                )
                retrieved = self._linear_solve(system.float(), rhs.float()).to(dtype=query_embeddings.dtype).squeeze(-1)
            else:
                values = torch.cat([support_residual, query_norm.new_zeros(query_norm.size(0))], dim=0)
                confidence = torch.cat([query_norm.new_ones(support_norm.size(0)), query_norm.new_zeros(query_norm.size(0))], dim=0)
                for _ in range(self.transductive_steps):
                    values = torch.matmul(weights, values)
                    confidence = torch.matmul(weights, confidence)
                    values[: support_norm.size(0)] = support_residual
                    confidence[: support_norm.size(0)] = 1.0
                retrieved = values[support_norm.size(0) :] / confidence[support_norm.size(0) :].clamp_min(1e-6)
        else:
            logits = torch.matmul(query_norm, support_norm.transpose(0, 1)) / self.support_retrieval_temperature
            weights = torch.softmax(logits, dim=-1)
            retrieved = torch.matmul(weights, support_residual.view(-1, 1)).squeeze(-1)
        retrieval_gate = torch.tanh(self.support_retrieval_gate)
        if user_adaptation:
            log_scale = user_adaptation.get("retrieval_log_scale")
            if log_scale is not None:
                retrieval_gate = retrieval_gate * torch.exp(torch.clamp(log_scale, min=-1.5, max=1.5))
        return retrieval_gate.expand_as(retrieved) * retrieved

    def _linear_solve(self, system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if self.linear_solve_device == "cpu" and system.is_cuda:
            solution = torch.linalg.solve(system.cpu(), rhs.cpu())
            return solution.to(device=system.device)
        return torch.linalg.solve(system, rhs)

    def _kernel_support_residual(
        self,
        query_embeddings: torch.Tensor,
        support_embeddings: torch.Tensor,
        support_scores: torch.Tensor,
        support_base_score: torch.Tensor,
        user_adaptation: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if (
            not self.support_kernel_residual
            or self.support_kernel_residual_scale == 0.0
            or support_embeddings.numel() == 0
            or support_scores.numel() == 0
        ):
            return query_embeddings.new_zeros(query_embeddings.size(0))

        support_residual = support_scores - support_base_score
        support_residual = self._apply_level_residual_adaptation(
            support_residual=support_residual,
            user_adaptation=user_adaptation,
        )
        if self.support_kernel_residual_detach:
            query_embeddings = query_embeddings.detach()
            support_embeddings = support_embeddings.detach()
            support_residual = support_residual.detach()

        query_norm = F.normalize(query_embeddings.float(), dim=-1)
        support_norm = F.normalize(support_embeddings.float(), dim=-1)
        residual_values = support_residual.float().view(-1)
        residual_mean = residual_values.mean()
        centered_values = residual_values - residual_mean
        support_kernel = torch.exp(
            (torch.matmul(support_norm, support_norm.transpose(0, 1)) - 1.0)
            / self.support_kernel_residual_temperature
        )
        identity = torch.eye(support_kernel.size(0), dtype=support_kernel.dtype, device=support_kernel.device)
        alpha = self._linear_solve(
            support_kernel + self.support_kernel_residual_ridge * identity,
            centered_values.view(-1, 1),
        ).squeeze(-1)
        query_kernel = torch.exp(
            (torch.matmul(query_norm, support_norm.transpose(0, 1)) - 1.0)
            / self.support_kernel_residual_temperature
        )
        residual = torch.matmul(query_kernel, alpha.view(-1, 1)).squeeze(-1) + residual_mean
        support_count = float(max(int(support_scores.numel()), 1))
        shrinkage = (
            support_count / (support_count + self.support_kernel_residual_shrinkage)
            if self.support_kernel_residual_shrinkage > 0.0
            else 1.0
        )
        residual = (self.support_kernel_residual_scale * shrinkage) * residual
        if self.support_kernel_residual_max_abs is not None:
            max_abs = float(self.support_kernel_residual_max_abs)
            if max_abs > 0.0:
                residual = torch.clamp(residual, min=-max_abs, max=max_abs)
        return residual.to(dtype=query_embeddings.dtype)

    def _estimate_user_state_residual(
        self,
        query_embeddings: torch.Tensor,
        support_embeddings: torch.Tensor,
        support_scores: torch.Tensor,
        support_base_score: torch.Tensor,
    ) -> torch.Tensor:
        if self.user_state_projection is None or support_embeddings.numel() == 0:
            return query_embeddings.new_zeros(query_embeddings.size(0))
        support_residual = support_scores - support_base_score.detach()
        if self.user_state_detach_support_residual:
            support_residual = support_residual.detach()
        device_type = "cuda" if query_embeddings.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            support_basis = self.user_state_projection(support_embeddings.float()).float()
            query_basis = self.user_state_projection(query_embeddings.float()).float()
            identity = torch.eye(
                support_basis.size(1),
                device=support_basis.device,
                dtype=support_basis.dtype,
            )
            normal = support_basis.transpose(0, 1) @ support_basis
            rhs = support_basis.transpose(0, 1) @ support_residual.float().view(-1, 1)
            coeff = self._linear_solve(normal + self.user_state_ridge * identity, rhs).view(-1)
            residual = query_basis @ coeff
        residual = residual.to(dtype=query_embeddings.dtype)
        if self.user_state_gate is not None:
            residual = residual * self.user_state_gate.to(dtype=query_embeddings.dtype)
        return residual

    def _fit_bridge_residual_scale(
        self,
        support_bridge_residual: torch.Tensor,
        support_scores: torch.Tensor,
        support_base_score: torch.Tensor,
    ) -> torch.Tensor:
        if not self.bridge_support_fit or support_bridge_residual.numel() < 2:
            return support_bridge_residual.new_tensor(1.0)
        bridge = support_bridge_residual.view(-1)
        target = (support_scores - support_base_score).view(-1)
        if self.bridge_support_fit_detach:
            bridge = bridge.detach()
            target = target.detach()
        bridge = bridge - bridge.mean()
        target = target - target.mean()
        bridge_var = bridge.pow(2).mean()
        if not torch.isfinite(bridge_var) or float(bridge_var.detach().cpu()) <= 1e-8:
            return support_bridge_residual.new_tensor(0.0)
        fitted = (bridge * target).mean() / (bridge_var + bridge.new_tensor(self.bridge_support_fit_ridge))
        if self.bridge_support_fit_shrinkage > 0.0:
            count = bridge.new_tensor(float(bridge.numel()))
            fitted = fitted * (count / (count + bridge.new_tensor(self.bridge_support_fit_shrinkage)))
        max_abs = float(self.bridge_support_fit_max_abs_scale)
        if max_abs > 0.0:
            fitted = fitted.clamp(min=-max_abs, max=max_abs)
        return fitted.to(dtype=support_bridge_residual.dtype, device=support_bridge_residual.device)

    def _fit_bridge_basis_residual(
        self,
        query_token_residuals: torch.Tensor,
        support_token_residuals: torch.Tensor,
        support_scores: torch.Tensor,
        support_base_score: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.bridge_basis_fit
            or self.bridge_basis_fit_scale == 0.0
            or support_token_residuals.numel() == 0
            or support_scores.numel() < 2
        ):
            return query_token_residuals.new_zeros(query_token_residuals.size(0))

        support_basis = support_token_residuals
        residual_target = support_scores - support_base_score
        if self.bridge_basis_fit_detach:
            support_basis = support_basis.detach()
            residual_target = residual_target.detach()

        device_type = "cuda" if query_token_residuals.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            support_basis_f = support_basis.float()
            query_basis_f = query_token_residuals.float()
            target_f = residual_target.float().view(-1)
            basis_mean = support_basis_f.mean(dim=0, keepdim=True)
            target_mean = target_f.mean()
            centered_support = support_basis_f - basis_mean
            centered_query = query_basis_f - basis_mean
            centered_target = target_f - target_mean
            dim = centered_support.size(1)
            identity = torch.eye(dim, device=centered_support.device, dtype=centered_support.dtype)
            normal = centered_support.transpose(0, 1) @ centered_support
            normal = normal / max(int(centered_support.size(0)), 1)
            rhs = centered_support.transpose(0, 1) @ centered_target.view(-1, 1)
            rhs = rhs / max(int(centered_support.size(0)), 1)
            coeff = self._linear_solve(normal + self.bridge_basis_fit_ridge * identity, rhs).view(-1)
            residual = centered_query @ coeff + target_mean

        count = float(max(int(support_scores.numel()), 1))
        shrinkage = (
            count / (count + self.bridge_basis_fit_shrinkage)
            if self.bridge_basis_fit_shrinkage > 0.0
            else 1.0
        )
        residual = residual.to(dtype=query_token_residuals.dtype)
        residual = residual * (self.bridge_basis_fit_scale * shrinkage)
        if self.bridge_basis_fit_max_abs is not None:
            max_abs = float(self.bridge_basis_fit_max_abs)
            if max_abs > 0.0:
                residual = torch.clamp(residual, min=-max_abs, max=max_abs)
        return residual

    @staticmethod
    def _residual_rank_loss(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if values.numel() < 2:
            return values.new_zeros(())
        prediction_diff = values.view(-1, 1) - values.view(1, -1)
        target_diff = target.view(-1, 1) - target.view(1, -1)
        valid = target_diff.abs() > 1e-6
        if not bool(valid.any().detach().cpu()):
            return values.new_zeros(())
        signed = torch.sign(target_diff[valid]) * prediction_diff[valid]
        return F.softplus(-signed).mean()

    def _fit_support_residual_fusion(
        self,
        query_components: dict[str, torch.Tensor],
        support_components: dict[str, torch.Tensor],
        support_scores: torch.Tensor,
        support_base_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not self.support_residual_fusion
            or support_scores.numel() < self.support_residual_fusion_min_support
            or not self.support_residual_fusion_components
        ):
            return None

        query_columns: list[torch.Tensor] = []
        support_columns: list[torch.Tensor] = []
        for component in self.support_residual_fusion_components:
            query_value = query_components.get(component)
            support_value = support_components.get(component)
            if query_value is None or support_value is None:
                continue
            if support_value.numel() != support_scores.numel() or query_value.dim() != 1:
                continue
            query_columns.append(query_value.view(-1))
            support_columns.append(support_value.view(-1))
        if not query_columns:
            return None

        query_default = torch.stack(query_columns, dim=1).sum(dim=1)
        support_default = torch.stack(support_columns, dim=1).sum(dim=1)
        target = support_scores - support_base_score
        if self.support_residual_fusion_detach:
            # Fit user-specific fusion weights from detached support labels/features,
            # but keep query-side residual components differentiable. Otherwise the
            # 100-shot query loss cannot train the bridge/retrieval estimators when
            # fusion is active.
            support_columns_for_fit = [column.detach() for column in support_columns]
            target_for_fit = target.detach()
        else:
            support_columns_for_fit = support_columns
            target_for_fit = target

        device_type = "cuda" if support_scores.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            query_matrix = torch.stack(query_columns, dim=1).float()
            support_matrix = torch.stack(support_columns_for_fit, dim=1).float()
            target_f = target_for_fit.float().view(-1)
            fit_indices = torch.arange(target_f.numel(), device=target_f.device)
            val_indices = None
            val_count = int(self.support_residual_fusion_validation_images)
            if val_count > 1 and target_f.numel() > val_count + support_matrix.size(1) + 1:
                count = min(val_count, target_f.numel() - support_matrix.size(1) - 1)
                order = torch.argsort(support_scores.detach().float().view(-1))
                positions = torch.linspace(
                    0,
                    target_f.numel() - 1,
                    steps=count,
                    device=target_f.device,
                ).round().long().unique()
                val_indices = order.index_select(0, positions[:count])
                keep = torch.ones(target_f.numel(), dtype=torch.bool, device=target_f.device)
                keep.index_fill_(0, val_indices, False)
                fit_indices = torch.nonzero(keep, as_tuple=False).view(-1)
                if fit_indices.numel() <= support_matrix.size(1) or val_indices.numel() < 2:
                    fit_indices = torch.arange(target_f.numel(), device=target_f.device)
                    val_indices = None

            fit_matrix = support_matrix.index_select(0, fit_indices)
            fit_target = target_f.index_select(0, fit_indices)
            matrix_mean = fit_matrix.mean(dim=0, keepdim=True)
            target_mean = fit_target.mean()
            centered_matrix = fit_matrix - matrix_mean
            centered_target = fit_target - target_mean
            dim = centered_matrix.size(1)
            identity = torch.eye(dim, device=centered_matrix.device, dtype=centered_matrix.dtype)
            normal = centered_matrix.transpose(0, 1) @ centered_matrix
            normal = normal / max(int(centered_matrix.size(0)), 1)
            rhs = centered_matrix.transpose(0, 1) @ centered_target.view(-1, 1)
            rhs = rhs / max(int(centered_matrix.size(0)), 1)
            try:
                coeff = self._linear_solve(
                    normal + self.support_residual_fusion_ridge * identity,
                    rhs,
                ).view(-1)
            except RuntimeError:
                return None

            max_weight = float(self.support_residual_fusion_max_abs_weight)
            if max_weight > 0.0:
                coeff = coeff.clamp(min=-max_weight, max=max_weight)
            intercept = target_mean - (matrix_mean.view(-1) * coeff).sum()
            query_fitted = query_matrix @ coeff + intercept
            support_fitted = support_matrix @ coeff + intercept
            if val_indices is not None:
                val_fitted = support_fitted.index_select(0, val_indices)
                val_default = support_default.detach().float().index_select(0, val_indices)
                metric = self.support_residual_fusion_validation_metric
                if metric == "score_rank":
                    val_target = support_scores.detach().float().view(-1).index_select(0, val_indices)
                    val_base = support_base_score.detach().float().view(-1).index_select(0, val_indices)
                    fitted_loss = self._residual_rank_loss(val_base + val_fitted, val_target)
                    default_loss = self._residual_rank_loss(val_base + val_default, val_target)
                elif metric == "both":
                    val_residual_target = target_f.index_select(0, val_indices)
                    val_score_target = support_scores.detach().float().view(-1).index_select(0, val_indices)
                    val_base = support_base_score.detach().float().view(-1).index_select(0, val_indices)
                    fitted_loss = 0.5 * (
                        self._residual_rank_loss(val_fitted, val_residual_target)
                        + self._residual_rank_loss(val_base + val_fitted, val_score_target)
                    )
                    default_loss = 0.5 * (
                        self._residual_rank_loss(val_default, val_residual_target)
                        + self._residual_rank_loss(val_base + val_default, val_score_target)
                    )
                else:
                    val_target = target_f.index_select(0, val_indices)
                    fitted_loss = self._residual_rank_loss(val_fitted, val_target)
                    default_loss = self._residual_rank_loss(val_default, val_target)
                required = default_loss - self.support_residual_fusion_validation_margin
                if float(fitted_loss.detach().cpu()) >= float(required.detach().cpu()):
                    return None

            count = float(max(int(fit_indices.numel()), 1))
            blend = (
                count / (count + self.support_residual_fusion_shrinkage)
                if self.support_residual_fusion_shrinkage > 0.0
                else 1.0
            )
            query_fused = query_default.float() + blend * (query_fitted - query_default.float())
            support_fused = support_default.float() + blend * (support_fitted - support_default.float())

        return (
            query_fused.to(dtype=query_default.dtype),
            support_fused.to(dtype=support_default.dtype),
        )

    def _fit_support_residual_score_scale(
        self,
        query_residual: torch.Tensor,
        support_residual: torch.Tensor,
        support_scores: torch.Tensor,
        support_base_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            not self.support_residual_score_fit
            or support_scores.numel() < self.support_residual_score_fit_min_support
            or support_residual.numel() != support_scores.numel()
        ):
            scale = query_residual.new_ones(())
            return query_residual, support_residual, scale

        target = support_scores - support_base_score
        fit_residual = support_residual.detach() if self.support_residual_score_fit_detach else support_residual
        fit_target = target.detach() if self.support_residual_score_fit_detach else target
        device_type = "cuda" if support_scores.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            support_r = fit_residual.float().view(-1)
            target_r = fit_target.float().view(-1)
            query_r = query_residual.float()
            support_apply_r = support_residual.float()
            fit_indices = torch.arange(target_r.numel(), device=target_r.device)
            val_indices = None
            val_count = int(self.support_residual_score_fit_validation_images)
            if val_count > 1 and target_r.numel() > val_count + 2:
                count = min(val_count, target_r.numel() - 2)
                order = torch.argsort(support_scores.detach().float().view(-1))
                positions = torch.linspace(
                    0,
                    target_r.numel() - 1,
                    steps=count,
                    device=target_r.device,
                ).round().long().unique()
                val_indices = order.index_select(0, positions[:count])
                keep = torch.ones(target_r.numel(), dtype=torch.bool, device=target_r.device)
                keep.index_fill_(0, val_indices, False)
                fit_indices = torch.nonzero(keep, as_tuple=False).view(-1)
                if fit_indices.numel() < 2 or val_indices.numel() < 2:
                    fit_indices = torch.arange(target_r.numel(), device=target_r.device)
                    val_indices = None

            fit_r = support_r.index_select(0, fit_indices)
            fit_t = target_r.index_select(0, fit_indices)
            denominator = fit_r.pow(2).mean() + self.support_residual_score_fit_ridge
            fitted_scale = (fit_r * fit_t).mean() / denominator
            max_scale = float(self.support_residual_score_fit_max_abs_scale)
            if max_scale > 0.0:
                fitted_scale = fitted_scale.clamp(min=-max_scale, max=max_scale)

            if val_indices is not None:
                val_base = support_base_score.detach().float().view(-1).index_select(0, val_indices)
                val_scores = support_scores.detach().float().view(-1).index_select(0, val_indices)
                val_residual = support_r.index_select(0, val_indices)
                if self.support_residual_score_fit_candidate_scales:
                    candidates = [
                        val_residual.new_tensor(float(scale))
                        for scale in self.support_residual_score_fit_candidate_scales
                    ]
                    candidates.append(fitted_scale)
                    fitted_scale = min(
                        candidates,
                        key=lambda candidate: float(
                            self._residual_rank_loss(val_base + candidate * val_residual, val_scores)
                            .detach()
                            .cpu()
                        ),
                    )
                fitted_loss = self._residual_rank_loss(val_base + fitted_scale * val_residual, val_scores)
                default_loss = self._residual_rank_loss(val_base + val_residual, val_scores)
                required = default_loss - self.support_residual_score_fit_validation_margin
                if float(fitted_loss.detach().cpu()) >= float(required.detach().cpu()):
                    scale = query_r.new_ones(())
                    return query_residual, support_residual, scale.to(dtype=query_residual.dtype)

            count = float(max(int(fit_indices.numel()), 1))
            blend = (
                count / (count + self.support_residual_score_fit_shrinkage)
                if self.support_residual_score_fit_shrinkage > 0.0
                else 1.0
            )
            scale = 1.0 + blend * (fitted_scale - 1.0)
            query_scaled = query_r * scale
            support_scaled = support_apply_r * scale

        return (
            query_scaled.to(dtype=query_residual.dtype),
            support_scaled.to(dtype=support_residual.dtype),
            scale.to(dtype=query_residual.dtype),
        )

    def _condition_with_memory(
        self,
        attribute_tokens: torch.Tensor,
        generic_score: torch.Tensor,
        positive_memory: torch.Tensor,
        negative_memory: torch.Tensor,
        user_adaptation: dict[str, torch.Tensor] | None = None,
        query_positive_memory: torch.Tensor | None = None,
        query_negative_memory: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        user_positive = user_adaptation.get("positive_tokens") if user_adaptation else None
        user_negative = user_adaptation.get("negative_tokens") if user_adaptation else None
        positive_memory = self._apply_user_memory_weights(
            positive_memory,
            user_adaptation,
            "positive_memory_log_weights",
        )
        negative_memory = self._apply_user_memory_weights(
            negative_memory,
            user_adaptation,
            "negative_memory_log_weights",
        )
        expanded_positive = positive_memory.expand(attribute_tokens.size(0), -1, -1)
        expanded_negative = negative_memory.expand(attribute_tokens.size(0), -1, -1)
        if query_positive_memory is not None:
            expanded_positive = torch.cat([expanded_positive, query_positive_memory], dim=1)
        if query_negative_memory is not None:
            expanded_negative = torch.cat([expanded_negative, query_negative_memory], dim=1)
        expanded_positive = self._append_user_memory(expanded_positive, user_positive)
        expanded_negative = self._append_user_memory(expanded_negative, user_negative)
        if self.cross_attention_bridge:
            bridge_outputs = self.bridge(
                positive_memory=expanded_positive,
                negative_memory=expanded_negative,
                attribute_tokens=attribute_tokens,
            )
            conditioned_tokens = bridge_outputs["conditioned_tokens"]
        else:
            conditioned_tokens = attribute_tokens
        if self.stage2_residual_prior:
            residual_outputs = self.residual_head(conditioned_tokens)
            raw_residual_score = residual_outputs["score"]
            token_weights = residual_outputs["token_weights"]
            token_residuals = residual_outputs["token_residuals"]
            pooled_tokens = residual_outputs["pooled_tokens"]
        else:
            batch_size, token_count = conditioned_tokens.shape[:2]
            token_weights = conditioned_tokens.new_full(
                (batch_size, token_count),
                1.0 / max(int(token_count), 1),
            )
            token_residuals = conditioned_tokens.new_zeros((batch_size, token_count))
            raw_residual_score = conditioned_tokens.new_zeros(batch_size)
            pooled_tokens = torch.sum(token_weights.unsqueeze(-1) * conditioned_tokens, dim=1)
        scaled_residual_score = self._scale_residual(raw_residual_score)
        return {
            "score": generic_score + scaled_residual_score,
            "residual_score": scaled_residual_score,
            "raw_residual_score": raw_residual_score,
            "conditioned_tokens": conditioned_tokens,
            "token_weights": token_weights,
            "token_residuals": token_residuals,
            "pooled_tokens": pooled_tokens,
        }

    def _support_stats(
        self,
        support_scores: torch.Tensor,
        support_base_score: torch.Tensor,
        support_stage1_score: torch.Tensor,
        support_prior_scores: torch.Tensor | None,
    ) -> torch.Tensor:
        support_scores = support_scores.detach()
        base_errors = support_scores - support_base_score.detach()
        if support_prior_scores is None:
            stage1_delta = torch.zeros_like(support_stage1_score.detach())
        else:
            stage1_delta = support_stage1_score.detach() - support_prior_scores.detach()
        values = [
            support_scores.mean(),
            support_scores.std(unbiased=False),
            base_errors.mean(),
            base_errors.std(unbiased=False),
            stage1_delta.mean(),
            stage1_delta.std(unbiased=False),
        ]
        return torch.stack(values).unsqueeze(0)

    def _calibration_from_memory(
        self,
        positive_memory: torch.Tensor,
        negative_memory: torch.Tensor,
        support_scores: torch.Tensor,
        support_base_score: torch.Tensor,
        support_stage1_score: torch.Tensor,
        support_prior_scores: torch.Tensor | None,
    ) -> dict[str, torch.Tensor] | None:
        if self.calibration_head is None:
            return None
        support_stats = self._support_stats(
            support_scores=support_scores,
            support_base_score=support_base_score,
            support_stage1_score=support_stage1_score,
            support_prior_scores=support_prior_scores,
        )
        return self.calibration_head(positive_memory, negative_memory, support_stats)

    def _apply_calibration(
        self,
        base_score: torch.Tensor,
        stage1_score: torch.Tensor,
        prior_scores: torch.Tensor | None,
        residual_score: torch.Tensor,
        calibration: dict[str, torch.Tensor] | None,
        user_adaptation: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if prior_scores is None:
            stage1_delta = torch.zeros_like(stage1_score)
        else:
            stage1_delta = stage1_score.detach() - prior_scores
        if calibration is None:
            stage1_adjust = torch.zeros_like(base_score)
            residual_gate = torch.ones_like(base_score)
            bias = torch.zeros_like(base_score)
        else:
            stage1_adjust = calibration["stage1_delta_adjust"].expand_as(base_score)
            residual_gate = calibration["residual_gate"].expand_as(base_score)
            bias = calibration["bias"].expand_as(base_score)
        if user_adaptation:
            user_stage1_adjust = user_adaptation.get("stage1_delta_adjust")
            if user_stage1_adjust is not None:
                user_stage1_adjust = torch.clamp(user_stage1_adjust, min=-0.75, max=0.75)
                stage1_adjust = stage1_adjust + user_stage1_adjust.expand_as(base_score)
        return base_score + stage1_adjust * stage1_delta + residual_gate * residual_score + bias

    def _support_leave_one_out(
        self,
        support_attribute_tokens: torch.Tensor,
        support_image_embeddings: torch.Tensor,
        support_scores: torch.Tensor,
        support_generic_score: torch.Tensor,
        support_stage1_score: torch.Tensor,
        support_prior_scores: torch.Tensor | None,
        user_adaptation: dict[str, torch.Tensor] | None = None,
        support_semantic_scores: torch.Tensor | None = None,
        support_loo_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        num_support = support_scores.size(0)
        if num_support <= 1:
            return {
                "score": support_generic_score,
                "raw_residual_score": torch.zeros_like(support_generic_score),
            }
        scores = []
        raw_residuals = []
        if support_loo_indices is None:
            loo_indices = torch.arange(num_support, device=support_scores.device)
        else:
            loo_indices = support_loo_indices.to(device=support_scores.device, dtype=torch.long).view(-1)
        for support_index_tensor in loo_indices:
            support_index = int(support_index_tensor.detach().cpu().item())
            keep_mask = torch.ones(num_support, dtype=torch.bool, device=support_scores.device)
            keep_mask[support_index] = False
            memory_scores = self._preference_memory_scores(
                support_scores=support_scores[keep_mask],
                support_base_score=support_generic_score[keep_mask],
            )
            preference_memory_scores = memory_scores
            memory_embeddings, memory_scores = self._preference_memory_inputs(
                support_attribute_tokens=support_attribute_tokens[keep_mask],
                support_image_embeddings=support_image_embeddings[keep_mask],
                memory_scores=memory_scores,
            )
            preference_outputs = self.preference_encoder(memory_embeddings, memory_scores)
            positive_memory, negative_memory = self._append_direct_support_memory(
                positive_memory=preference_outputs["positive_memory"],
                negative_memory=preference_outputs["negative_memory"],
                memory_embeddings=memory_embeddings,
                memory_scores=memory_scores,
                max_tokens=(
                    self.direct_support_memory_loo_tokens
                    if self.direct_support_memory_loo_tokens > 0
                    else self.direct_support_memory_tokens
                ),
            )
            positive_memory, negative_memory = self._append_residual_level_memory(
                positive_memory=positive_memory,
                negative_memory=negative_memory,
                support_embeddings=support_image_embeddings[keep_mask],
                memory_scores=preference_memory_scores,
                user_adaptation=user_adaptation,
            )
            positive_memory, negative_memory = self._append_semantic_prompt_memory(
                positive_memory=positive_memory,
                negative_memory=negative_memory,
                support_semantic_scores=(support_semantic_scores[keep_mask] if support_semantic_scores is not None else None),
                memory_scores=preference_memory_scores,
                user_adaptation=user_adaptation,
            )
            piaa_positive_memory, piaa_negative_memory = self._piaa_token_fusion_memory(
                support_attribute_tokens=support_attribute_tokens[keep_mask],
                support_scores=support_scores[keep_mask],
                support_base_score=support_generic_score[keep_mask],
            )
            positive_memory, negative_memory = self._append_piaa_token_fusion_memory(
                positive_memory=positive_memory,
                negative_memory=negative_memory,
                piaa_positive_memory=piaa_positive_memory,
                piaa_negative_memory=piaa_negative_memory,
            )
            keep_prior_scores = support_prior_scores[keep_mask] if support_prior_scores is not None else None
            calibration = self._calibration_from_memory(
                positive_memory=positive_memory,
                negative_memory=negative_memory,
                support_scores=support_scores[keep_mask],
                support_base_score=support_generic_score[keep_mask],
                support_stage1_score=support_stage1_score[keep_mask],
                support_prior_scores=keep_prior_scores,
            )
            query_positive_memory, query_negative_memory = self._query_direct_support_memory(
                query_attribute_tokens=support_attribute_tokens[support_index : support_index + 1],
                support_attribute_tokens=support_attribute_tokens[keep_mask],
                support_image_embeddings=support_image_embeddings[keep_mask],
                memory_scores=preference_memory_scores,
            )
            conditioned = self._condition_with_memory(
                attribute_tokens=support_attribute_tokens[support_index : support_index + 1],
                generic_score=support_generic_score[support_index : support_index + 1],
                positive_memory=positive_memory,
                negative_memory=negative_memory,
                user_adaptation=user_adaptation,
                query_positive_memory=query_positive_memory,
                query_negative_memory=query_negative_memory,
            )
            use_attribute_similarity = self.support_residual_similarity_source in {
                "attribute_pooled",
                "attribute",
                "contrast_direct",
                "direct",
                "raw",
            }
            if use_attribute_similarity:
                heldout_similarity_embeddings = support_image_embeddings[support_index : support_index + 1]
                loo_support_similarity_embeddings = support_image_embeddings[keep_mask]
                keep_conditioned = None
            else:
                keep_positive_memory, keep_negative_memory = self._query_direct_support_memory(
                    query_attribute_tokens=support_attribute_tokens[keep_mask],
                    support_attribute_tokens=support_attribute_tokens[keep_mask],
                    support_image_embeddings=support_image_embeddings[keep_mask],
                    memory_scores=preference_memory_scores,
                )
                keep_conditioned = self._condition_with_memory(
                    attribute_tokens=support_attribute_tokens[keep_mask],
                    generic_score=support_generic_score[keep_mask],
                    positive_memory=positive_memory,
                    negative_memory=negative_memory,
                    user_adaptation=user_adaptation,
                    query_positive_memory=keep_positive_memory,
                    query_negative_memory=keep_negative_memory,
                )
                heldout_similarity_embeddings = conditioned["pooled_tokens"]
                loo_support_similarity_embeddings = keep_conditioned["pooled_tokens"]
            if keep_conditioned is None and (self.bridge_support_fit or self.bridge_basis_fit):
                keep_positive_memory, keep_negative_memory = self._query_direct_support_memory(
                    query_attribute_tokens=support_attribute_tokens[keep_mask],
                    support_attribute_tokens=support_attribute_tokens[keep_mask],
                    support_image_embeddings=support_image_embeddings[keep_mask],
                    memory_scores=preference_memory_scores,
                )
                keep_conditioned = self._condition_with_memory(
                    attribute_tokens=support_attribute_tokens[keep_mask],
                    generic_score=support_generic_score[keep_mask],
                    positive_memory=positive_memory,
                    negative_memory=negative_memory,
                    user_adaptation=user_adaptation,
                    query_positive_memory=keep_positive_memory,
                    query_negative_memory=keep_negative_memory,
                )
            retrieved_residual = self._retrieve_support_residual(
                query_embeddings=heldout_similarity_embeddings,
                support_embeddings=loo_support_similarity_embeddings,
                support_scores=support_scores[keep_mask],
                support_base_score=support_generic_score[keep_mask],
                user_adaptation=user_adaptation,
            )
            kernel_residual = self._kernel_support_residual(
                query_embeddings=heldout_similarity_embeddings,
                support_embeddings=loo_support_similarity_embeddings,
                support_scores=support_scores[keep_mask],
                support_base_score=support_generic_score[keep_mask],
                user_adaptation=user_adaptation,
            )
            user_state_residual = self._estimate_user_state_residual(
                query_embeddings=heldout_similarity_embeddings,
                support_embeddings=loo_support_similarity_embeddings,
                support_scores=support_scores[keep_mask],
                support_base_score=support_generic_score[keep_mask],
            )
            heldout_prior_scores = support_prior_scores[support_index : support_index + 1] if support_prior_scores is not None else None
            bridge_residual = conditioned["residual_score"] * float(self.bridge_residual_scale)
            if keep_conditioned is not None:
                keep_bridge_residual = keep_conditioned["residual_score"] * float(self.bridge_residual_scale)
                bridge_fit_scale = self._fit_bridge_residual_scale(
                    support_bridge_residual=keep_bridge_residual,
                    support_scores=support_scores[keep_mask],
                    support_base_score=support_generic_score[keep_mask],
                )
                bridge_residual = bridge_residual * bridge_fit_scale
            if keep_conditioned is not None:
                bridge_basis_residual = self._fit_bridge_basis_residual(
                    query_token_residuals=conditioned["token_residuals"],
                    support_token_residuals=keep_conditioned["token_residuals"],
                    support_scores=support_scores[keep_mask],
                    support_base_score=support_generic_score[keep_mask],
                )
                bridge_residual = bridge_residual + bridge_basis_residual
            calibrated_score = self._apply_calibration(
                base_score=support_generic_score[support_index : support_index + 1],
                stage1_score=support_stage1_score[support_index : support_index + 1],
                prior_scores=heldout_prior_scores,
                residual_score=self._apply_user_residual_adaptation(
                    bridge_residual + retrieved_residual + kernel_residual + user_state_residual,
                    user_adaptation,
                ),
                calibration=calibration,
                user_adaptation=user_adaptation,
            )
            scores.append(calibrated_score)
            raw_residuals.append(conditioned["raw_residual_score"])
        return {
            "score": torch.cat(scores, dim=0),
            "raw_residual_score": torch.cat(raw_residuals, dim=0),
            "indices": loo_indices,
        }

    def freeze_stage1(
        self,
        freeze_backbone: bool = True,
        freeze_adapter: bool = True,
        freeze_attribute_extractor: bool = True,
        freeze_general_head: bool = True,
        freeze_attribute_head: bool = True,
    ) -> None:
        modules = [self.stage1]
        if self.contrast_stage1 is not None:
            modules.append(self.contrast_stage1)
        for stage1_module in modules:
            for parameter in stage1_module.backbone.backbone.parameters():
                parameter.requires_grad = not freeze_backbone
            for parameter in stage1_module.backbone.adapter.parameters():
                parameter.requires_grad = not freeze_adapter
            for parameter in stage1_module.backbone.output_norm.parameters():
                parameter.requires_grad = not freeze_adapter
            for parameter in stage1_module.attribute_extractor.parameters():
                parameter.requires_grad = not freeze_attribute_extractor
            for parameter in stage1_module.general_head.parameters():
                parameter.requires_grad = not freeze_general_head
            for parameter in stage1_module.attribute_head.parameters():
                parameter.requires_grad = not freeze_attribute_head
            for parameter in stage1_module.projection_head.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        support_images: torch.Tensor | dict[str, torch.Tensor],
        support_scores: torch.Tensor,
        query_images: torch.Tensor | dict[str, torch.Tensor],
        support_prior_scores: torch.Tensor | None = None,
        query_prior_scores: torch.Tensor | None = None,
        user_adaptation: dict[str, torch.Tensor] | None = None,
        compute_support_loo: bool = True,
        support_loo_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        support_features = self._encode_stage1_chunked(self.stage1, support_images)
        support_stage1_score = self.stage1.general_head(support_features["pooled_output"])
        support_semantic_scores = self._semantic_prompt_scores(support_features["pooled_output"])
        support_base_score = self._base_score(support_stage1_score, support_prior_scores)

        query_stage1 = self.stage1(query_images)
        query_base_score = self._base_score(query_stage1["score"], query_prior_scores)

        if self.contrast_stage1 is None:
            support_attribute_outputs = self.stage1.attribute_extractor(support_features["patch_tokens"])
            support_attribute_tokens = support_attribute_outputs["attribute_tokens"]
            query_attribute_tokens = query_stage1["attribute_tokens"]
            attribute_head = self.stage1.attribute_head
        else:
            support_contrast_features = self._encode_stage1_chunked(self.contrast_stage1, support_images)
            support_attribute_outputs = self.contrast_stage1.attribute_extractor(support_contrast_features["patch_tokens"])
            support_attribute_tokens = support_attribute_outputs["attribute_tokens"]
            query_contrast = self.contrast_stage1(query_images)
            query_attribute_tokens = query_contrast["attribute_tokens"]
            attribute_head = self.contrast_stage1.attribute_head
        support_attribute_tokens = self._adapt_attribute_tokens(support_attribute_tokens)
        query_attribute_tokens = self._adapt_attribute_tokens(query_attribute_tokens)
        support_attribute_summary = attribute_head(support_attribute_tokens)
        support_image_embeddings = support_attribute_summary["pooled_tokens"]
        query_attribute_summary = attribute_head(query_attribute_tokens)
        query_image_embeddings = query_attribute_summary["pooled_tokens"]
        preference_memory_scores = self._preference_memory_scores(
            support_scores=support_scores,
            support_base_score=support_base_score,
        )
        memory_embeddings, memory_scores = self._preference_memory_inputs(
            support_attribute_tokens=support_attribute_tokens,
            support_image_embeddings=support_image_embeddings,
            memory_scores=preference_memory_scores,
        )
        preference_outputs = self.preference_encoder(memory_embeddings, memory_scores)
        positive_memory, negative_memory = self._append_direct_support_memory(
            positive_memory=preference_outputs["positive_memory"],
            negative_memory=preference_outputs["negative_memory"],
            memory_embeddings=memory_embeddings,
            memory_scores=memory_scores,
        )
        positive_memory, negative_memory = self._append_residual_level_memory(
            positive_memory=positive_memory,
            negative_memory=negative_memory,
            support_embeddings=support_image_embeddings,
            memory_scores=preference_memory_scores,
            user_adaptation=user_adaptation,
        )
        positive_memory, negative_memory = self._append_semantic_prompt_memory(
            positive_memory=positive_memory,
            negative_memory=negative_memory,
            support_semantic_scores=support_semantic_scores,
            memory_scores=preference_memory_scores,
            user_adaptation=user_adaptation,
        )
        piaa_positive_memory, piaa_negative_memory = self._piaa_token_fusion_memory(
            support_attribute_tokens=support_attribute_tokens,
            support_scores=support_scores,
            support_base_score=support_base_score,
        )
        positive_memory, negative_memory = self._append_piaa_token_fusion_memory(
            positive_memory=positive_memory,
            negative_memory=negative_memory,
            piaa_positive_memory=piaa_positive_memory,
            piaa_negative_memory=piaa_negative_memory,
        )
        calibration = self._calibration_from_memory(
            positive_memory=positive_memory,
            negative_memory=negative_memory,
            support_scores=support_scores,
            support_base_score=support_base_score,
            support_stage1_score=support_stage1_score,
            support_prior_scores=support_prior_scores,
        )

        query_positive_memory, query_negative_memory = self._query_direct_support_memory(
            query_attribute_tokens=query_attribute_tokens,
            support_attribute_tokens=support_attribute_tokens,
            support_image_embeddings=support_image_embeddings,
            memory_scores=preference_memory_scores,
        )
        support_positive_memory, support_negative_memory = self._query_direct_support_memory(
            query_attribute_tokens=support_attribute_tokens,
            support_attribute_tokens=support_attribute_tokens,
            support_image_embeddings=support_image_embeddings,
            memory_scores=preference_memory_scores,
        )
        query_conditioned = self._condition_with_memory(
            attribute_tokens=query_attribute_tokens,
            generic_score=query_base_score,
            positive_memory=positive_memory,
            negative_memory=negative_memory,
            user_adaptation=user_adaptation,
            query_positive_memory=query_positive_memory,
            query_negative_memory=query_negative_memory,
        )
        support_conditioned = self._condition_with_memory(
            attribute_tokens=support_attribute_tokens,
            generic_score=support_base_score,
            positive_memory=positive_memory,
            negative_memory=negative_memory,
            user_adaptation=user_adaptation,
            query_positive_memory=support_positive_memory,
            query_negative_memory=support_negative_memory,
        )
        use_attribute_similarity = self.support_residual_similarity_source in {
            "attribute_pooled",
            "attribute",
            "contrast_direct",
            "direct",
            "raw",
        }
        query_similarity_embeddings = query_image_embeddings if use_attribute_similarity else query_conditioned["pooled_tokens"]
        support_similarity_embeddings = support_image_embeddings if use_attribute_similarity else support_conditioned["pooled_tokens"]
        query_retrieved_residual = self._retrieve_support_residual(
            query_embeddings=query_similarity_embeddings,
            support_embeddings=support_similarity_embeddings,
            support_scores=support_scores,
            support_base_score=support_base_score,
            user_adaptation=user_adaptation,
        )
        support_retrieved_residual = self._retrieve_support_residual(
            query_embeddings=support_similarity_embeddings,
            support_embeddings=support_similarity_embeddings,
            support_scores=support_scores,
            support_base_score=support_base_score,
            user_adaptation=user_adaptation,
        )
        query_kernel_residual = self._kernel_support_residual(
            query_embeddings=query_similarity_embeddings,
            support_embeddings=support_similarity_embeddings,
            support_scores=support_scores,
            support_base_score=support_base_score,
            user_adaptation=user_adaptation,
        )
        support_kernel_residual = self._kernel_support_residual(
            query_embeddings=support_similarity_embeddings,
            support_embeddings=support_similarity_embeddings,
            support_scores=support_scores,
            support_base_score=support_base_score,
            user_adaptation=user_adaptation,
        )
        query_user_state_residual = self._estimate_user_state_residual(
            query_embeddings=query_similarity_embeddings,
            support_embeddings=support_similarity_embeddings,
            support_scores=support_scores,
            support_base_score=support_base_score,
        )
        support_user_state_residual = self._estimate_user_state_residual(
            query_embeddings=support_similarity_embeddings,
            support_embeddings=support_similarity_embeddings,
            support_scores=support_scores,
            support_base_score=support_base_score,
        )
        query_bridge_residual = query_conditioned["residual_score"] * float(self.bridge_residual_scale)
        support_bridge_residual = support_conditioned["residual_score"] * float(self.bridge_residual_scale)
        bridge_fit_scale = self._fit_bridge_residual_scale(
            support_bridge_residual=support_bridge_residual,
            support_scores=support_scores,
            support_base_score=support_base_score,
        )
        query_bridge_residual = query_bridge_residual * bridge_fit_scale
        support_bridge_residual = support_bridge_residual * bridge_fit_scale
        query_bridge_basis_residual = self._fit_bridge_basis_residual(
            query_token_residuals=query_conditioned["token_residuals"],
            support_token_residuals=support_conditioned["token_residuals"],
            support_scores=support_scores,
            support_base_score=support_base_score,
        )
        support_bridge_basis_residual = self._fit_bridge_basis_residual(
            query_token_residuals=support_conditioned["token_residuals"],
            support_token_residuals=support_conditioned["token_residuals"],
            support_scores=support_scores,
            support_base_score=support_base_score,
        )
        query_bridge_total_residual = query_bridge_residual + query_bridge_basis_residual
        support_bridge_total_residual = support_bridge_residual + support_bridge_basis_residual
        query_component_residuals = {
            "bridge": query_bridge_residual,
            "bridge_basis": query_bridge_basis_residual,
            "retrieved": query_retrieved_residual,
            "kernel": query_kernel_residual,
            "user_state": query_user_state_residual,
        }
        support_component_residuals = {
            "bridge": support_bridge_residual,
            "bridge_basis": support_bridge_basis_residual,
            "retrieved": support_retrieved_residual,
            "kernel": support_kernel_residual,
            "user_state": support_user_state_residual,
        }
        fitted_fusion = self._fit_support_residual_fusion(
            query_components=query_component_residuals,
            support_components=support_component_residuals,
            support_scores=support_scores,
            support_base_score=support_base_score,
        )
        if fitted_fusion is None:
            query_unadapted_residual = (
                query_bridge_total_residual
                + query_retrieved_residual
                + query_kernel_residual
                + query_user_state_residual
            )
            support_unadapted_residual = (
                support_bridge_total_residual
                + support_retrieved_residual
                + support_kernel_residual
                + support_user_state_residual
            )
        else:
            query_unadapted_residual, support_unadapted_residual = fitted_fusion
        query_total_residual = self._apply_user_residual_adaptation(
            query_unadapted_residual,
            user_adaptation,
        )
        support_total_residual = self._apply_user_residual_adaptation(
            support_unadapted_residual,
            user_adaptation,
        )
        query_total_residual, support_total_residual, residual_score_fit_scale = (
            self._fit_support_residual_score_scale(
                query_residual=query_total_residual,
                support_residual=support_total_residual,
                support_scores=support_scores,
                support_base_score=support_base_score,
            )
        )
        query_score = self._apply_calibration(
            base_score=query_base_score,
            stage1_score=query_stage1["score"],
            prior_scores=query_prior_scores,
            residual_score=query_total_residual,
            calibration=calibration,
            user_adaptation=user_adaptation,
        )
        support_score = self._apply_calibration(
            base_score=support_base_score,
            stage1_score=support_stage1_score,
            prior_scores=support_prior_scores,
            residual_score=support_total_residual,
            calibration=calibration,
            user_adaptation=user_adaptation,
        )
        if compute_support_loo:
            support_loo = self._support_leave_one_out(
                support_attribute_tokens=support_attribute_tokens,
                support_image_embeddings=support_image_embeddings,
                support_scores=support_scores,
                support_generic_score=support_base_score,
                support_stage1_score=support_stage1_score,
                support_prior_scores=support_prior_scores,
                user_adaptation=user_adaptation,
                support_semantic_scores=support_semantic_scores,
                support_loo_indices=support_loo_indices,
            )
        else:
            support_loo = {
                "score": support_score.detach(),
                "raw_residual_score": support_conditioned["raw_residual_score"].detach(),
                "indices": torch.arange(support_scores.size(0), device=support_scores.device),
            }
        return {
            "score": query_score,
            "generic_score": query_base_score,
            "stage1_generic_score": query_stage1["score"],
            "query_prior_score": query_prior_scores,
            "residual_score": query_total_residual,
            "bridge_residual_score": query_bridge_residual,
            "bridge_basis_residual_score": query_bridge_basis_residual,
            "retrieved_residual_score": query_retrieved_residual,
            "kernel_residual_score": query_kernel_residual,
            "user_state_residual_score": query_user_state_residual,
            "raw_residual_score": query_conditioned["raw_residual_score"],
            "support_score": support_score,
            "support_generic_score": support_base_score,
            "support_stage1_generic_score": support_stage1_score,
            "support_prior_score": support_prior_scores,
            "support_residual_score": support_total_residual,
            "support_bridge_residual_score": support_bridge_residual,
            "support_bridge_basis_residual_score": support_bridge_basis_residual,
            "support_retrieved_residual_score": support_retrieved_residual,
            "support_kernel_residual_score": support_kernel_residual,
            "support_user_state_residual_score": support_user_state_residual,
            "support_raw_residual_score": support_conditioned["raw_residual_score"],
            "support_loo_score": support_loo["score"],
            "support_loo_raw_residual_score": support_loo["raw_residual_score"],
            "support_loo_indices": support_loo["indices"],
            "attribute_tokens": query_attribute_tokens,
            "conditioned_tokens": query_conditioned["conditioned_tokens"],
            "query_token_weights": query_conditioned["token_weights"],
            "query_token_residuals": query_conditioned["token_residuals"],
            "positive_memory": positive_memory,
            "negative_memory": negative_memory,
            "personalized_representation": query_conditioned["pooled_tokens"],
            "support_attribute_tokens": support_attribute_tokens,
            "support_image_embeddings": support_image_embeddings,
            "support_conditioned_tokens": support_conditioned["conditioned_tokens"],
            "support_personalized_representation": support_conditioned["pooled_tokens"],
            "support_token_weights": support_conditioned["token_weights"],
            "support_token_residuals": support_conditioned["token_residuals"],
            "calibration_stage1_delta_adjust": (
                calibration["stage1_delta_adjust"] if calibration is not None else query_base_score.new_zeros(1)
            ),
            "calibration_residual_gate": (
                calibration["residual_gate"] if calibration is not None else query_base_score.new_ones(1)
            ),
            "calibration_bias": (
                calibration["bias"] if calibration is not None else query_base_score.new_zeros(1)
            ),
            "bridge_residual_fit_scale": bridge_fit_scale.reshape(1),
            "support_residual_score_fit_scale": residual_score_fit_scale.reshape(1),
        }
