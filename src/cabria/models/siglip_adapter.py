from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from cabria.utils.common import ensure_dir, resolve_path


class ResidualAdapter(nn.Module):
    def __init__(self, embed_dim: int, bottleneck_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.proj(self.norm(tokens))


class MockVisionBackbone(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if pixel_values.dim() == 4:
            tokens = self.conv(pixel_values).flatten(2).transpose(1, 2)
        elif pixel_values.dim() == 3:
            tokens = self.patch_proj(pixel_values)
        else:
            raise ValueError(f"Unsupported mock backbone input shape: {tuple(pixel_values.shape)}")
        tokens = self.norm(tokens)
        return {"patch_tokens": tokens, "pooled_output": tokens.mean(dim=1)}


class ResNetVisionBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, show_download_progress: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet101_Weights, ResNet50_Weights, resnet50, resnet101
        except ImportError as exc:
            raise ImportError("torchvision is required for ResNet backbones.") from exc

        name = model_name.lower().replace("torchvision/", "")
        if name == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            model = resnet101(weights=weights, progress=show_download_progress)
        elif name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights, progress=show_download_progress)
        else:
            raise ValueError(f"Unsupported ResNet backbone: {model_name!r}.")

        self.features = nn.Sequential(*list(model.children())[:-2])

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if pixel_values.dim() != 4:
            raise ValueError(f"ResNet backbones expect BCHW image tensors, got {tuple(pixel_values.shape)}")
        feature_map = self.features(pixel_values)
        patch_tokens = feature_map.flatten(2).transpose(1, 2)
        pooled_output = feature_map.mean(dim=(2, 3))
        return {"patch_tokens": patch_tokens, "pooled_output": pooled_output}


@dataclass
class BackboneConfig:
    model_name: str
    pretrained: bool
    image_size: int
    embed_dim: int
    patch_size: int
    adapter_dim: int
    adapter_dropout: float = 0.0
    model_dir: str | None = None
    show_download_progress: bool = True
    use_naflex: bool = False
    preferred_long_side: int | None = None
    max_num_patches: int | None = None


class Siglip2AdapterBackbone(nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = self._build_backbone(config)
        self.adapter = ResidualAdapter(config.embed_dim, config.adapter_dim, config.adapter_dropout)
        self.output_norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self._forward_backbone(pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)
        patch_tokens = self.output_norm(self.adapter(outputs["patch_tokens"]))
        return {"patch_tokens": patch_tokens, "pooled_output": outputs["pooled_output"]}

    def _build_backbone(self, config: BackboneConfig) -> nn.Module:
        if config.model_name == "mock":
            return MockVisionBackbone(embed_dim=config.embed_dim, patch_size=config.patch_size)
        if config.model_name.lower().replace("torchvision/", "") in {"resnet50", "resnet101"}:
            return ResNetVisionBackbone(
                model_name=config.model_name,
                pretrained=config.pretrained,
                show_download_progress=config.show_download_progress,
            )
        try:
            from transformers import Siglip2VisionModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for SigLIP-2. Install CABRIA requirements or set model_name=mock."
            ) from exc
        if not config.pretrained:
            raise ValueError("SigLIP-2 backbone currently expects pretrained weights from Hugging Face.")
        model_source = self._resolve_model_source(config)
        return Siglip2VisionModel.from_pretrained(model_source, local_files_only=isinstance(model_source, str) and Path(model_source).exists())

    def _resolve_model_source(self, config: BackboneConfig) -> str:
        if config.model_dir:
            target_dir = resolve_path(config.model_dir)
            if self._has_model_files(target_dir):
                print(f"[CABRIA] Loading SigLIP-2 from local backbone directory: {target_dir.as_posix()}")
                return str(target_dir)
            target_dir = ensure_dir(target_dir)
            print(f"[CABRIA] SigLIP-2 not found locally. Downloading into: {target_dir.as_posix()}")
            print("[CABRIA] First download may take a while. Hugging Face will display file progress.")
            try:
                from huggingface_hub import snapshot_download
                from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
            except ImportError as exc:
                raise ImportError(
                    "huggingface_hub is required to download SigLIP-2 into the project backbone directory."
                ) from exc
            if config.show_download_progress:
                enable_progress_bars()
            else:
                disable_progress_bars()
            snapshot_download(
                repo_id=config.model_name,
                local_dir=str(target_dir),
            )
            return str(target_dir)
        return config.model_name

    @staticmethod
    def _has_model_files(target_dir: Path) -> bool:
        return (target_dir / "config.json").exists()

    def _forward_backbone(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(self.backbone, (MockVisionBackbone, ResNetVisionBackbone)):
            return self.backbone(pixel_values)

        if pixel_values.dim() == 4:
            batch_size, _, height, width = pixel_values.shape
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=pixel_values.device)
            if spatial_shapes is None:
                spatial_shapes = torch.tensor([[height, width]] * batch_size, dtype=torch.long, device=pixel_values.device)
        elif pixel_values.dim() == 3:
            batch_size, num_patches, _ = pixel_values.shape
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones((batch_size, num_patches), dtype=torch.bool, device=pixel_values.device)
            if spatial_shapes is None:
                raise ValueError("NaFlex patch inputs require spatial_shapes.")
        else:
            raise ValueError(f"Unsupported SigLIP-2 input shape: {tuple(pixel_values.shape)}")

        outputs = self.backbone(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_hidden_states=False,
            output_attentions=False,
        )
        patch_tokens = outputs.last_hidden_state
        pooled_output = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else patch_tokens.mean(dim=1)
        return {"patch_tokens": patch_tokens, "pooled_output": pooled_output}
