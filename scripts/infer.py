from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cobra.data.image_batch import pad_image_list, prepare_image_tensor
from cobra.models.cobra_model import COBRAStage2Model
from cobra.utils.common import get_device, load_yaml, resolve_path
from cobra.utils.factory import build_model_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-user COBRA inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--support-images", nargs="+", required=True)
    parser.add_argument("--support-scores", nargs="+", type=float, required=True)
    parser.add_argument("--query-image", required=True)
    args = parser.parse_args()

    if len(args.support_images) != len(args.support_scores):
        raise ValueError("support-images and support-scores must have the same length")

    config = load_yaml(args.config)
    device = get_device()
    model = COBRAStage2Model(
        build_model_config(config),
        stage1_checkpoint=config["experiment"]["stage1_checkpoint"],
        contrast_checkpoint=config["experiment"].get("contrast_checkpoint"),
    ).to(device)
    checkpoint = torch.load(resolve_path(args.stage2_checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    backbone_cfg = config["backbone"]

    def load_image(path: str) -> tuple[torch.Tensor, tuple[int, int]]:
        with Image.open(resolve_path(path)) as image:
            return prepare_image_tensor(
                image=image,
                train=False,
                mean=backbone_cfg.get("mean"),
                std=backbone_cfg.get("std"),
                fixed_image_size=int(backbone_cfg["image_size"]),
                use_naflex=bool(backbone_cfg.get("use_naflex", False)),
                preferred_long_side=backbone_cfg.get("preferred_long_side"),
                patch_size=int(backbone_cfg["patch_size"]),
                max_num_patches=backbone_cfg.get("max_num_patches"),
            )

    support_tensors, support_shapes = zip(*[load_image(path) for path in args.support_images])
    support_scores = torch.tensor(args.support_scores, dtype=torch.float32, device=device)
    query_tensor, query_shape = load_image(args.query_image)
    support_images = {
        key: value.to(device)
        for key, value in pad_image_list(
            list(support_tensors),
            list(support_shapes),
            use_naflex=bool(backbone_cfg.get("use_naflex", False)),
            patch_size=int(backbone_cfg["patch_size"]),
            max_num_patches=backbone_cfg.get("max_num_patches"),
        ).items()
    }
    query_images = {
        key: value.to(device)
        for key, value in pad_image_list(
            [query_tensor],
            [query_shape],
            use_naflex=bool(backbone_cfg.get("use_naflex", False)),
            patch_size=int(backbone_cfg["patch_size"]),
            max_num_patches=backbone_cfg.get("max_num_patches"),
        ).items()
    }

    with torch.no_grad():
        output = model(support_images=support_images, support_scores=support_scores, query_images=query_images)
    print(json.dumps({"prediction": float(output["score"].item())}))


if __name__ == "__main__":
    main()
