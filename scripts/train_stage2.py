from __future__ import annotations

import argparse
from datetime import timedelta
import copy
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cobra.engine.stage2_trainer import train_stage2
from cobra.utils.common import get_device, load_yaml, set_seed
from cobra.utils.support_size_overrides import apply_support_size_overrides
from cobra.utils.tracking import init_tracker


def _stage2_run_label(config: dict, support_size: int) -> str:
    experiment_cfg = config.get("experiment", {}) or {}
    explicit_label = experiment_cfg.get("checkpoint_label")
    if explicit_label:
        return str(explicit_label)
    train_sizes = [
        int(size)
        for size in config.get("loss", {}).get("train_support_sizes", [])
        if 0 < int(size) <= int(support_size)
    ]
    return "multishot" if len(set(train_sizes)) > 1 else f"s{int(support_size)}"


def _maybe_relaunch_with_torchrun(num_gpus: int) -> None:
    if num_gpus <= 1 or "LOCAL_RANK" in os.environ:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("--num-gpus > 1 requires CUDA, but CUDA is not available.")
    available = torch.cuda.device_count()
    if num_gpus > available:
        raise RuntimeError(f"Requested {num_gpus} GPUs, but only {available} are visible.")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(num_gpus),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    os.execvpe(sys.executable, command, env)


def _init_device() -> tuple[torch.device, int, int]:
    if "LOCAL_RANK" not in os.environ:
        return get_device(), 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=os.environ.get("COBRA_DDP_BACKEND", "gloo"), timeout=timedelta(hours=4))
    return torch.device("cuda", local_rank), dist.get_rank(), dist.get_world_size()


def _configure_cuda_linalg_backend(is_main: bool) -> None:
    backend = os.environ.get("COBRA_CUDA_LINALG_BACKEND", "").strip().lower()
    if not backend:
        return
    if backend not in {"default", "cusolver", "magma"}:
        raise ValueError(
            "COBRA_CUDA_LINALG_BACKEND must be one of: default, cusolver, magma."
        )
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.preferred_linalg_library(backend)
    if is_main:
        current = torch.backends.cuda.preferred_linalg_library()
        print(f"[COBRA] CUDA linalg backend: {current}", flush=True)



def main() -> None:
    parser = argparse.ArgumentParser(description="Train COBRA stage 2.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument(
        "--support-sizes",
        nargs="+",
        type=int,
        default=[100],
        help="Maximum support size for the unified Stage2 run. Multishot configs train one run at max support.",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of visible GPUs to use for DDP training.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest stage2 checkpoint for each support size.")
    parser.add_argument("--resume-checkpoint", default=None, help="Explicit checkpoint path to resume from.")
    args = parser.parse_args()

    _maybe_relaunch_with_torchrun(int(args.num_gpus))
    device, rank, _world_size = _init_device()
    is_main = rank == 0
    _configure_cuda_linalg_backend(is_main)

    base_config = load_yaml(args.config)
    data_config = load_yaml(args.data_config)
    if len(args.support_sizes) != 1:
        raise ValueError(
            "Stage2 is now a unified multishot run. Pass one maximum support size, normally '--support-sizes 100'."
        )
    try:
        for support_size in args.support_sizes:
            config = copy.deepcopy(base_config)
            config["data"]["support_size"] = int(support_size)
            apply_support_size_overrides(config, support_size)
            run_label = _stage2_run_label(config, support_size)
            tracking_cfg = config.setdefault("tracking", {})
            tracking_cfg["tags"] = [
                *base_config.get("tracking", {}).get("tags", []),
                run_label,
                f"max-support-{support_size}",
            ]
            if args.resume:
                config.setdefault("experiment", {})["resume"] = True
            if args.resume_checkpoint:
                config.setdefault("experiment", {})["resume_checkpoint"] = args.resume_checkpoint
            set_seed(int(config["experiment"]["seed"]))
            tracker = init_tracker(config, job_type=f"train_stage2_{run_label}") if is_main else None
            try:
                checkpoint = train_stage2(config, data_config, device, tracker=tracker)
                if is_main and tracker is not None:
                    tracker.log_summary({"artifacts/final_stage2_checkpoint": checkpoint.as_posix()})
                if is_main:
                    print(checkpoint.as_posix())
            finally:
                if tracker is not None:
                    tracker.finish()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
