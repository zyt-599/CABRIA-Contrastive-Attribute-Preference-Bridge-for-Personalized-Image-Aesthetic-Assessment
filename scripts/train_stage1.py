from __future__ import annotations

import argparse
from datetime import timedelta
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cobra.engine.stage1_trainer import train_stage1
from cobra.utils.common import get_device, load_yaml, set_seed
from cobra.utils.tracking import init_tracker


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


def _init_device() -> tuple[torch.device, int]:
    if "LOCAL_RANK" not in os.environ:
        return get_device(), 0
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=os.environ.get("COBRA_DDP_BACKEND", "gloo"), timeout=timedelta(hours=4))
    return torch.device("cuda", local_rank), dist.get_rank()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train COBRA stage 1.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "stage1_giaa_flickr_prior17337.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "data_stage1_flickr_prior17337.yaml"))
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of visible GPUs to use for DDP training.")
    args = parser.parse_args()

    _maybe_relaunch_with_torchrun(int(args.num_gpus))
    device, rank = _init_device()
    is_main = rank == 0

    config = load_yaml(args.config)
    data_config = load_yaml(args.data_config)
    set_seed(int(config["experiment"]["seed"]))
    tracker = init_tracker(config, job_type="train_stage1") if is_main else None
    try:
        checkpoint = train_stage1(config, data_config, device, tracker=tracker)
        if is_main and tracker is not None:
            tracker.log_summary({"artifacts/final_stage1_checkpoint": checkpoint.as_posix()})
        if is_main:
            print(checkpoint.as_posix())
    finally:
        if tracker is not None:
            tracker.finish()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
