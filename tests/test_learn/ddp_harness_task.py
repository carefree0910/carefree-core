from __future__ import annotations

import os
import time
import torch
import argparse

import torch.distributed as dist

from typing import Optional
from pathlib import Path
from datetime import timedelta
from core.toolkit.misc import is_rank_0
from torch.distributed.elastic.multiprocessing.errors import record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=("success", "rank_failure", "global_main_write", "timeout"),
        required=True,
    )
    parser.add_argument("--shared-target", type=Path)
    return parser.parse_args()


def _run_success(rank: int) -> None:
    value = torch.tensor([rank + 1], dtype=torch.int64)
    dist.all_reduce(value)
    assert value.item() == 3
    print(f"rank {rank}: all_reduce success", flush=True)


def _run_rank_failure(rank: int) -> None:
    dist.barrier()
    if rank == 1:
        raise RuntimeError("intentional failure from rank 1")
    print("rank 0: waiting for rank 1 failure propagation", flush=True)
    time.sleep(60.0)


def _run_global_main_write(rank: int, shared_target: Optional[Path]) -> None:
    if shared_target is None:
        raise ValueError("'--shared-target' is required")
    expected_is_main = rank == 0
    assert is_rank_0() is expected_is_main
    if is_rank_0():
        shared_target.write_text(f"rank={rank}\n", encoding="utf-8")
    dist.barrier()
    assert shared_target.read_text(encoding="utf-8") == "rank=0\n"
    print(f"rank {rank}: observed the global-main artifact", flush=True)


def _run_timeout(rank: int) -> None:
    dist.barrier()
    print(f"rank {rank}: waiting for the parent timeout", flush=True)
    time.sleep(60.0)


@record
def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2
    assert os.environ["LOCAL_RANK"] == str(rank)
    assert not torch.cuda.is_available()

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=timedelta(seconds=15.0),
    )
    try:
        assert dist.get_backend() == "gloo"
        if args.scenario == "success":
            _run_success(rank)
        elif args.scenario == "rank_failure":
            _run_rank_failure(rank)
        elif args.scenario == "global_main_write":
            _run_global_main_write(rank, args.shared_target)
        else:
            _run_timeout(rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
