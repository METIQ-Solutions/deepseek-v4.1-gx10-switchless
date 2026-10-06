#!/usr/bin/env python3
"""Model-free four-rank NCCL collective over the switchless ring.

This proves the transport independently of the model: if the ring-only NCCL
build, the /30 addressing and the GID selection are correct, four ranks
complete an all-reduce; if anything is wrong, the collective either fails or
falls back to a transport the ring cannot actually carry.

Run inside the runtime image on every rank with RANK/WORLD_SIZE/MASTER_ADDR/
MASTER_PORT set. Prints transport evidence to stdout and exits non-zero on any
mismatch.
"""

from __future__ import annotations

import datetime
import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    if world_size != 4:
        print(f"COLLECTIVE_FAIL expected a four-rank ring, got world_size={world_size}", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("COLLECTIVE_FAIL CUDA is not available in the container", file=sys.stderr)
        return 1

    nccl_version = torch.cuda.nccl.version()
    print(f"rank={rank} torch={torch.__version__} cuda={torch.version.cuda} nccl={nccl_version}", flush=True)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=10),
    )

    device = torch.device("cuda", torch.cuda.current_device())
    tensor = torch.full((1 << 20,), float(rank), dtype=torch.float32, device=device)
    dist.all_reduce(tensor)
    torch.cuda.synchronize(device)

    expected = float(world_size * (world_size - 1) // 2)
    observed = float(tensor[0].item())
    if observed != expected:
        print(f"COLLECTIVE_FAIL all_reduce produced {observed}, expected {expected}", file=sys.stderr)
        return 1

    dist.barrier()
    dist.destroy_process_group()
    print(f"COLLECTIVE_OK rank={rank} world_size={world_size} all_reduce={observed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
