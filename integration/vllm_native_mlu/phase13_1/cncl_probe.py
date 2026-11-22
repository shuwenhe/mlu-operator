"""Single-node TP=8 check using the installed Vendor vLLM device group."""
import json
import os

import torch
import torch.distributed as dist
import torch_mlu  # Registers the MLU device and CNCL backend.
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8 or rank != local_rank or torch.mlu.device_count() != 8:
        raise RuntimeError("Expected eight local ranks and eight visible MLU devices")
    if current_platform.dist_backend != "cncl":
        raise RuntimeError(f"Unexpected platform backend: {current_platform.dist_backend}")

    device = torch.device(f"mlu:{local_rank}")
    torch.mlu.set_device(device)
    init_distributed_environment(
        world_size=world_size, rank=rank, local_rank=local_rank,
        distributed_init_method="env://", backend="cncl",
    )
    try:
        initialize_model_parallel(tensor_model_parallel_size=world_size, backend="cncl")
        tp = get_tp_group()
        communicator = tp.device_communicator
        if (
            tp.world_size != world_size or tp.rank_in_group != rank
            or dist.get_backend(tp.device_group) != "cncl"
            or type(communicator).__name__ != "MLUCommunicator"
            or torch.mlu.current_device() != local_rank
        ):
            raise RuntimeError("Unexpected TP group or MLU device binding")

        for dtype in (torch.float32, torch.bfloat16):
            rows, cols = 2, 16
            local = torch.full((rows, cols), rank + 1, dtype=dtype, device=device)
            reduced = tp.all_reduce(local.clone())
            gathered = tp.all_gather(local, dim=1)
            shards = torch.arange(world_size, dtype=torch.float32).view(world_size, 1, 1)
            scatter_input = (shards + rank + 1).expand(world_size, rows, cols)
            scatter_input = scatter_input.reshape(world_size * rows, cols).to(device, dtype)
            scattered = tp.reduce_scatter(scatter_input, dim=0)
            torch.mlu.synchronize()

            rank_sum = world_size * (world_size + 1) // 2
            expected_gather = torch.arange(1, world_size + 1, dtype=dtype)
            expected_gather = expected_gather.repeat_interleave(cols).expand(rows, -1)
            torch.testing.assert_close(reduced.cpu(), torch.full((rows, cols), rank_sum, dtype=dtype), rtol=0, atol=0)
            torch.testing.assert_close(gathered.cpu(), expected_gather, rtol=0, atol=0)
            torch.testing.assert_close(scattered.cpu(), torch.full((rows, cols), rank_sum + world_size * rank, dtype=dtype), rtol=0, atol=0)
            print(json.dumps({
                "rank": rank, "device": str(device), "world_size": world_size,
                "backend": "cncl", "communicator": type(communicator).__name__,
                "dtype": str(dtype), "all_reduce": "PASS", "all_gather": "PASS",
                "reduce_scatter": "PASS",
            }), flush=True)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=8, disable_custom_all_reduce=True))
    with set_current_vllm_config(config):
        main()
