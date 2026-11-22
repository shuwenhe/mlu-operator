"""Eight-MLU acceptance and synchronized MoE alternative measurements."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import torch_mlu  # noqa: F401
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment, destroy_model_parallel, get_tp_group,
    init_distributed_environment, initialize_model_parallel,
)

from ...phase13_2.linear import validate_tp
from ...phase13_3_2.probe import LIB, ROOT, bf16
from ...phase13_4.mla import NativeCalls
from ..tp8.model import ExpertParallelDecoder
from .moe import TunableDecoder, owner_slots


def require(group, device, condition, name):
    bad = torch.tensor([not condition], dtype=torch.int32, device=device)
    if group.all_reduce(bad).item():
        raise AssertionError(f"Acceptance failed: {name}")


def emit(rank, **row):
    with Path(f"/results/rank{rank}.jsonl").open("a") as file:
        file.write(json.dumps(dict(rank=rank, **row)) + "\n")


def timed(ops, operation):
    start = perf_counter()
    result = operation()
    ops.stream.synchronize()
    return result, (perf_counter() - start) * 1000


def probe(group, device, overlap_only):
    validate_tp(group, device)
    rank = group.rank_in_group
    info = torch.mlu.get_device_properties(rank)
    require(group, device, group.world_size == 8 and device.index == rank and dist.get_rank() == rank
            and "MLU590-M9" in info.name, "MLU590 rank/device")
    require(group, device, group.all_gather(torch.tensor([rank], device=device, dtype=torch.int32)).tolist()
            == list(range(8)), "CNCL rank order")
    emit(rank, gate="rank_device", device=str(device), card=info.name,
         backend=dist.get_backend(group.device_group))
    cos, sin = (bf16(ROOT / "rope_yarn" / f"{name}.bin", (131, 64)).to(device) for name in ("cos", "sin"))
    scale = float((ROOT / "rope_yarn/scale.txt").read_text())
    ops = NativeCalls(group, device, LIB, Path(os.environ["NEURX_PAGED_TWO"]), cos, sin, scale)
    reference = ExpertParallelDecoder(ops)
    variants = {
        "compact": TunableDecoder(ops, compact=True),
        "fused": TunableDecoder(ops, fuse=True),
        "batched": TunableDecoder(ops, batch=True),
        "fused_batched": TunableDecoder(ops, fuse=True, batch=True),
        "all": TunableDecoder(ops, compact=True, fuse=True, batch=True),
    }
    if overlap_only:
        variants = {
            "fused_batched": variants["fused_batched"],
            "overlap": TunableDecoder(ops, overlap=True),
            "fused_batched_overlap": TunableDecoder(ops, fuse=True, batch=True, overlap=True),
        }
    fused_decoder = variants["fused_batched"] if overlap_only else variants["fused"]
    try:
        fixture = ROOT / "native-dump-yarn-scale/request2"
        for step in (0, 1):
            for layer in (1, 13, 26):
                prefix = f"model.layers.{layer}."
                hidden = bf16(fixture / f"{step}-layer{layer-1}.bin", (-1, 2048)).to(device)
                x = reference.norm.forward_native(hidden, reference._resident(prefix + "post_attention_layernorm.weight"), 1e-6)
                mlp = prefix + "mlp."
                chosen, weights = fused_decoder.route(mlp, x)
                owners = owner_slots(chosen, group.world_size)
                expected = reference._moe(mlp, x)
                for name, decoder in variants.items():
                    actual = decoder._moe(mlp, x)
                    require(group, device, torch.equal(actual, expected) and torch.isfinite(actual).all().item(),
                            f"{name} MoE parity layer={layer} step={step}")
                owned = sorted({chosen[slot] for slot in owners[rank]})
                for expert in [f"{mlp}experts.{i}." for i in owned] + [mlp + "shared_experts."]:
                    gate = reference._project(x, expert + "gate_proj.weight")
                    up = reference._project(x, expert + "up_proj.weight")
                    fused = fused_decoder.project_gate_up(x, expert)
                    require_local = torch.equal(fused, torch.cat((gate, up), dim=1))
                    if not require_local:
                        raise AssertionError(f"Fused FP32 expert GEMM mismatch on rank {rank} for {expert}")
                emit(rank, gate="numerical", layer=layer, step=step, rows=x.shape[0],
                     slots=len(owners[rank]), experts=len(owned), capacity=max(map(len, owners)))
                for sample in range(10):
                    cases = list({"baseline": reference, **variants}.items())
                    offset = sample % len(cases)
                    for name, decoder in cases[offset:] + cases[:offset]:
                        dist.barrier(group=group.device_group)
                        _, elapsed = timed(ops, lambda: decoder._moe(mlp, x))
                        emit(rank, variant=name, layer=layer, step=step, rows=x.shape[0],
                             sample=sample, milliseconds=elapsed)
    finally:
        for decoder in (reference, *variants.values()):
            decoder.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-only", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.mlu.set_device(rank)
    device = torch.device("mlu", rank)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=8, disable_custom_all_reduce=True))
    with torch.inference_mode(), set_current_vllm_config(config), torch.mlu.stream(torch.mlu.Stream(device=device)):
        init_distributed_environment(world_size=8, rank=rank, local_rank=rank,
                                     distributed_init_method="env://", backend="cncl")
        try:
            initialize_model_parallel(tensor_model_parallel_size=8, backend="cncl")
            probe(get_tp_group(), device, args.overlap_only)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
            emit(rank, lifecycle="cleanup")


if __name__ == "__main__":
    main()
