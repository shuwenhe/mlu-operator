"""Isolated hardware regression for ordered rank-local O accumulation."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch_mlu
from safetensors import safe_open
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment, destroy_model_parallel, get_tp_group,
    init_distributed_environment, initialize_model_parallel,
)

from .. import NativeRMSNorm
from ..phase13_2.linear import ColumnParallelLinear
from ..phase13_3_2.projection import MLAProjection, checksum
from ..phase13_3_2.probe import HISTORY, LIB, MODEL, ROOT, bf16, compare, emit


REFERENCE = Path("/project/artifacts/build/phase13_3_2")


def bitwise(a, b):
    kind = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return torch.equal(a.contiguous().view(kind), b.contiguous().view(kind))


def measured(fn, synchronize):
    start = time.perf_counter()
    result = fn()
    synchronize()
    return result, 1000 * (time.perf_counter() - start)


class OrderedOutput:
    def __init__(self, library, row, group, device, rows):
        self.group, self.device = group, device
        self.rank, self.tp = group.rank_in_group, group.world_size
        self.state = torch.empty((rows, 16, 64, 128), dtype=torch.float32, device=device)
        self.scratch_bytes = self.state.numel() * self.state.element_size()
        self.out = torch.empty((rows, 2048), dtype=torch.float32, device=device)
        self.packed = row.gemm.packed
        self.stream = torch.mlu.current_stream(device)
        self.launch = ctypes.CDLL(str(library)).phase1333_continue
        self.launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
        self.launch.restype = ctypes.c_int

    def __call__(self, x, residual):
        if (
            x.shape != (self.state.shape[0], self.packed.k)
            or not x.is_contiguous() or x.dtype != torch.bfloat16 or x.device != self.device
            or residual.shape != self.out.shape or residual.dtype != x.dtype or residual.device != x.device
        ):
            raise ValueError("Invalid rank-local O input or residual")
        local_ms = comm_ms = 0.0
        for rank, source in enumerate(self.group.ranks):
            last = rank == self.tp - 1
            if rank == self.rank:
                status, local_ms = measured(
                    lambda: self.launch(self.stream.mlu_stream, x.data_ptr(), self.packed.handle,
                                        self.state.data_ptr(), self.out.data_ptr(), x.shape[0], rank == 0, last),
                    self.stream.synchronize,
                )
                if status:
                    raise RuntimeError(f"Ordered GEMM CNRT status {status}")
            if self.tp > 1:
                _, elapsed = measured(
                    lambda: dist.broadcast(self.out if last else self.state, src=source,
                                           group=self.group.device_group),
                    self.stream.synchronize,
                )
                comm_ms += elapsed
        rounded, cast_ms = measured(lambda: self.out.to(torch.bfloat16), self.stream.synchronize)
        added = rounded + residual
        self.stream.synchronize()
        return rounded, added, dict(local_gemm_fold_ms=local_ms, communication_ms=comm_ms, cast_ms=cast_ms)


class DeferredExperiment(OrderedOutput):
    """Diagnostic for independent lane-state SUM followed by the frozen fold."""

    def __init__(self, library, row, group, device, rows):
        super().__init__(library, row, group, device, rows)
        self.fold = ctypes.CDLL(str(library.with_name("libdeferred.so"))).phase1333_fold
        self.fold.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int]
        self.fold.restype = ctypes.c_int

    def __call__(self, x, residual):
        status, local_ms = measured(
            lambda: self.launch(self.stream.mlu_stream, x.data_ptr(), self.packed.handle,
                                self.state.data_ptr(), self.out.data_ptr(), x.shape[0], True, False),
            self.stream.synchronize,
        )
        if status:
            raise RuntimeError(f"Deferred accumulation CNRT status {status}")
        state, comm_ms = measured(lambda: self.group.all_reduce(self.state), self.stream.synchronize)
        status, fold_ms = measured(
            lambda: self.fold(self.stream.mlu_stream, state.data_ptr(), self.out.data_ptr(), x.shape[0]),
            self.stream.synchronize,
        )
        if status:
            raise RuntimeError(f"Deferred fold CNRT status {status}")
        rounded, cast_ms = measured(lambda: self.out.to(torch.bfloat16), self.stream.synchronize)
        added = rounded + residual
        self.stream.synchronize()
        return rounded, added, dict(local_gemm_fold_ms=local_ms+fold_ms, communication_ms=comm_ms,
                                    cast_ms=cast_ms, fold_only_ms=fold_ms)


class ColumnExperiment:
    """Full-K per output tile, evaluated separately from the row contract."""

    def __init__(self, column, group, rows):
        self.column, self.group = column, group
        self.scratch_bytes = rows * (2048 * 6 + 4 * (2048 // group.world_size))
        self.sync = torch.mlu.current_stream().synchronize

    def __call__(self, x, residual):
        full, gather_in = measured(lambda: self.group.all_gather(x, dim=1), self.sync)
        local, gemm = measured(lambda: self.column(full.contiguous()), self.sync)
        self.out, gather_out = measured(lambda: self.column.gather(local), self.sync)
        rounded, cast = measured(lambda: self.out.to(torch.bfloat16), self.sync)
        added = rounded + residual
        self.sync()
        return rounded, added, dict(local_gemm_fold_ms=gemm, communication_ms=gather_in+gather_out,
                                    cast_ms=cast)


def run(args, group, device):
    rank, tp = group.rank_in_group, group.world_size
    meta = dict(rank=rank, tp=tp, device=str(device))
    assert dist.get_backend(group.device_group) == "cncl"
    assert tp == 1 or type(group.device_communicator).__name__ == "MLUCommunicator"
    emit("rank", **meta, backend="cncl", communicator=type(group.device_communicator).__name__)
    layer = MLAProjection(MODEL, 0, group, device, LIB,
                          lambda name, w, mapping: emit("weight", **meta, parameter=name,
                                                         sha256=checksum(w), mapping=mapping))
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    gamma_name = "model.layers.0.input_layernorm.weight"
    with safe_open(MODEL / index[gamma_name], framework="pt", device="cpu") as file:
        gamma = file.get_tensor(gamma_name).to(device)
    norm = NativeRMSNorm(LIB / "attention/libnative_mlu.so")
    sync = torch.mlu.current_stream().synchronize
    bad = 0
    column = None
    try:
        if args.output_axis:
            name = "model.layers.0.self_attn.o_proj.weight"
            rows = 2048 // tp
            with safe_open(MODEL / index[name], framework="pt", device="cpu") as file:
                weight = file.get_slice(name)[rank*rows:(rank+1)*rows].to(device)
            column = ColumnParallelLinear(weight, (2048,), group, LIB / "projection/libprojection.so")
            emit("alternative_weight", **meta, shape=list(weight.shape), sha256=checksum(weight))
            del weight
        for request, length in enumerate((1, 7, 17, 65)):
            root = ROOT / "native-dump-yarn" / f"request{request}"
            prompt = (root / "prompt.ids").read_bytes()
            assert prompt == (HISTORY / f"prompts/{length}.bin").read_bytes()
            for step in range(4):
                saved = REFERENCE / f"boundary_{length}_{step}.pt"
                ref = torch.load(saved, map_location="cpu", weights_only=True)
                residual = bf16(root / f"{step}-embedding.bin", (-1, 2048)).to(device)
                value = ref["av"].to(device)
                width = value.shape[1] // tp
                local = value[:, rank*width:(rank+1)*width].contiguous()
                case = dict(**meta, length=length, step=step,
                            fixture_sha256=checksum(ref["av"]), prompt_sha256=hashlib.sha256(prompt).hexdigest())
                if column is None:
                    projection = DeferredExperiment if args.deferred else OrderedOutput
                    projector = projection(args.output / "libordered.so", layer.o, group, device, value.shape[0])
                else:
                    projector = ColumnExperiment(column, group, value.shape[0])
                partial = layer.o.gemm(local)
                summed = group.all_reduce(partial.clone())
                emit("legacy", **case, partial_sha256=checksum(partial), cncl_sum_sha256=checksum(summed),
                     shape=list(partial.shape))
                bad += compare(summed, ref["reduced"], case, "legacy_fp32_cncl_sum_diagnostic")
                rounded, added, _ = projector(local, residual)
                frozen = bf16(HISTORY / f"reference{length}" / f"{step}-layer0-attention.bin", (-1, 2048))
                for name, actual, expected in (
                    ("ordered_fp32", projector.out, ref["reduced"]),
                    ("frozen_o_bf16", rounded, frozen),
                    ("residual_bf16", added, ref["added"]),
                ):
                    bad += compare(actual, expected, case, name)
                    if not bitwise(actual.cpu(), expected):
                        bad += 1
                if args.capture:
                    capture = args.output / "tensors"
                    capture.mkdir(exist_ok=True)
                    path = capture / f"tp{tp}_rank{rank}_len{length}_step{step}.pt"
                    tensors = dict(partial=partial, legacy_sum=summed, ordered_fp32=projector.out,
                                   ordered_o=rounded, ordered_residual=added)
                    with path.open("xb") as stream:
                        torch.save({name: tensor.cpu() for name, tensor in tensors.items()}, stream)
                if (length, step) == (7, 2):
                    midpoint = projector.out.flatten()[1134].item()
                    emit("midpoint", **case, index=1134, fp32=midpoint, bf16=rounded.flatten()[1134].item(),
                         legacy_fp32=summed.flatten()[1134].item())
                    bad += midpoint != 0.0543212965130806
                if (length, step) == (17, 0):
                    torch.mlu.reset_peak_memory_stats(device)
                    hidden = norm.forward_native(residual, gamma, 1e-6)
                    expected_fp32 = projector.out.clone()
                    for repeat in range(args.repeats):
                        current = layer.o.gemm(local)
                        if not bitwise(current, partial):
                            bad += 1
                        (out, total, stages), o_ms = measured(lambda: projector(local, residual), sync)
                        if not all(bitwise(a, b) for a, b in
                                   ((out, rounded), (total, added), (projector.out, expected_fp32))):
                            bad += 1

                        def chain():
                            layer.query(hidden)
                            latent, _ = layer.latent(hidden)
                            layer.expand(layer.latent_norm(latent))
                            projector(local, residual)

                        _, chain_ms = measured(chain, sync)
                        emit("performance", **case, **stages, repeat=repeat, o_ms=o_ms, chain_ms=chain_ms,
                             scratch_bytes=projector.scratch_bytes,
                             peak_allocator=torch.mlu.max_memory_allocated(device))
    finally:
        if column is not None:
            column.close()
        layer.close()
    emit("result", **meta, mismatches=bad)
    if bad:
        raise AssertionError(f"Ordered projection failures: {bad}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("test",))
    parser.add_argument("output", type=Path)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--output-axis", action="store_true")
    variants.add_argument("--deferred", action="store_true")
    parser.add_argument("--capture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.capture and (args.output_axis or args.deferred):
        parser.error("Alternative experiments require --no-capture to protect ordered evidence")
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert rank == int(os.environ["LOCAL_RANK"]) and torch.mlu.device_count() == tp
    device = torch.device("mlu", rank)
    torch.mlu.set_device(device)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp, disable_custom_all_reduce=True))
    with torch.inference_mode(), set_current_vllm_config(config), torch.mlu.stream(torch.mlu.Stream(device=device)):
        init_distributed_environment(world_size=tp, rank=rank, local_rank=rank,
                                     distributed_init_method="env://", backend="cncl")
        try:
            initialize_model_parallel(tensor_model_parallel_size=tp, backend="cncl")
            run(args, get_tp_group(), device)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
    emit("cleanup", tp=tp, rank=rank)


if __name__ == "__main__":
    main()
