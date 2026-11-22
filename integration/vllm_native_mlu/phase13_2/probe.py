"""Hardware fixture and isolated single-card reference generation."""
import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch_mlu
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment, destroy_model_parallel, get_tp_group,
    init_distributed_environment, initialize_model_parallel,
)

from .checkpoint import checksum, load_local
from .linear import ColumnParallelLinear, RowParallelLinear, validate_tp
from .shards import column_spans, row_span


CASES = {
    "small": (128, 512, 256),
    "medium": (512, 2048, 1024),
    "large": (2048, 4096, 4096),
    "projection": (256, 7168, 4096),
}


def fixture(rows, cols, scale, dtype):
    row = torch.arange(rows.start, rows.stop, dtype=torch.int64)[:, None]
    col = torch.arange(cols.start, cols.stop, dtype=torch.int64)[None, :]
    values = ((row * 131 + col * 17) ^ (row * 41 + col * 107)) % 251 - 125
    return (values * scale).to(dtype)


def emit(kind, **fields):
    data = json.dumps({"kind": kind, **fields})
    os.write(1, (data + "\n").encode())


def timed(device, function):
    torch.mlu.synchronize(device)
    start = time.perf_counter()
    result = function()
    torch.mlu.synchronize(device)
    return result, (time.perf_counter() - start) * 1000


def compare(actual, expected, **fields):
    actual = actual.cpu().contiguous()
    expected = expected.contiguous()
    error = (actual - expected).abs()
    relative = error / expected.abs().clamp_min(torch.finfo(torch.float32).tiny)
    actual_bits = actual.view(torch.int32).to(torch.int64)
    expected_bits = expected.view(torch.int32).to(torch.int64)
    def ordered(bits):
        return torch.where(bits < 0, -2147483648 - bits, bits)
    mismatches = (actual_bits != expected_bits).flatten()
    bad = ~torch.isclose(actual, expected, atol=2e-5, rtol=2e-5)
    emit(
        "accuracy", **fields, max_abs=error.max().item(),
        max_rel=relative.max().item(), mean_abs=error.mean().item(),
        max_ulp=(ordered(actual_bits) - ordered(expected_bits)).abs().max().item(),
        bitwise_mismatches=mismatches.count_nonzero().item(),
        tolerance_mismatches=bad.count_nonzero().item(),
        first_mismatch=int(mismatches.nonzero()[0]) if mismatches.any() else None,
    )
    if bad.any():
        raise AssertionError(f"Numerical tolerance exceeded: {fields}")


def peak_bytes(device, layer):
    gemm = layer.gemm
    buffers = [packed for _, packed in gemm.packed_tiles]
    if gemm.packed is not None:
        buffers.append(gemm.packed)
    packed_bytes = sum(2 * ((p.k + 127) // 128 * 128) * ((p.n + 127) // 128 * 128)
                       for p in buffers)
    return torch.mlu.max_memory_allocated(device) + packed_bytes


def run_row(args, group, device, fields, x, residual, bias_cpu, refs):
    m, k = x.shape
    n = bias_cpu.numel()
    start, width = row_span(k, group.rank_in_group, group.world_size)
    weight = fixture(range(n), range(start, start + width), 1 / 1024, x.dtype)
    emit("weight", **fields, parameter="row.weight", original_shape=[n, k],
         local_shape=list(weight.shape), axis=1, spans=[(start, width)], sha256=checksum(weight))
    torch.mlu.reset_peak_memory_stats(device)
    def load_layer():
        w = weight.to(device)
        return RowParallelLinear(w, k, group, args.library, bias_cpu.to(device)), w

    (layer, w), load = timed(device, load_layer)
    try:
        local_input = x[:, start:start + width]
        expected = torch.load(refs, map_location="cpu")["row"]
        try:
            layer(local_input[:, :-1], input_is_parallel=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Row accepted mismatched shard input")
        try:
            layer(x, residual=residual[:, :-1])
        except ValueError:
            pass
        else:
            raise AssertionError("Row accepted mismatched residual")
        try:
            RowParallelLinear(w[:, :-1], k, group, args.library)
        except ValueError:
            pass
        else:
            raise AssertionError("Row accepted mismatched checkpoint shard")
        emit("negative", **fields, rejected=["activation_width", "residual_width", "weight_width"])
        partial = layer.gemm(local_input)
        local_expected = torch.mm(local_input.float(), w.float().T).cpu()
        compare(partial, local_expected, **fields, check="local")
        full = layer(x, residual=residual)
        compare(full, expected, **fields, check="replicated_input")
        compare(layer(local_input, input_is_parallel=True, residual=residual),
                expected, **fields, check="sharded_input")
        for iteration in range(args.repeats):
            out, end = timed(device, lambda: layer(x, residual=residual))
            comm_input = partial.clone()
            _, comm = timed(device, lambda: group.all_reduce(comm_input))
            _, gemm = timed(device, lambda: layer.gemm(local_input))
            if not torch.equal(out, full):
                raise AssertionError("Row repeated output changed")
            emit("performance", **fields, gemm_ms=gemm, communication_ms=comm,
                 end_to_end_ms=end, load_ms=load, iteration=iteration, peak_bytes=peak_bytes(device, layer))
    finally:
        layer.close()


def run_case(args, group, device, case, dtype):
    m, k, n = CASES[case]
    tp, rank = group.world_size, group.rank_in_group
    fp = str(dtype).removeprefix("torch.")
    fields = dict(mode=args.mode, case=case, dtype=fp, tp=tp, rank=rank, device=str(device))
    refs = args.output / f"{case}_{fp}.pt"
    sections = (n // 2, n // 2)
    bias_cpu = fixture(range(1), range(n), 1 / 512, dtype).flatten()
    x = fixture(range(m), range(k), 1 / 1024, dtype).to(device)
    residual = fixture(range(m), range(n), 1 / 1024, dtype).to(device)
    if args.mode == "reference":
        weight = fixture(range(n), range(k), 1 / 1024, dtype).to(device)
        ref = torch.mm(x.float(), weight.float().T) + bias_cpu.to(device)
        torch.save({"column": ref.cpu(), "row": (ref + residual).cpu()}, refs)
        return

    if args.mode == "row":
        run_row(args, group, device, fields, x, residual, bias_cpu, refs)
        return

    spans = column_spans(sections, rank, tp)
    weight = torch.cat(
        [fixture(range(offset, offset + size), range(k), 1 / 1024, dtype)
         for offset, size in spans]
    )
    bias = torch.cat([bias_cpu[offset:offset + size] for offset, size in spans])
    emit("weight", **fields, parameter="packed.weight", original_shape=[n, k],
         local_shape=list(weight.shape), axis=0, spans=spans,
         sha256=checksum(weight))
    torch.mlu.reset_peak_memory_stats(device)
    layer, load = timed(device, lambda: ColumnParallelLinear(
        weight.to(device), sections, group, args.library, bias.to(device),
    ))
    try:
        expected = torch.load(refs, map_location="cpu")["column"]
        local_expected = torch.cat([expected[:, offset:offset + size] for offset, size in spans], dim=1)
        try:
            layer(x[:, :-1])
        except ValueError:
            pass
        else:
            raise AssertionError("Column accepted mismatched input")
        try:
            ColumnParallelLinear(weight[:-1].to(device), sections, group, args.library)
        except ValueError:
            pass
        else:
            raise AssertionError("Column accepted mismatched checkpoint shard")
        emit("negative", **fields, rejected=["activation_width", "weight_width"])
        local = layer(x)
        compare(local, local_expected, **fields, check="local")
        full_reference = layer.gather(local)
        compare(full_reference, expected, **fields, check="gather")
        for iteration in range(args.repeats):
            out, end = timed(device, lambda: layer(x))
            _, comm = timed(device, lambda: layer.gather(out))
            full, global_end = timed(device, lambda: layer.gather(layer(x)))
            gemm_out, gemm = timed(device, lambda: layer.gemm(x))
            if not torch.equal(full, full_reference) or not torch.equal(gemm_out + layer.bias, local):
                raise AssertionError("Column repeated output changed")
            emit("performance", **fields, gemm_ms=gemm, communication_ms=comm,
                 end_to_end_ms=end, global_end_to_end_ms=global_end, load_ms=load,
                 iteration=iteration, peak_bytes=peak_bytes(device, layer))
    finally:
        layer.close()


def run_wide_checkpoint(group, device, library, output_dir):
    tp, rank = group.world_size, group.rank_in_group
    weights = {
        name: (shape, shard) for name, _, shape, _, _, shard in load_local(Path("/checkpoint"), tp, rank)
        if name.endswith(("gate_proj.weight", "up_proj.weight"))
    }
    prefix = "model.layers.0.mlp."
    gate_shape, gate = weights[prefix + "gate_proj.weight"]
    up_shape, up = weights[prefix + "up_proj.weight"]
    sections = (gate_shape[0], up_shape[0])
    weight = torch.cat((gate, up)).to(device)
    x = fixture(range(1), range(weight.shape[1]), 1 / 1024, weight.dtype).to(device)
    reference = output_dir / "checkpoint_gate_up_reference.pt"
    if tp == 1:
        expected = torch.mm(x.float(), weight.float().T).cpu()
        torch.save(expected, reference)
    else:
        expected = torch.load(reference, map_location="cpu")
    layer = ColumnParallelLinear(weight, sections, group, library)
    fields = dict(mode="wide_checkpoint", case="gate_up", dtype="bfloat16",
                  tp=tp, rank=rank, device=str(device))
    try:
        local = layer(x)
        spans = column_spans(sections, rank, tp)
        local_expected = torch.cat([expected[:, offset:offset + width] for offset, width in spans], dim=1)
        compare(local, local_expected, **fields, check="local")
        compare(layer.gather(local), expected, **fields, check="gather")
        emit("wide_checkpoint", **fields, sections=sections)
    finally:
        layer.close()


def run_precision(group, device, library):
    # Both operands have significant bits that would be lost to TF32 or BF16.
    value = 1.0 + 2.0**-13
    tp, rank = group.world_size, group.rank_in_group
    x = torch.zeros((1, tp), dtype=torch.float32, device=device)
    x[0, 0] = value
    w_column = torch.full((1, tp), value, dtype=torch.float32, device=device)
    w_row = torch.full((tp, 1), value, dtype=torch.float32, device=device)
    expected = torch.full((1, tp), value * value, dtype=torch.float32, device=device)
    column = ColumnParallelLinear(w_column, (tp,), group, library)
    row = RowParallelLinear(w_row, tp, group, library)
    try:
        if not torch.equal(column.gather(column(x)), expected) or not torch.equal(row(x), expected):
            raise AssertionError("FP32 input precision lost")
        emit("precision", tp=tp, rank=rank, device=str(device), expected=expected[0, 0].item())
        bias = (torch.arange(tp, device=device, dtype=torch.float32) + 1) / 16
        column.bias = bias[rank:rank + 1].clone()
        row.bias = bias
        if not torch.equal(column.gather(column(x)), expected + bias) or not torch.equal(row(x), expected + bias):
            raise AssertionError("Incorrect bias ownership or addition")
        for mode, layer, offset in (("column", column, rank), ("row", row, 0)):
            emit("weight", mode=mode, case="precision", dtype="float32", tp=tp, rank=rank,
                 device=str(layer.bias.device), parameter=f"{mode}.bias",
                 original_shape=[tp], local_shape=list(layer.bias.shape), axis=0,
                 spans=[(offset, layer.bias.numel())], sha256=checksum(layer.bias.cpu()))
    finally:
        column.close()
        row.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["reference", "column", "row", "checkpoint", "precision", "wide_checkpoint"])
    parser.add_argument("output", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if tp not in (1, 2, 4, 8) or rank != int(os.environ["LOCAL_RANK"]):
        raise RuntimeError("Expected single-node TP=1/2/4/8")
    device = torch.device("mlu", rank)
    if torch.mlu.device_count() != tp:
        raise RuntimeError("Visible MLU count must equal requested TP size")
    torch.mlu.set_device(device)
    torch.backends.mlu.matmul.allow_tf32 = False
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp, disable_custom_all_reduce=True))
    with torch.inference_mode(), set_current_vllm_config(config), torch.mlu.stream(torch.mlu.Stream(device=device)):
        init_distributed_environment(
            world_size=tp, rank=rank, local_rank=rank,
            distributed_init_method="env://", backend="cncl",
        )
        try:
            initialize_model_parallel(tensor_model_parallel_size=tp, backend="cncl")
            group = get_tp_group()
            validate_tp(group, device)
            if (group.world_size, group.rank_in_group) != (tp, rank):
                raise RuntimeError("Unexpected Vendor TP size or rank")
            emit("rank", tp=tp, rank=rank, device=str(device), backend=dist.get_backend(group.device_group),
                 communicator=type(group.device_communicator).__name__)
            if args.mode == "precision":
                run_precision(group, device, args.library)
            elif args.mode == "wide_checkpoint":
                run_wide_checkpoint(group, device, args.library, args.output)
            elif args.mode == "checkpoint":
                for name, axis, shape, offset, width, shard in load_local(Path("/checkpoint"), tp, rank):
                    local = shard.to(device)
                    emit("weight", mode="checkpoint", case="deepseek", dtype=str(shard.dtype),
                         tp=tp, rank=rank, device=str(local.device), parameter=name,
                         original_shape=shape, local_shape=list(local.shape), axis=axis,
                         spans=[(offset, width)], sha256=checksum(local.cpu()))
            else:
                for case in args.cases:
                    for dtype in (torch.bfloat16, torch.float32):
                        run_case(args, group, device, case, dtype)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
    emit("cleanup", mode=args.mode, tp=tp, rank=rank)


if __name__ == "__main__":
    main()
