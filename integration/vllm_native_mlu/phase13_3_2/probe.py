"""Frozen activation replay and distributed projection acceptance."""
import argparse
import csv
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
from ..attention import NativeMaterializedAttention
from .projection import MLAProjection, checksum, load_weight


ROOT = Path("/project/artifacts/build/phase12_5_4")
LIB = ROOT / "nativelibs"
HISTORY = Path("/project/artifacts/build/vllm_mlu/phase12_5")
MODEL = Path("/checkpoint")


def emit(kind, **fields):
    os.write(1, (json.dumps(dict(kind=kind, **fields)) + "\n").encode())


def bf16(path, shape):
    return torch.from_file(str(path), size=path.stat().st_size // 2, dtype=torch.int16).view(
        torch.bfloat16).reshape(shape)


def compare(actual, expected, fields, name):
    a, b = actual.cpu().contiguous(), expected.cpu().contiguous()
    assert a.shape == b.shape and a.dtype == b.dtype, name
    error = (a.double() - b.double()).abs()
    bad = (~torch.isfinite(a)) | (~torch.isfinite(b)) | (error > 1e-5 + 1e-5 * b.double().abs())
    bits = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    sign = -(1 << (15 if bits == torch.int16 else 31))
    ai, bi = [value.view(bits).long() for value in (a, b)]
    ulp = (torch.where(ai < 0, sign-ai, ai) - torch.where(bi < 0, sign-bi, bi)).abs()
    emit("accuracy", **fields, boundary=name, dtype=str(a.dtype), shape=list(a.shape),
         max_abs=error.max().item(), max_rel=(error/b.double().abs().clamp_min(1e-30)).max().item(),
         mean_abs=error.mean().item(), max_ulp=ulp.max().item(), bit_mismatch=(ai != bi).sum().item(),
         tolerance_mismatch=bad.sum().item(),
         first_mismatch=int(bad.flatten().nonzero()[0]) if bad.any() else None,
         nan=a.isnan().sum().item(), inf=a.isinf().sum().item(),
         reference_sha256=checksum(b), result_sha256=checksum(a))
    return int(bad.sum())


def timed(fn, device):
    begin, end = torch.mlu.Event(enable_timing=True), torch.mlu.Event(enable_timing=True)
    torch.mlu.synchronize(device)
    start = time.perf_counter()
    begin.record()
    result = fn()
    end.record()
    end.synchronize()
    return result, (time.perf_counter()-start)*1000, begin.elapsed_time(end)


def frozen_value_input(q, directory, step, length, device, attention):
    rows = q.shape[0]
    timeline = length if step == 0 else length + step
    pos = torch.arange(timeline-rows, timeline, dtype=torch.int32, device=device)
    cos, sin = [bf16(ROOT / "rope_yarn" / f"{name}.bin", (131, 64)).to(device)[pos.long()]
                for name in ("cos", "sin")]
    query = q.reshape(rows, 16, 192).transpose(0, 1)
    pe = query[..., 128:].to(torch.bfloat16)
    even, odd = pe[..., ::2], pe[..., 1::2]
    pe = torch.cat((even, odd), -1)*cos + torch.cat((-odd, even), -1)*sin
    q = torch.cat((query[..., :128].to(torch.bfloat16), pe), -1).contiguous()
    k = bf16(directory / f"{step}-layer0-key.bin", (16, timeline, 192)).to(device)
    v = bf16(directory / f"{step}-layer0-value.bin", (16, timeline, 128)).to(device)
    scale = float((ROOT / "rope_yarn/scale.txt").read_text())
    return attention.forward_native(q, k, v, pos, 0, timeline, scale)[-1].transpose(
        0, 1).contiguous().view(rows, 2048)


def run(args, group, device):
    rank, tp = group.rank_in_group, group.world_size
    fields = dict(rank=rank, tp=tp, device=str(device))
    emit("rank", **fields, global_rank=int(os.environ["RANK"]), local_rank=int(os.environ["LOCAL_RANK"]),
         backend=dist.get_backend(group.device_group), communicator=type(group.device_communicator).__name__)
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    with (Path("/project/artifacts/logs/phase13_3_1_checkpoint_inventory.csv")).open() as file:
        audited = {(r["rank"], r["parameter"]): r for r in csv.DictReader(file) if r["layer"] == "0"}

    def record(name, weight, mapping):
        full = f"model.layers.0.self_attn.{name}"
        digest = checksum(weight)
        if tp == 8:
            expected = audited[str(rank), full]["local_sha256"]
        else:
            expected = checksum(load_weight(MODEL, index, 0, name, rank, tp))
        assert digest == expected, full
        emit("weight", **fields, parameter=full, shape=list(weight.shape), mapping=mapping, sha256=digest)

    torch.mlu.reset_peak_memory_stats(device)
    start = time.perf_counter()
    layer = MLAProjection(MODEL, 0, group, device, LIB, record)
    emit("load", **fields, verification_included=True, host_ms=(time.perf_counter()-start)*1000)
    bridge = LIB / "attention/libnative_mlu.so"
    norm = NativeRMSNorm(bridge)
    gamma_name = "model.layers.0.input_layernorm.weight"
    with safe_open(MODEL / index[gamma_name], framework="pt", device="cpu") as file:
        input_gamma = file.get_tensor(gamma_name).to(device)
    if args.mode == "reference":
        assert tp == 1
        attention = NativeMaterializedAttention(bridge)
        weights = {name: load_weight(MODEL, index, 0, name, 0, 1).to(device).float() for name in
                   ("q_proj.weight", "kv_a_proj_with_mqa.weight", "kv_b_proj.weight", "o_proj.weight")}
    failures = 0
    try:
        for idx, length in enumerate((1, 7, 17, 65)):
            directory = ROOT / "native-dump-yarn" / f"request{idx}"
            prompt = (directory / "prompt.ids").read_bytes()
            assert prompt == (HISTORY / f"prompts/{length}.bin").read_bytes()
            for step in range(4):
                path = directory / f"{step}-embedding.bin"
                residual = bf16(path, (-1, 2048)).to(device)
                hidden = norm.forward_native(residual, input_gamma, 1e-6)
                meta = dict(**fields, length=length, step=step, input_shape=list(hidden.shape),
                            fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            prompt_sha256=hashlib.sha256(prompt).hexdigest())
                refs = args.output / f"boundary_{length}_{step}.pt"
                q = layer.query(hidden)
                latent, rotary = layer.latent(hidden)
                normalized = layer.latent_norm(latent)
                expanded = layer.expand(normalized)
                if args.mode == "reference":
                    av = frozen_value_input(q, directory, step, length, device, attention)
                    reduced, rounded, added = layer.output(av, residual)
                    for name, actual, expected in (
                        ("q_fp32", q, hidden.float() @ weights["q_proj.weight"].T),
                        ("kv_a_fp32", torch.cat((latent,rotary),1), hidden.float() @ weights["kv_a_proj_with_mqa.weight"].T),
                        ("kv_b_fp32", expanded.flatten(1), normalized.float() @ weights["kv_b_proj.weight"].T),
                        ("o_fp32", reduced, av.float() @ weights["o_proj.weight"].T),
                    ):
                        failures += compare(actual, expected.cpu(), meta, name)
                    torch.save({name: value.cpu() for name,value in
                                dict(q=q, latent=latent, rotary=rotary, normalized=normalized,
                                     expanded=expanded, av=av, reduced=reduced, rounded=rounded,
                                     added=added).items()}, refs)
                else:
                    expected = torch.load(refs, map_location="cpu")
                    q_heads, kv_heads = 3072//tp, 16//tp
                    value = expected["av"][:, rank*(2048//tp):(rank+1)*(2048//tp)].contiguous().to(device)
                    reduced, rounded, added = layer.output(value, residual)
                    exact = expected["reduced"].to(device)
                    unequal = (rounded != expected["rounded"].to(device)).flatten()
                    if unequal.any():
                        examples = unequal.nonzero().flatten()[:8].tolist()
                        emit("rounding", **meta, indexes=examples,
                             tp_fp32=reduced.flatten()[examples].tolist(),
                             ref_fp32=exact.flatten()[examples].tolist(),
                             tp_bf16=rounded.flatten()[examples].tolist(),
                             ref_bf16=expected["rounded"].flatten()[examples].tolist())
                    summed = group.all_reduce(layer.o.gemm(value))
                    if not torch.equal(summed, reduced):
                        raise AssertionError("Explicit CNCL FP32 sum differs from Row result")
                    kv_local = expected["expanded"][:, rank*kv_heads:(rank+1)*kv_heads]
                    expanded_global = layer.kv_b.gather(expanded.flatten(1)).view_as(expected["expanded"])
                    for name, actual, target in (
                        ("q_local", q, expected["q"][:, rank*q_heads:(rank+1)*q_heads]),
                        ("q_head_order", layer.q.gather(q), expected["q"]),
                        ("q_rotary_input", q.view(-1,kv_heads,192)[...,128:].to(torch.bfloat16),
                         expected["q"][:,rank*q_heads:(rank+1)*q_heads].reshape(-1,kv_heads,192)[...,128:].to(torch.bfloat16)),
                        ("kv_a_latent", latent, expected["latent"]),
                        ("kv_a_shared_rotary", rotary, expected["rotary"]),
                        ("kv_a_norm", normalized, expected["normalized"]),
                        ("kv_b_k_nope", expanded[..., :128], kv_local[..., :128]),
                        ("kv_b_v", expanded[..., 128:], kv_local[..., 128:]),
                        ("kv_b_head_order", expanded_global, expected["expanded"]),
                        ("o_partial_fp32", layer.o.gemm(value),
                         (value.float() @ load_weight(MODEL,index,0,"o_proj.weight",rank,tp).to(device).float().T).cpu()),
                        ("o_cncl_fp32", reduced, expected["reduced"]),
                        ("o_cast_bf16", rounded, expected["rounded"]),
                        ("residual_bf16", added, expected["added"]),
                    ):
                        failures += compare(actual, target, meta, name)
                    if length == 17 and step == 0:
                        benchmark(layer, group, hidden, value, residual, meta, args.repeats, device)
                frozen = bf16(HISTORY / f"reference{length}" / f"{step}-layer0-attention.bin", (-1,2048))
                failures += compare(rounded, frozen, meta, "o_frozen_12_5_4")
    finally:
        layer.close()
    emit("result", **fields, mode=args.mode, tolerance_mismatches=failures)
    if failures:
        raise AssertionError(f"rank {rank}: {failures} numerical mismatches")


def device_kernel_ms(gemm, x, norm=None):
    stream = torch.mlu.current_stream()
    output = x.new_empty((x.shape[0], x.shape[1] if norm else gemm.packed.n),
                         dtype=torch.bfloat16 if norm else torch.float32)
    start, end = torch.mlu.Event(enable_timing=True), torch.mlu.Event(enable_timing=True)
    start.record()
    if norm:
        status = norm.launch(stream.mlu_stream, x.data_ptr(), gemm.data_ptr(), output.data_ptr(),
                             None, None, x.shape[0], x.shape[1], 1e-6)
    else:
        packed = gemm.packed
        assert x.is_contiguous() and x.shape[1] == packed.k
        status = packed.library.phase124_project(
            stream.mlu_stream, x.data_ptr(), output.data_ptr(), packed.handle, x.shape[0])
    end.record()
    end.synchronize()
    if status:
        raise RuntimeError(f"Timing launch CNRT status={status}")
    return start.elapsed_time(end)


def benchmark(layer, group, x, value, residual, meta, count, device):
    def chain():
        query = layer.query(x)
        latent, rotary = layer.latent(x)
        normalized = layer.latent_norm(latent)
        expanded = layer.expand(normalized)
        reduced, rounded, added = layer.output(value, residual)
        return query, latent, rotary, normalized, expanded, reduced, rounded, added

    first = chain()
    packed = [part.packed for part in (layer.q.gemm, layer.kv_a, layer.kv_b.gemm, layer.o.gemm)]
    native_bytes = sum(2 * ((p.n+127)//128*128) * ((p.k+127)//128*128) for p in packed)
    kernel = {
        "q": lambda: device_kernel_ms(layer.q.gemm, x),
        "kv_a": lambda: device_kernel_ms(layer.kv_a, x),
        "kv_norm": lambda: device_kernel_ms(layer.gamma, latent.to(torch.bfloat16), layer.norm),
        "kv_b": lambda: device_kernel_ms(layer.kv_b.gemm, normalized),
        "o_gemm": lambda: device_kernel_ms(layer.o.gemm, value),
    }
    for repeat in range(count):
        ops = {
            "q": lambda: layer.query(x),
            "kv_a": lambda: layer.latent(x),
            "kv_norm": lambda: layer.latent_norm(latent),
            "kv_b": lambda: layer.expand(normalized),
            "o_gemm": lambda: layer.o.gemm(value),
            "allreduce": lambda: group.all_reduce(partial),
            "chain": chain,
        }
        for name, fn in ops.items():
            if name == "allreduce":
                partial = local.clone()
            result, host, event = timed(fn, device)
            if name == "kv_a":
                latent = result[0]
            elif name == "kv_norm":
                normalized = result
            elif name == "o_gemm":
                local = result
            elif name == "chain" and any(not torch.equal(a,b) for a,b in zip(first,result)):
                raise AssertionError("Repeated projection outputs changed")
            elif name == "chain":
                emit("stability", **meta, repeat=repeat)
            emit("performance", **meta, repeat=repeat, op=name, host_ms=host, stream_ms=event,
                 allocator_peak=torch.mlu.max_memory_allocated(device), native_bytes=native_bytes,
                 kernel_ms=kernel[name]() if name in kernel else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("reference", "test"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert rank == int(os.environ["LOCAL_RANK"]) and torch.mlu.device_count() == tp
    device = torch.device("mlu", rank)
    torch.mlu.set_device(device)
    torch.backends.mlu.matmul.allow_tf32 = False
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
            emit("cleanup", rank=rank, tp=tp, mode=args.mode)


if __name__ == "__main__":
    main()
