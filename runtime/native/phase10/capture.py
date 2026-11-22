"""Capture original eager MLA with genuine BF16 and FP16 inputs."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p0"))
from checkpoint_trace import Cache, names


def save(path, tensor):
    tensor.contiguous().view(torch.int16).numpy().tofile(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "weights", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--full-sequence-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    source = args.checkpoint / "modeling_deepseek.py"
    content = source.read_text()
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    cfg["_attn_implementation"] = "eager"
    retained = [node for node in ast.parse(content).body if
                isinstance(node, (ast.ClassDef, ast.FunctionDef)) and
                (node.name in names or node.name.startswith("yarn_"))]
    ctx = dict(torch=torch, nn=nn, F=F, math=math, warnings=warnings, Optional=Optional,
               Tuple=Tuple, DeepseekV2Config=SimpleNamespace, Cache=Cache,
               ACT2FN={"silu": F.silu}, ATTENTION_CLASSES={})
    exec(compile(ast.Module(body=retained, type_ignores=[]), str(source), "exec"), ctx)
    lines = content.splitlines()
    masked_line = next(i for i, line in enumerate(lines, 1) if
                       "# upcast attention to fp32" in line) + 1
    av_line = next(i for i, line in enumerate(lines, 1) if
                   line.strip() == "attn_output = torch.matmul(attn_weights, value_states)")
    checkpoint_weights = {
        "q": "q_proj.weight", "kv_a": "kv_a_proj_with_mqa.weight",
        "kv_b": "kv_b_proj.weight", "gamma": "kv_a_layernorm.weight",
        "o": "o_proj.weight",
    }
    raw_tokens = (args.weights / "tokens.bin").read_bytes()
    tokens = [torch.frombuffer(bytearray(raw_tokens[i*4100+4:(i+1)*4100]),
                               dtype=torch.bfloat16) for i in range(6)]
    torch.set_num_threads(2)
    stages = {}

    def trace(frame, event, arg):
        if event == "call":
            if frame.f_code.co_filename == str(source) and frame.f_code.co_name == "forward":
                return trace
            return None
        if event == "line" and frame.f_lineno == masked_line:
            live = frame.f_locals
            stages.update(q=live["query_states"][0], k=live["key_states"][0],
                          v=live["value_states"][0], masked=live["attn_weights"][0])
        if event == "line" and frame.f_lineno == av_line:
            stages["prob"] = frame.f_locals["attn_weights"][0]
        return trace

    with torch.no_grad():
        for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            root = args.output / name
            root.mkdir(parents=True, exist_ok=True)
            attn = ctx["DeepseekV2Attention"](SimpleNamespace(**cfg), 0).eval().to(dtype)
            norm = ctx["DeepseekV2RMSNorm"](2048, eps=cfg["rms_norm_eps"]).eval().to(dtype)
            norm.weight.copy_(torch.frombuffer(bytearray((args.weights/"input_gamma.bin").read_bytes()),
                                               dtype=torch.bfloat16))
            tensors = attn.state_dict()
            for alias, param in checkpoint_weights.items():
                ref = tensors[param]
                ref.copy_(torch.frombuffer(bytearray((args.weights/f"{alias}.bin").read_bytes()),
                                           dtype=torch.bfloat16).reshape_as(ref))
                if not args.full_sequence_only and not args.benchmark:
                    save(root / f"weight_{alias}.bin", ref.T if ref.ndim == 2 else ref)
            for tag, layer in (("q_projection",attn.q_proj), ("kv_a",attn.kv_a_proj_with_mqa),
                               ("latent",attn.kv_a_layernorm), ("kv_b",attn.kv_b_proj)):
                layer.register_forward_hook(lambda _, inp, out, key=tag: stages.update({key: out}))
            attn.o_proj.register_forward_pre_hook(lambda _, inp: stages.update(merged=inp[0]))
            cos, sin = attn.rotary_emb(torch.zeros(1, dtype=dtype), seq_len=131)
            if not args.full_sequence_only and not args.benchmark:
                save(root/"cos.bin", cos)
                save(root/"sin.bin", sin)
                (root/"scale.txt").write_text(repr(attn.softmax_scale)+"\n")
            for n in (1, 7, 32, 128):
                for request in (0, 1):
                    if args.full_sequence_only:
                        count = n + 3
                        hidden = norm(torch.stack([tokens[3*request+t%3]
                                                   for t in range(count)])[None].to(dtype))
                        pos = torch.arange(count)[None]
                        mask = torch.zeros((1, 1, count, count), dtype=dtype)
                        mask.masked_fill_(pos > pos.T, -torch.inf)
                        result = attn(hidden, attention_mask=mask, position_ids=pos,
                                      past_key_value=Cache(), use_cache=True)[0]
                        save(root / str(n) / str(request) / "full_output.bin", result)
                        continue
                    if args.benchmark:
                        for sample in range(4):
                            cache = Cache()
                            cursor = 0
                            for count in (n, 1, 1, 1):
                                hidden = norm(torch.stack([tokens[3*request+t%3]
                                                           for t in range(cursor,cursor+count)])[None].to(dtype))
                                pos = torch.arange(cursor,cursor+count)[None]
                                mask = torch.zeros((1, 1, count, cursor+count), dtype=dtype)
                                mask.masked_fill_(torch.arange(cursor+count)[None,:] > pos.T, -torch.inf)
                                start = time.perf_counter_ns()
                                attn(hidden, attention_mask=mask, position_ids=pos,
                                     past_key_value=cache, use_cache=True)
                                elapsed = (time.perf_counter_ns() - start) / 1000
                                if sample:
                                    print(f"{name},{n},{request},{cursor},{count},{sample},{elapsed:.3f}")
                                cursor += count
                        continue
                    cache = Cache()
                    cursor = 0
                    for count in (n, 1, 1, 1):
                        path = root / str(n) / str(request) / str(cursor)
                        path.mkdir(parents=True, exist_ok=True)
                        hidden = norm(torch.stack([tokens[3*request+t%3]
                                                   for t in range(cursor,cursor+count)])[None].to(dtype))
                        positions = torch.arange(cursor, cursor + count)[None]
                        mask = torch.zeros((1, 1, count, cursor+count), dtype=dtype)
                        mask.masked_fill_(torch.arange(cursor+count)[None,:] > positions.T, -torch.inf)
                        stages.clear()
                        sys.settrace(trace)
                        try:
                            result = attn(hidden, attention_mask=mask, position_ids=positions,
                                          past_key_value=cache, use_cache=True)[0]
                        finally:
                            sys.settrace(None)
                        stages.update(input=hidden, output=result)
                        assert len(stages) == 12, stages.keys()
                        for stage, values in stages.items():
                            save(path/f"{stage}.bin", values)
                        print(f"{name} requests={request} prefill={n} offset={cursor} count={count}")
                        cursor += count
    if args.full_sequence_only or args.benchmark:
        return
    (args.output/"provenance.json").write_text(json.dumps({
        "checkpoint_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "torch":torch.__version__, "capability":torch.backends.cpu.get_cpu_capability(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
