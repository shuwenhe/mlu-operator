"""Run a single native BF16 TP8 request and record its worker evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from math import ceil
from time import perf_counter

import torch
import torch.distributed as dist
import torch_mlu  # noqa: F401
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import get_tp_group
from vllm_mlu.compilation.mlu_graph import MLUGraphWrapper

from .stage_c2 import plugin as stage_c2
from .stage_c2.model import StageC2Model
from .stage_c2.output_projection import OutputAxisProjection
from .stage_c3 import plugin as stage_c3
from .stage_c3.model import StageC3Model
from .stage_c3.moe import TunableDecoder
from .tp8 import plugin as stage_c1
from .tp8.inference import memory, write_csv
from .tp8.model import ExpertParallelDecoder, NeurXDeepSeekV2ForCausalLMTP8


VARIANT_EXPERIMENT = {
    "stage_c1": "baseline",
    "stage_c2": "output-axis",
    "stage_c3": "fused,batched",
}


def worker_identity(worker):
    wrapped = worker.model_runner.model
    model = wrapped.runnable if type(wrapped) is MLUGraphWrapper else wrapped
    group = get_tp_group()
    decoder = model.decoder
    return dict(rank=group.rank_in_group, world_size=group.world_size,
                device=str(model.device), current_device=torch.mlu.current_device(),
                backend=dist.get_backend(group.device_group), pid=os.getpid(),
                model_class=qualified(type(model)),
                runner_class=qualified(type(wrapped)),
                decoder_class=qualified(type(decoder)),
                output_axis_layers=sum(type(layer.o) is OutputAxisProjection for layer in model.mla),
                fused=getattr(decoder, "fuse", False), batched=getattr(decoder, "batch", False),
                compact=getattr(decoder, "compact", False), overlap=getattr(decoder, "overlap", False))


def qualified(kind):
    return f"{kind.__module__}.{kind.__name__}"


def verify_workers(rows, model_kind, decoder_kind):
    if [row["rank"] for row in rows] != list(range(8)):
        raise RuntimeError(f"Expected TP8 ranks 0–7: {rows}")
    candidate = model_kind is StageC3Model
    output_axis = model_kind in (StageC2Model, StageC3Model)
    for rank, row in enumerate(rows):
        if (
            row["world_size"] != 8 or row["current_device"] != rank or row["device"] != f"mlu:{rank}"
            or row["backend"] != "cncl" or row["model_class"] != qualified(model_kind)
            or row["decoder_class"] != qualified(decoder_kind)
            or row["output_axis_layers"] != (27 if output_axis else 0)
            or row["fused"] != candidate or row["batched"] != candidate or row["compact"] or row["overlap"]
        ):
            raise RuntimeError(f"Native worker identity mismatch: {row}")


def generate(engine, prompt_ids, count):
    start = perf_counter()
    engine.add_request("make-deepseek", {"prompt_token_ids": prompt_ids},
                       SamplingParams(temperature=0, max_tokens=count, ignore_eos=True))
    timestamps, tokens = [], []
    text = ""
    while engine.has_unfinished_requests():
        for result in engine.step():
            output = result.outputs[0]
            current = list(output.token_ids)
            if len(current) == len(tokens):
                continue
            if len(current) != len(tokens) + 1 or current[:-1] != tokens:
                raise RuntimeError("Expected one new token per engine step")
            tokens = current
            text = output.text
            timestamps.append(perf_counter())
    if len(tokens) != count:
        raise RuntimeError(f"Expected {count} generated tokens, got {len(tokens)}")
    intervals = [(b - a) * 1000 for a, b in zip(timestamps, timestamps[1:])]
    tpot = mean(intervals) if intervals else None
    stats = dict(prompt_tokens=len(prompt_ids), generated_tokens=len(tokens),
                 ttft_ms=(timestamps[0] - start) * 1000, mean_tpot_ms=tpot,
                 p95_tpot_ms=sorted(intervals)[ceil(.95 * len(intervals)) - 1] if intervals else None,
                 decode_tokens_s=1000 / tpot if tpot else None,
                 end_to_end_tokens_s=len(tokens) / (timestamps[-1] - start))
    return stats, tokens, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--variant", choices=("stage_c1", "stage_c2", "stage_c3", "tp8"), default="stage_c3")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", default="/results")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--c3-experiment", default="")
    args = parser.parse_args()
    if args.output_dir != "/results":
        raise ValueError("Mount the evidence directory at /results")
    if args.tp != 8 or args.dtype != "bfloat16" or not 0 < args.max_tokens < 128:
        raise ValueError("Expected BF16 TP8 with 1–127 output tokens")
    variant = "stage_c1" if args.variant == "tp8" else args.variant
    experiment = args.experiment or args.c3_experiment or os.getenv("DEEPSEEK_EXPERIMENT", "")
    if not experiment:
        experiment = VARIANT_EXPERIMENT[variant]
    expected_experiment = VARIANT_EXPERIMENT[variant]
    if experiment != expected_experiment:
        raise ValueError(f"{variant} requires experiment={expected_experiment}, got {experiment}")
    if not torch.mlu.is_available() or torch.mlu.device_count() != 8:
        raise RuntimeError("Native inference requires exactly eight usable MLU devices")
    for rank in range(8):
        if torch.mlu.get_device_properties(rank).name != "MLU590-M9":
            raise RuntimeError(f"Unexpected MLU device at rank {rank}")

    variants = {
        "stage_c1": (stage_c1, NeurXDeepSeekV2ForCausalLMTP8, ExpertParallelDecoder),
        "stage_c2": (stage_c2, StageC2Model, ExpertParallelDecoder),
        "stage_c3": (stage_c3, StageC3Model, TunableDecoder),
    }
    plugin, model_kind, decoder_kind = variants[variant]
    env_update = {"NEURX_C2_PROFILE": "0", "NEURX_C2_CAPTURE_LOGITS": "0", "NEURX_C3_PROFILE": "0"}
    if variant in ("stage_c2", "stage_c3"):
        env_update["NEURX_C2_VARIANT"] = "candidate"
    if variant == "stage_c3":
        env_update["NEURX_C3_EXPERIMENT"] = experiment
    os.environ.update(env_update)
    os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "30")
    plugin.register()
    snapshots = memory("baseline")
    llm = LLM(model=args.model_dir, trust_remote_code=True, dtype=args.dtype, tensor_parallel_size=8,
              distributed_executor_backend="mp", hf_overrides={"architectures": [plugin.ARCH]},
              enforce_eager=True, max_model_len=128, block_size=16, max_num_seqs=1,
              max_num_batched_tokens=128, num_gpu_blocks_override=16, gpu_memory_utilization=0.45,
              enable_prefix_caching=False, disable_log_stats=True, disable_custom_all_reduce=True)
    engine = llm.llm_engine
    core = engine.engine_core
    workers = [handle.proc for handle in core.engine_core.model_executor.workers]
    try:
        rows = sorted(llm.collective_rpc(worker_identity, timeout=60), key=lambda row: row["rank"])
        write_csv("workers.csv", rows)
        verify_workers(rows, model_kind, decoder_kind)
        (Path(args.output_dir) / "runtime_configuration.json").write_text(
            json.dumps(
                {
                    "requested": {
                        "variant": variant,
                        "experiment": experiment,
                        "tp": args.tp,
                        "dtype": args.dtype,
                    },
                    "verified": {
                        "worker_count": len(rows),
                        "ranks": [row["rank"] for row in rows],
                        "devices": [row["device"] for row in rows],
                        "backends": sorted(set(row["backend"] for row in rows)),
                        "model_class": sorted(set(row["model_class"] for row in rows)),
                        "decoder_class": sorted(set(row["decoder_class"] for row in rows)),
                        "output_axis_layers": sorted(set(row["output_axis_layers"] for row in rows)),
                        "fused": sorted(set(row["fused"] for row in rows)),
                        "batched": sorted(set(row["batched"] for row in rows)),
                    },
                },
                indent=2,
            )
            + "\n"
        )
        snapshots += memory("loaded")
        prompt_ids = llm.get_tokenizer().encode(args.prompt)
        if not prompt_ids or len(prompt_ids) + args.max_tokens > 128:
            raise ValueError(f"Prompt has {len(prompt_ids)} tokens. Prompt and output must fit 128 tokens.")
        stats, tokens, text = generate(engine, prompt_ids, args.max_tokens)
        snapshots += memory("generated")
    finally:
        core.shutdown(timeout=120)
        lifecycle = [dict(pid=proc.pid, exit_code=proc.exitcode, alive=proc.is_alive()) for proc in workers]
        if lifecycle:
            write_csv("lifecycle.csv", lifecycle)
        write_csv("memory_per_card.csv", snapshots + memory("shutdown"))
        if len(lifecycle) != 8 or any(row["alive"] or row["exit_code"] != 0 for row in lifecycle):
            raise RuntimeError(f"TP8 shutdown failed: {lifecycle}")
    write_csv("performance.csv", [dict(variant=variant, dtype=args.dtype, tp=8, **stats)])
    generation = dict(variant=variant, architecture=plugin.ARCH, prompt=args.prompt,
                      prompt_token_ids=prompt_ids, generated_text=text, generated_token_ids=tokens)
    (Path(args.output_dir) / "generation_outputs.json").write_text(json.dumps([generation], indent=2) + "\n")


if __name__ == "__main__":
    main()
