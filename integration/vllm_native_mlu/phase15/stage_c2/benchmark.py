"""Measure identical cold and resident-weight TP8 requests in one engine."""
import argparse
from array import array
import json
import os
from pathlib import Path
from statistics import mean
from math import ceil
from time import perf_counter

import torch
import torch_mlu  # noqa: F401
from vllm import LLM, SamplingParams

from ..tp8.inference import FIXTURE, OUT, memory, write_csv
from ..tp8.plugin import ARCH, register


def run_request(engine, index, prompt, expected):
    start = perf_counter()
    engine.add_request(f"stage-c2-{index}", {"prompt_token_ids": prompt},
                       SamplingParams(temperature=0, max_tokens=32, ignore_eos=True))
    submitted = perf_counter()
    timestamps, step_ms, tokens = [], [], []
    while engine.has_unfinished_requests():
        step_start = perf_counter()
        outputs = engine.step()
        now = perf_counter()
        step_ms.append((now - step_start) * 1000)
        for output in outputs:
            current = list(output.outputs[0].token_ids)
            if current == tokens:
                continue
            if len(current) != len(tokens) + 1 or current[:-1] != tokens:
                raise RuntimeError("Expected one new token per engine step")
            tokens = current
            timestamps.append(now)
    if tokens != expected:
        raise ValueError(f"Stage C.1 token mismatch in request {index}: {tokens}")
    intervals = [(b - a) * 1000 for a, b in zip(timestamps, timestamps[1:])]
    elapsed = timestamps[-1] - start
    return dict(request=index, phase="warm" if index else "cold", prompt_tokens=len(prompt),
                generated_tokens=len(tokens), ttft_ms=(timestamps[0] - start) * 1000,
                tpot_mean_ms=mean(intervals), tpot_p95_ms=sorted(intervals)[ceil(.95 * len(intervals)) - 1],
                decode_tokens_s=1000 / mean(intervals), end_to_end_tokens_s=len(tokens) / elapsed,
                submission_ms=(submitted-start)*1000, engine_step_ms=sum(step_ms)), intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "profile", "candidate"), default="baseline")
    parser.add_argument("--warm-requests", type=int, default=2)
    parser.add_argument("--capture-logits", action="store_true")
    parser.add_argument("--profile-components", action="store_true")
    args = parser.parse_args()
    if args.capture_logits:
        os.environ["NEURX_C2_CAPTURE_LOGITS"] = "1"
    if args.profile_components:
        os.environ["NEURX_C2_PROFILE"] = "1"
    arch = ARCH
    if args.variant == "baseline":
        register()
    else:
        from .plugin import ARCH as PROFILE_ARCH, register as register_profile
        register_profile()
        arch = PROFILE_ARCH
    os.environ["NEURX_C2_VARIANT"] = args.variant
    os.environ["NEURX_TP8_LOGIT_FIXTURE"] = str(FIXTURE)
    os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "30")
    if torch.mlu.device_count() != 8:
        raise RuntimeError("Expected eight MLU devices")
    stage_c1 = Path("/project/artifacts/logs/phase15/stage_c1/generation_outputs.json")
    gold = json.loads(stage_c1.read_text())
    prompt = array("I")
    prompt.frombytes((FIXTURE / "prompt.ids").read_bytes())
    if list(prompt) != gold["prompt_token_ids"] or len(gold["generated_token_ids"]) != 32:
        raise ValueError("Invalid Stage C.1 token fixture")
    records = memory("baseline")
    load_start = perf_counter()
    llm = LLM(model="/checkpoint", trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=8,
              distributed_executor_backend="mp", hf_overrides={"architectures": [arch]},
              enforce_eager=True, max_model_len=128, block_size=16, max_num_seqs=1,
              max_num_batched_tokens=128, num_gpu_blocks_override=16, gpu_memory_utilization=0.45,
              enable_prefix_caching=False, disable_log_stats=True, disable_custom_all_reduce=True)
    load_ms = (perf_counter() - load_start) * 1000
    engine = llm.llm_engine
    core = engine.engine_core
    workers = [handle.proc for handle in core.engine_core.model_executor.workers]
    schedule_ms = []
    if args.variant == "profile" or args.profile_components:
        scheduler = core.engine_core.scheduler
        schedule = scheduler.schedule

        def timed_schedule(*args, **kwargs):
            start = perf_counter()
            result = schedule(*args, **kwargs)
            elapsed = (perf_counter() - start) * 1000
            if result.num_scheduled_tokens:
                schedule_ms.append(elapsed)
            return result

        scheduler.schedule = timed_schedule
    measurements, intervals = [], []
    try:
        records += memory("loaded")
        for index in range(1 + args.warm_requests):
            result, token_intervals = run_request(engine, index, list(prompt), gold["generated_token_ids"])
            result.update(variant=args.variant, engine_init_ms=load_ms)
            measurements.append(result)
            intervals.extend(dict(request=index, token=token, interval_ms=ms)
                             for token, ms in enumerate(token_intervals, 2))
            records += memory(f"request_{index}")
            print(json.dumps(result), flush=True)
        write_csv("performance.csv", measurements)
        write_csv("token_intervals.csv", intervals)
        if schedule_ms:
            write_csv("host_schedule.csv", [dict(step=i, milliseconds=ms) for i, ms in enumerate(schedule_ms)])
    finally:
        core.shutdown(timeout=120)
        lifecycle = [dict(pid=proc.pid, exit_code=proc.exitcode, alive=proc.is_alive()) for proc in workers]
        write_csv("lifecycle.csv", lifecycle)
        write_csv("memory_per_card.csv", records + memory("shutdown"))
        if len(lifecycle) != 8 or any(row["alive"] or row["exit_code"] != 0 for row in lifecycle):
            raise RuntimeError(f"TP8 shutdown failed: {lifecycle}")


if __name__ == "__main__":
    main()
