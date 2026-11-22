"""Run comparable Stage C.2 and Stage C.3 cold and resident-weight requests."""
import argparse
from array import array
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import torch_mlu  # noqa: F401
from vllm import LLM

from ..stage_c2.benchmark import run_request
from ..tp8.inference import FIXTURE, memory, write_csv
from .plugin import ARCH, register


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="")
    parser.add_argument("--warm-requests", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    os.environ.update(NEURX_C2_VARIANT="candidate", NEURX_C3_EXPERIMENT=args.experiment,
                      NEURX_C3_PROFILE=str(int(args.profile)), NEURX_C2_CAPTURE_LOGITS=str(int(args.profile)),
                      NEURX_TP8_LOGIT_FIXTURE=str(FIXTURE))
    os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "30")
    if torch.mlu.device_count() != 8:
        raise RuntimeError("Expected eight MLU devices")
    register()
    gold = json.loads(Path("/project/artifacts/logs/phase15/stage_c1/generation_outputs.json").read_text())
    prompt = array("I")
    prompt.frombytes((FIXTURE / "prompt.ids").read_bytes())
    if list(prompt) != gold["prompt_token_ids"] or len(gold["generated_token_ids"]) != 32:
        raise ValueError("Invalid Stage C.1 prompt or tokens")
    snapshots = memory("baseline")
    start = perf_counter()
    llm = LLM(model="/checkpoint", trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=8,
              distributed_executor_backend="mp", hf_overrides={"architectures": [ARCH]},
              enforce_eager=True, max_model_len=128, block_size=16, max_num_seqs=1,
              max_num_batched_tokens=128, num_gpu_blocks_override=16, gpu_memory_utilization=0.45,
              enable_prefix_caching=False, disable_log_stats=True, disable_custom_all_reduce=True)
    init_ms = (perf_counter() - start) * 1000
    engine = llm.llm_engine
    core = engine.engine_core
    workers = [handle.proc for handle in core.engine_core.model_executor.workers]
    measurements, intervals = [], []
    try:
        snapshots += memory("loaded")
        for index in range(args.warm_requests + 1):
            row, times = run_request(engine, index, list(prompt), gold["generated_token_ids"])
            row.update(variant=args.experiment or "baseline", engine_init_ms=init_ms)
            measurements.append(row)
            intervals.extend(dict(request=index, token=token, interval_ms=ms)
                             for token, ms in enumerate(times, 2))
            snapshots += memory(f"request_{index}")
            print(json.dumps(row), flush=True)
        write_csv("performance.csv", measurements)
        write_csv("token_intervals.csv", intervals)
    finally:
        core.shutdown(timeout=120)
        lifecycle = [dict(pid=proc.pid, exit_code=proc.exitcode, alive=proc.is_alive()) for proc in workers]
        write_csv("lifecycle.csv", lifecycle)
        write_csv("memory_per_card.csv", snapshots + memory("shutdown"))
        if len(lifecycle) != 8 or any(row["alive"] or row["exit_code"] != 0 for row in lifecycle):
            raise RuntimeError(f"TP8 shutdown failed: {lifecycle}")


if __name__ == "__main__":
    main()
