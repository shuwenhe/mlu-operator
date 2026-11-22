"""Generate 32 real tokens and record timing from vendor engine steps."""
from array import array
import csv
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import torch_mlu  # noqa: F401
from vllm import LLM, SamplingParams

from .plugin import ARCH, register


OUT = Path("/results")
FIXTURE = Path("/project/artifacts/build/phase12_5_4/native-dump-yarn-scale/request2")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def memory(phase):
    rows = []
    for rank in range(8):
        free, total = torch.mlu.mem_get_info(rank)
        rows.append(dict(phase=phase, card=rank, used_bytes=total - free, total_bytes=total))
    return rows


def main():
    if torch.mlu.device_count() != 8:
        raise RuntimeError("Expected eight MLU590 devices")
    register()
    os.environ["NEURX_TP8_LOGIT_FIXTURE"] = str(FIXTURE)
    os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "30")
    records = memory("baseline")
    llm = LLM(model="/checkpoint", trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=8,
              distributed_executor_backend="mp", hf_overrides={"architectures": [ARCH]},
              enforce_eager=True, max_model_len=128, block_size=16, max_num_seqs=1,
              max_num_batched_tokens=128, num_gpu_blocks_override=16, gpu_memory_utilization=0.45,
              enable_prefix_caching=False, disable_log_stats=True, disable_custom_all_reduce=True)
    engine = llm.llm_engine
    core = engine.engine_core
    workers = [handle.proc for handle in core.engine_core.model_executor.workers]
    try:
        if len(workers) != 8:
            raise ValueError(f"Expected eight workers, got {len(workers)}")
        records += memory("loaded")
        prompt = array("I")
        prompt.frombytes((FIXTURE / "prompt.ids").read_bytes())
        start = perf_counter()
        engine.add_request("stage-c1", {"prompt_token_ids": list(prompt)},
                           SamplingParams(temperature=0, max_tokens=32, ignore_eos=True))
        timestamps = []
        tokens = []
        while engine.has_unfinished_requests():
            for output in engine.step():
                current = list(output.outputs[0].token_ids)
                if len(current) == len(tokens):
                    continue
                if len(current) != len(tokens) + 1 or current[:-1] != tokens:
                    raise RuntimeError("Expected one new real token per engine step")
                tokens = current
                timestamps.append(perf_counter())
        if len(tokens) != 32:
            raise RuntimeError(f"Expected 32 generated tokens, got {len(tokens)}")
        expected = []
        for step in range(4):
            reference = array("f")
            reference.frombytes((FIXTURE / f"{step}-logits.bin").read_bytes())
            expected.append(max(range(len(reference)), key=reference.__getitem__))
        if tokens[:4] != expected:
            raise ValueError(f"Frozen greedy parity failed: {tokens[:4]} != {expected}")
        records += memory("generated")
        (OUT / "generation_outputs.json").write_text(
            json.dumps(dict(prompt_token_ids=list(prompt), generated_token_ids=tokens,
                            frozen_greedy_prefix=expected, text=output.outputs[0].text), indent=2) + "\n"
        )
        write_csv("performance.csv", [dict(prompt_tokens=len(prompt), generated_tokens=len(tokens),
                                           ttft_ms=(timestamps[0] - start) * 1000,
                                           tpot_ms=(timestamps[-1] - timestamps[0]) * 1000 / 31,
                                           throughput_tokens_per_s=32 / (timestamps[-1] - start))])
    finally:
        core.shutdown(timeout=120)
        lifecycle = [dict(pid=proc.pid, exit_code=proc.exitcode, alive=proc.is_alive()) for proc in workers]
        write_csv("lifecycle.csv", lifecycle)
        write_csv("memory_per_card.csv", records + memory("shutdown"))
        if len(lifecycle) != 8 or any(row["alive"] or row["exit_code"] != 0 for row in lifecycle):
            raise RuntimeError(f"TP8 shutdown was not graceful: {lifecycle}")
        print(json.dumps(dict(stage_c1="engine_shutdown")), flush=True)


if __name__ == "__main__":
    main()
