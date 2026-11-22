"""Trace the installed vendor engine as an explicit scheduler control run."""
import json
import os
from functools import wraps

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch_mlu
from vllm import LLM, SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager


def log(event, **fields):
    print(json.dumps({"backend": "vendor", "event": event, **fields}, default=str), flush=True)


def enable_trace():
    from vllm_mlu.v1.core.sched.scheduler import SchedulerWithProfiler

    schedule = SchedulerWithProfiler.schedule
    free = KVCacheManager.free

    @wraps(schedule)
    def traced_schedule(self, *args, **kwargs):
        output = schedule(self, *args, **kwargs)
        log(
            "schedule",
            scheduled_tokens=output.num_scheduled_tokens,
            new=[{"request": req.req_id, "blocks": req.block_ids}
                 for req in output.scheduled_new_reqs],
            cached=output.scheduled_cached_reqs.req_ids,
            finished=sorted(output.finished_req_ids),
        )
        return output

    @wraps(free)
    def traced_free(self, request):
        result = free(self, request)
        log("cache_free", request=request.request_id)
        return result

    SchedulerWithProfiler.schedule = traced_schedule
    KVCacheManager.free = traced_free


def main():
    engine = LLM(model="/checkpoint", trust_remote_code=True, dtype="bfloat16",
                 max_model_len=128, max_num_seqs=4, max_num_batched_tokens=128,
                 gpu_memory_utilization=0.45, enforce_eager=True,
                 enable_prefix_caching=False, disable_log_stats=True)
    enable_trace()
    for prompts in (["The capital of France is"],
                    ["The capital of France is",
                     "Explain why the history of mathematics matters. " * 8]):
        outputs = engine.generate(
            prompts, SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), use_tqdm=False
        )
        for result in outputs:
            log("tokens", request=result.request_id, prompt_length=len(result.prompt_token_ids),
                token_ids=result.outputs[0].token_ids)


if __name__ == "__main__":
    main()
