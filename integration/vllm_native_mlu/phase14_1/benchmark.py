"""Actual LLM.generate timing on the already accepted frozen length-one fixture."""
import argparse
import json
from pathlib import Path
import time

import torch_mlu
from vllm import LLM, SamplingParams

from .engine import ARCH, close_model, expected_token, prompt


def rank_times(worker):
    import torch
    from integration.vllm_native_mlu.phase13_4.full_model import allocated_pack_bytes

    model=worker.model_runner.model
    steps=model.benchmark_steps
    model.benchmark_steps=[]
    return dict(rank=model.rank,steps=steps,
                peak_allocator_bytes=torch.mlu.max_memory_allocated(model.device),
                native_pack_bytes=allocated_pack_bytes(model.mla,model.decoder))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tp",type=int,choices=(1,2),required=True)
    args=parser.parse_args()
    arch=ARCH
    if args.tp==1:
        from ..phase12_5.run import ARCHITECTURE,register
        register()
        arch=ARCHITECTURE
    fixture=Path("/project/artifacts/build/phase12_5_4/native-dump-yarn-scale")
    ids,reference=prompt(fixture,1)
    expected=[expected_token(reference/f"{i}-logits.bin") for i in range(2)]
    llm=LLM(model="/checkpoint",trust_remote_code=True,dtype="bfloat16",
            tensor_parallel_size=args.tp,hf_overrides={"architectures":[arch]},
            enforce_eager=True,max_model_len=128,block_size=16,max_num_seqs=2,
            max_num_batched_tokens=128,num_gpu_blocks_override=32,enable_prefix_caching=False,
            gpu_memory_utilization=0.45,disable_log_stats=False,disable_custom_all_reduce=True)
    try:
        params=SamplingParams(temperature=0,max_tokens=2,ignore_eos=True)
        for repeat in range(-3,100):
            start=time.perf_counter()
            result,=llm.generate([{"prompt_token_ids":ids}],params,use_tqdm=False)
            wall=(time.perf_counter()-start)*1000
            if list(result.outputs[0].token_ids)!=expected:
                raise ValueError(f"Frozen benchmark token mismatch at {repeat}")
            metrics=result.metrics
            if metrics is None or not 0<metrics.scheduled_ts<=metrics.first_token_ts<=metrics.last_token_ts:
                raise ValueError("Vendor engine did not expose valid request timings")
            ranks=llm.collective_rpc(rank_times) if args.tp==2 else []
            if args.tp==2 and any(len(row["steps"])!=2 for row in ranks):
                raise ValueError("Expected both worker forward timings")
            if repeat>=0:
                print(json.dumps(dict(phase14_1="benchmark",tp=args.tp,repeat=repeat,wall_ms=wall,
                                      ttft_ms=metrics.first_token_latency*1000,
                                      prefill_ms=(metrics.first_token_ts-metrics.scheduled_ts)*1000,
                                      tpot_ms=(metrics.last_token_ts-metrics.first_token_ts)*1000,
                                      tokens_per_second=2000/wall,ranks=ranks)),flush=True)
    finally:
        try:
            if args.tp==2:
                llm.collective_rpc(close_model)
        finally:
            llm.llm_engine.engine_core.shutdown()


if __name__=="__main__":
    main()
