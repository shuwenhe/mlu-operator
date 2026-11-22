"""Vendor vLLM compatibility baseline. This test does not select mlu-operator."""
import time
import torch_mlu
from vllm import LLM, SamplingParams


def main():
    begin=time.monotonic()
    engine=LLM(model="/checkpoint", trust_remote_code=True, dtype="bfloat16",
               max_model_len=128, max_num_seqs=4, max_num_batched_tokens=128,
               gpu_memory_utilization=0.45, enforce_eager=True)
    print("vendor_initialization_seconds",time.monotonic()-begin,flush=True)
    begin=time.monotonic()
    result=engine.generate(["The capital of France is","The history of mathematics"],
                           SamplingParams(temperature=0,max_tokens=4),use_tqdm=False)
    print("vendor_generation_seconds",time.monotonic()-begin,flush=True)
    print([(r.outputs[0].token_ids,r.outputs[0].text) for r in result],flush=True)


if __name__=="__main__":
    main()
