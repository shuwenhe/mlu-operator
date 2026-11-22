"""Register the native architecture and submit requests to Vendor vLLM."""
from array import array
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch_mlu
from vllm import LLM, ModelRegistry, SamplingParams

ARCHITECTURE = "NeurXDeepSeekV2ForCausalLM"
_REGISTERED = False


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    ModelRegistry.register_model(
        ARCHITECTURE, "integration.vllm_native_mlu.phase12_5.model:NeurXDeepSeekV2ForCausalLM"
    )
    _REGISTERED = True


# Keep worker subprocess registries consistent under spawn-based TP execution.
register()


def main():
    register()
    print(f"native architecture={ARCHITECTURE}", flush=True)
    engine = LLM(model="/checkpoint", trust_remote_code=True, dtype="bfloat16",
                 hf_overrides={"architectures": [ARCHITECTURE]}, enforce_eager=True,
                 max_model_len=128, max_num_seqs=2, max_num_batched_tokens=128,
                 num_gpu_blocks_override=32, enable_prefix_caching=False,
                 gpu_memory_utilization=0.45, disable_log_stats=True)
    from ..phase12_3.vendor_trace import enable_trace
    enable_trace()
    text = "Explain why the history of mathematics matters. " * 10
    tokens = engine.get_tokenizer().encode(text)
    sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
    for lengths in ((1,), (7, 17), (65,), (1,)):
        prompt_root = os.environ.get("NATIVE_DEEPSEEK_PROMPTS")
        prompts = []
        for length in lengths:
            ids = tokens[:length]
            if prompt_root:
                data = array("I")
                data.frombytes((Path(prompt_root) / f"{length}.bin").read_bytes())
                if len(data) != length:
                    raise ValueError("Invalid fixed acceptance prompt")
                ids = list(data)
            prompts.append({"prompt_token_ids": ids})
        results = engine.generate(prompts, sampling, use_tqdm=False)
        for length, result in zip(lengths, results, strict=True):
            print(json.dumps({"native_request": result.request_id, "prompt_length": length,
                              "prompt_ids": result.prompt_token_ids,
                              "tokens": result.outputs[0].token_ids}), flush=True)


if __name__ == "__main__":
    main()
