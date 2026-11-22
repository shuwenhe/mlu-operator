"""Measure actual per-layer MLA cache buffers during a vendor control request."""

import json
import os
from functools import wraps

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import torch_mlu
from vllm import LLM, SamplingParams


def emit(**fields):
    print(json.dumps({"backend": "vendor", **fields}), flush=True)


def describe(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype), "device": str(tensor.device)}
    return [describe(item) for item in tensor]


def main():
    engine = LLM(model="/checkpoint", trust_remote_code=True, dtype="bfloat16",
                 max_model_len=128, max_num_seqs=4, max_num_batched_tokens=128,
                 gpu_memory_utilization=0.45, enforce_eager=True,
                 enable_prefix_caching=False, disable_log_stats=True)
    from vllm_mlu.v1.attention.backends.mla.tmo_flashmla import TMOFlashMLAImpl
    original = TMOFlashMLAImpl.forward
    seen = set()

    @wraps(original)
    def trace(self, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs):
        flags = kwargs.get("kwargs") or (args[2] if len(args) > 2 else None) or {}
        name = layer.layer_name
        mode = "decode" if flags.get("only_decode") else "prefill" if flags.get("only_prefill") else "unknown"
        result = original(self, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)
        if (name, mode) not in seen and attn_metadata is not None:
            seen.add((name, mode))
            selected = getattr(attn_metadata, mode, None)
            blocks = getattr(selected, "block_table", None)
            emit(event="attention", layer=name, mode=mode, compressed=bool(self.use_mla),
                 queue=torch_mlu.mlu.current_stream().mlu_stream,
                 query=describe(query), key=describe(key), value=describe(value),
                 cache=describe(kv_cache), blocks=blocks.cpu().tolist() if blocks is not None else None)
        return result

    TMOFlashMLAImpl.forward = trace
    result = engine.generate(
        ["Explain why the history of mathematics matters. " * 8],
        SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), use_tqdm=False,
    )[0]
    emit(event="tokens", prompt_length=len(result.prompt_token_ids), tokens=result.outputs[0].token_ids)
    if not seen:
        raise RuntimeError("Vendor TMOFlashMLAImpl was not observed")


if __name__ == "__main__":
    main()
