"""Explicit hybrid control: native Layer 0 O projection within vendor requests."""

import ctypes
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import torch_mlu
from vllm import LLM, SamplingParams

from .projection import PackedProjection


def main(library):
    engine = LLM(model="/checkpoint", trust_remote_code=True, dtype="bfloat16",
                 max_model_len=128, max_num_seqs=4, max_num_batched_tokens=128,
                 gpu_memory_utilization=0.45, enforce_eager=True,
                 enable_prefix_caching=False, disable_log_stats=True)
    prompt = engine.get_tokenizer().encode("Explain why the history of mathematics matters. " * 10)
    prompts = [{"prompt_token_ids": prompt[:length]} for length in (1, 7, 17, 65)]
    sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)

    def generate():
        return [result.outputs[0].token_ids for result in
                engine.generate(prompts, sampling, use_tqdm=False)]

    control = generate()
    installed = []

    def bind(model):
        layer = model.model.layers[0].self_attn.o_proj
        assert layer.bias is None and layer.weight.shape == (2048, 2048)
        original = layer.forward
        stream = torch_mlu.mlu.Stream(device=layer.weight.device)
        with torch_mlu.mlu.stream(stream):
            projection = PackedProjection(library, layer.weight)
        cast = projection.library.phase10_cast_bf16
        cast.restype = ctypes.c_int
        cast.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int]
        calls = []

        def forward(x):
            caller = torch_mlu.mlu.current_stream(x.device)
            stream.wait_stream(caller)
            with torch_mlu.mlu.stream(stream):
                wide = projection(x)
                result = x.new_empty(wide.shape)
                projection._check(cast(stream.mlu_stream, wide.data_ptr(), result.data_ptr(),
                                       x.shape[0]), "BF16 cast")
                stream.synchronize()
            calls.append(x.shape[0])
            print(json.dumps({"backend": "native", "operator": "layer0.o_proj",
                              "rows": x.shape[0], "device": str(x.device),
                              "queue": hex(stream.mlu_stream)}), flush=True)
            return result, None

        layer.forward = forward
        installed.append((layer, original, projection, calls))
        return type(model).__name__

    print("hybrid_control_model", engine.apply_model(bind), flush=True)
    if len(installed) != 1:
        raise RuntimeError("Expected one in-process vendor model")
    layer, original, projection, calls = installed[0]
    try:
        hybrid = generate()
        print(json.dumps({"backend": "hybrid_control", "vendor_tokens": control,
                          "hybrid_tokens": hybrid, "native_rows": calls}), flush=True)
        if not calls:
            raise RuntimeError("Vendor model did not invoke the native O projection")
        if control != hybrid:
            raise RuntimeError("Hybrid token IDs differ from the vendor control")
    finally:
        layer.forward = original
        projection.close()


if __name__ == "__main__":
    main(str(Path(sys.argv[1]).resolve()))
