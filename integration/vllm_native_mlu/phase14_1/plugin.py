"""Per-process entry point discovered by Vendor's general plugin loader."""
import json
import os

ARCH = "NeurXDeepSeekV2ForCausalLMTP2"


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        ARCH, "integration.vllm_native_mlu.phase14_1.model:NeurXDeepSeekV2ForCausalLM"
    )
    print(json.dumps({"phase14_1": "registered", "architecture": ARCH, "pid": os.getpid()}), flush=True)
