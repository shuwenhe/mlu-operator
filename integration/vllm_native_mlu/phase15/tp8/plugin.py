"""Register the TP8 worker architecture in each vendor process."""
ARCH = "NeurXDeepSeekV2ForCausalLMTP8"


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        ARCH, "integration.vllm_native_mlu.phase15.tp8.model:NeurXDeepSeekV2ForCausalLMTP8"
    )
