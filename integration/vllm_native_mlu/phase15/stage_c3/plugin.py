"""Register the optional Stage C.3 architecture with the vendor workers."""
ARCH = "NeurXDeepSeekV2ForCausalLMStageC3"


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        ARCH, "integration.vllm_native_mlu.phase15.stage_c3.model:StageC3Model"
    )
