"""Worker registration for Stage C.2 experiments."""
ARCH = "NeurXDeepSeekV2ForCausalLMStageC2"


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        ARCH, "integration.vllm_native_mlu.phase15.stage_c2.model:StageC2Model"
    )
