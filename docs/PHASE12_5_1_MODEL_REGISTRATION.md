# Phase 12.5.1 model registration

## Scope

- Registered a distinct architecture name: `NeurXDeepSeekV2ForCausalLM`.
- Kept the original Vendor architecture (`DeepseekV2ForCausalLM`) untouched.
- Bound model registry entry to `integration.vllm_native_mlu.phase12_5.model:NeurXDeepSeekV2ForCausalLM`.

## Code changes

- `integration/vllm_native_mlu/phase12_5/run.py`
  - Updated `ARCHITECTURE` to `NeurXDeepSeekV2ForCausalLM`.
  - Updated `ModelRegistry.register_model` target class path.
- `integration/vllm_native_mlu/phase12_5/model.py`
  - Added canonical class `NeurXDeepSeekV2ForCausalLM`.
  - Kept compatibility alias `NativeDeepseekV2Lite = NeurXDeepSeekV2ForCausalLM`.
  - Added structured runtime instrumentation fields:
    - `architecture_name`
    - `model_instance_id`
    - `static_context_layer_count`
    - `native_cache_layer_count`

## Contract preservation

- Preserves 27-layer layout and Layer 0 dense + Layers 1–26 MoE path.
- Preserves `q_lora_rank=null` expectation and BF16 single-card constraints.
- Reuses accepted native projection C ABI and Phase 11.5 router path.
- Preserves chunked LM head execution.

## Validation

- Python syntax validation passed for updated modules via `python3 -m py_compile`.
- Vendor-container runtime verification completed in Phase 12.5.2 with:
  - Pinned Python `/torch/venv3/pytorch_infer/bin/python3`
  - Architecture resolution `NeurXDeepSeekV2ForCausalLM`
  - Successful generation requests through registered native architecture
  - Evidence: `artifacts/logs/phase12_5_2_environment.log`, `artifacts/logs/phase12_5_2_registration_hardware.log`

