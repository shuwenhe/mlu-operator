# Phase 12.5.1 native forward contract

## Implemented forward contract

- Native forward receives:
  - input token IDs
  - positions
  - forward-context attention metadata
  - worker-owned K/V cache views
  - active CNRT stream handle
- Native forward explicitly rejects unsupported `intermediate_tensors` and `inputs_embeds`.

## Instrumentation additions

- `integration/vllm_native_mlu/phase12_5/model.py` now emits structured trace events (when `NATIVE_DEEPSEEK_TRACE=1`) including:
  - `architecture_name`
  - `model_instance_id`
  - `layer_id`
  - `layer_class`
  - `cache_spec`
  - `worker_allocation_id`
  - `cache_base_address`
  - `cache_allocation_bytes`
  - `k_view_address`
  - `v_view_address`
  - `reshape_hook_invoked`
  - `zero_copy_verified`
  - `scheduler_metadata_received`
  - `block_table`
  - `slot_mapping`
  - `request_id`
  - `cnrt_queue_id`
  - `native_cache_layer_count`
  - `fallback_count`

## Phase boundary

Phase 12.5.2 executed real Vendor requests through this path and verified:

- Native model instantiation and forward invocation.
- 27-layer cache discovery and reshape-hook usage.
- Scheduler metadata arrival in native forward.
- Incremental decode scheduling after prefill.

Full numerical fixture-regression acceptance remains tracked separately from runtime integration.

