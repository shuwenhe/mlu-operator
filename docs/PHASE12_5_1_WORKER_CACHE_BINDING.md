# Phase 12.5.1 worker cache binding

## Scope

This phase binds Vendor worker-owned allocations to native materialized K/V views through the backend reshape hook, without replacing scheduler ownership or allocation lifetime.

## Code changes

- `integration/vllm_native_mlu/phase12_5/cache.py`
  - `NativeBackend.reshape_kv_cache` now enforces:
    - BF16 alignment (`raw.data_ptr() % 2 == 0`).
    - Spec page-size compatibility against `FullAttentionSpec`.
    - Shared underlying storage for raw/K/V views (no duplicate allocation).
    - Expected K/V storage offsets for accepted layout.
  - Added runtime instrumentation:
    - `reshape_hook_invoked`
    - `cache_allocation_bytes`
    - `cache_base_address`
    - `k_view_address`
    - `v_view_address`
    - `zero_copy_verified`
    - `cache_spec`
- `integration/vllm_native_mlu/phase12_5/model.py`
  - Emits per-layer runtime binding diagnostics while executing requests:
    - Layer class/index and cache spec.
    - Worker allocation IDs and K/V addresses.
    - Scheduler block table and slot mapping slices.

## Ownership guarantees

- No replacement cache storage allocation in reshape hook.
- No ownership transfer from worker allocation to native model.
- K/V views remain zero-copy aliases over worker-provided allocation.

## Layer coverage

- Native model instantiates 27 `NativeCacheLayer` entries.
- Each layer advertises cache spec:
  - BF16, 16 heads, K width 192, V width 128.

## Validation

- Python syntax validation passed for updated modules.
- Live worker allocation and reshape-hook execution were verified in Phase 12.5.2.
- Evidence includes:
  - 27-layer static-context discovery.
  - Worker-owned allocations reshaped for all layers.
  - Zero-copy verification (`zero_copy_verified=true`) for worker K/V views.
  - `artifacts/logs/phase12_5_2_worker_cache_hardware.log`

