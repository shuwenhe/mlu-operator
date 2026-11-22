# Phase 12.5.1 scheduler metadata contract

## Scope

The native path consumes scheduler-produced metadata from the active forward context and validates it before cache writes.

## Metadata path

- `integration/vllm_native_mlu/phase12_5/model.py`
  - Reads `get_forward_context().attn_metadata`.
  - Uses per-layer metadata entries keyed by registered static layer names.
  - Passes scheduler metadata into request-span validation and cache read/write flow.
- `integration/vllm_native_mlu/phase12_5/cache.py`
  - `NativeMetadataBuilder.build` now logs metadata build events and enforces exclusive-page assumption.
- `integration/vllm_native_mlu/phase12_3/scheduled_attention.py`
  - Existing validation remains the source of truth for:
    - query spans
    - sequence lengths
    - block tables
    - slot mappings
    - block ownership and bounds

## Preserved ownership boundaries

- No scheduler-state reconstruction in native model.
- No synthetic block tables.
- No fixture metadata substitution for production path.
- Request isolation checks remain explicit in span validation.

## Validation

- Static validation path is exercised by syntax check.
- Runtime scheduler metadata dispatch was verified in Phase 12.5.2 on real Vendor requests.
- Evidence includes:
  - Vendor scheduler events with prefill/decode scheduling and finished-request transitions.
  - Native per-layer metadata logs containing block tables and slot mappings.
  - `artifacts/logs/phase12_5_2_scheduler_metadata.log`

