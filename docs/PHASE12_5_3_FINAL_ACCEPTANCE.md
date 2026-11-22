# Phase 12.5.3 Final Numerical Acceptance (Vendor vLLM + MLU590-M9 card0)

## Scope and runtime

- Repository: `/app/shuwen/mlu-operator`
- Model mount: `/app/shuwen/model/DeepSeek-V2-Lite` -> `/checkpoint`
- Vendor image: `teleavatar-mlu:v0.1`
- Python: `/torch/venv3/pytorch_infer/bin/python3`
- Device: `MLU590-M9 card0`

## Gate status (required)

| Gate | Result | Evidence |
|---|---|---|
| Vendor Environment | **PASS** | `artifacts/logs/phase12_5_3_environment.log` |
| Native Model Registration | **PASS** | `artifacts/logs/phase12_5_3_registration_hardware.log` (`Resolved architecture: NeurXDeepSeekV2ForCausalLM`) |
| Model Instantiation | **PASS** | same log (native model loaded and requests executed) |
| 27-layer Cache Discovery | **PASS** | `artifacts/logs/phase12_5_3_worker_cache_hardware.log` (`static_context_layer_count` reaches 27) |
| Worker Cache Allocation | **PASS** | same log (`worker_allocation_id`, per-layer allocation present) |
| Reshape Hook | **PASS** | same log (`reshape_hook_invoked=true`) |
| K/V Zero-copy | **PASS** | same log (`zero_copy_verified=true`) |
| Scheduler Metadata | **PASS** | `artifacts/logs/phase12_5_3_scheduler_metadata.log` (real `schedule` events and native request metadata path) |
| Native Forward | **PASS** | `artifacts/logs/phase12_5_3_registration_hardware.log` (real prefill + incremental decode) |
| Hardware Acceptance | **BLOCKED** | Historical fixture comparison fails at scale (`reference tolerance failures=65270140`); token-id match `2/20` |

## Phase 12.5.3 required checks

| Check | Result | Evidence |
|---|---|---|
| Preserve 12.5.2 pass path/evidence | **PASS** | 12.5.2 logs unchanged; this phase adds 12.5.3 logs only |
| Restore and use historical fixtures (11.5/12.3/12.4/12.5 refs) | **PASS** | `artifacts/logs/phase12_5_3_source_hashes.txt` (`fixture_hashes_pre_*`) |
| Native Router / Projection / RMSNorm / QK / Softmax / AV regression | **PASS** | `artifacts/logs/phase12_5_3_component_hardware.log` + phase-specific accuracy CSVs |
| Real 27-layer prefill + incremental decode | **PASS** | `artifacts/logs/phase12_5_3_registration_hardware.log` |
| Layer-wise + logits + greedy token comparison vs historical references | **BLOCKED** | `artifacts/logs/phase12_5_3_accuracy.csv`, `artifacts/logs/phase12_5_3_accuracy_breakdown.csv`, `artifacts/logs/phase12_5_3_token_ids.csv` |
| Scheduler allocation / KV write-read / reclaim / block reuse | **PASS** | `artifacts/logs/phase12_5_3_scheduler_metadata.log` (`new`/`cached`/`finished` + `cache_free`; block reuse observed, e.g. blocks 1/2/3 reused) |
| Continuous batching + multi-request isolation | **PASS** | same log (`scheduled_tokens` contains 2 requests in same step; independent block sets) |
| Performance metrics (TTFT/TPOT/throughput/peak memory) | **PASS** | `artifacts/logs/phase12_5_3_performance.csv` |
| Silent fallback in native acceptance | **PASS** | `fallback_count=0` throughout native layer instrumentation in hardware log |

## Quantitative summary

- Full comparison command result: `reference tolerance failures=65270140`
- Accuracy CSV status counts: `PASS=176`, `FAIL=2024`
- Greedy token-id match vs historical references: `2/20`
- Performance:
  - TTFT: len1 `0.1663s`, len7 `0.7346s`, len17 `1.6993s`, len65 `6.5804s`
  - TPOT: len1 `0.1691s/token`, len7 `0.1686s/token`, len17 `0.1718s/token`, len65 `0.1870s/token`
  - Throughput (batch 1/7/17/65, max_tokens=4): `1.4285 tokens/s`
  - Peak memory: up to `582,897,664 bytes`

## Current blocker

- The final fixture-based numerical gate is not accepted.
- Mismatch is broad (including logits and most per-layer tensors), with `key.bin`/`value.bin` divergence dominating totals in `phase12_5_3_accuracy_breakdown.csv`.
- Until this is resolved, final Phase 12.5.3 hardware acceptance remains **BLOCKED**.
