# Phase 12 production inference acceptance

- [Phase 12.5.1 / 12.5.2 runtime acceptance status](#phase12-5-1-acceptance)
- [Phase 12.5.3 Final Numerical Acceptance (Vendor vLLM + MLU590-M9 card0)](#phase12-5-3-final-acceptance)
- [Phase 12.5.4 Acceptance](#phase12-5-4-acceptance)
- [Phase 12.5.4 Root Cause Report — Native KV Cache Numerical Investigation](#phase12-5-4-root-cause-report)
- [Phase 12.6 — DeepSeek-V2-Lite MLU Production Inference Acceptance](#phase12-6-acceptance)

<a id="phase12-5-1-acceptance"></a>

## Phase 12.5.1 / 12.5.2 runtime acceptance status

| Gate | Result | Evidence |
|---|---|---|
| Vendor Environment | **PASS** | `artifacts/logs/phase12_5_2_environment.log` |
| Native Model Registration | **PASS** | Architecture resolved as `NeurXDeepSeekV2ForCausalLM` in `phase12_5_2_registration_hardware.log` |
| Model Instantiation | **PASS** | Native model constructed and weights loaded in Vendor worker runtime |
| 27-layer Cache Discovery | **PASS** | Static context reached 27 layers (`static_context_layer_count`) |
| Worker Cache Allocation | **PASS** | Worker allocated all native cache groups; per-layer allocation IDs logged |
| Reshape Hook | **PASS** | Reshape hook invoked for all layers (`reshape_hook_invoked=true`) |
| K/V Zero-copy | **PASS** | Zero-copy storage checks logged (`zero_copy_verified=true`) |
| Scheduler Metadata | **PASS** | Real scheduler schedule events and native metadata consumption logged |
| Native Forward | **PASS** | Real prefill + decode requests executed; native request outputs logged |
| Hardware Acceptance | **BLOCKED** | Runtime integration passed, but fixture-based regression controls from prior phases were not rerun in this pass due unavailable mounted fixture set |

### Notes

- No package/environment upgrades were performed.
- Vendor baseline (`DeepseekV2ForCausalLM`) remained runnable and unchanged during this phase.
- Final numerical follow-up is tracked in [PHASE12 5 3 FINAL ACCEPTANCE](#phase12-5-3-final-acceptance).
- Root-cause closure and final numerical pass are recorded in [PHASE12 5 4 ACCEPTANCE](#phase12-5-4-acceptance).
- Detailed runtime evidence is captured in:
  - `artifacts/logs/phase12_5_2_registration_hardware.log`
  - `artifacts/logs/phase12_5_2_worker_cache_hardware.log`
  - `artifacts/logs/phase12_5_2_scheduler_metadata.log`

<a id="phase12-5-3-final-acceptance"></a>

## Phase 12.5.3 Final Numerical Acceptance (Vendor vLLM + MLU590-M9 card0)

### Scope and runtime

- Repository: `/app/shuwen/mlu-operator`
- Model mount: `/app/shuwen/model/DeepSeek-V2-Lite` -> `/checkpoint`
- Vendor image: `teleavatar-mlu:v0.1`
- Python: `/torch/venv3/pytorch_infer/bin/python3`
- Device: `MLU590-M9 card0`

### Gate status (required)

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

### Phase 12.5.3 required checks

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

### Quantitative summary

- Full comparison command result: `reference tolerance failures=65270140`
- Accuracy CSV status counts: `PASS=176`, `FAIL=2024`
- Greedy token-id match vs historical references: `2/20`
- Performance:
  - TTFT: len1 `0.1663s`, len7 `0.7346s`, len17 `1.6993s`, len65 `6.5804s`
  - TPOT: len1 `0.1691s/token`, len7 `0.1686s/token`, len17 `0.1718s/token`, len65 `0.1870s/token`
  - Throughput (batch 1/7/17/65, max_tokens=4): `1.4285 tokens/s`
  - Peak memory: up to `582,897,664 bytes`

### Current blocker

- The final fixture-based numerical gate is not accepted.
- Mismatch is broad (including logits and most per-layer tensors), with `key.bin`/`value.bin` divergence dominating totals in `phase12_5_3_accuracy_breakdown.csv`.
- Until this is resolved, final Phase 12.5.3 hardware acceptance remains **BLOCKED**.

<a id="phase12-5-4-acceptance"></a>

## Phase 12.5.4 Acceptance

### Required gates

| Gate | Result | Evidence |
|---|---|---|
| Historical fixture integrity | **PASS** | `artifacts/logs/phase12_5_4_source_hashes.txt` |
| KV Cache Write | **PASS** | `artifacts/logs/phase12_5_4_kv_write_accuracy.csv` |
| KV Cache Read | **PASS** | `artifacts/logs/phase12_5_4_kv_read_accuracy.csv` |
| Block mapping | **PASS** | `artifacts/logs/phase12_5_4_hardware.log` (`schedule/new/cached/cache_free`) |
| MLA numerical correctness | **PASS** | `artifacts/logs/phase12_5_4_attention_chain_accuracy.csv`, `phase12_5_4_phase12_3_accuracy.csv` |
| 27-layer numerical comparison | **PASS** | `artifacts/logs/phase12_5_4_layer_accuracy.csv` |
| Logits comparison | **PASS** | `artifacts/logs/phase12_5_4_logits_accuracy.csv` |
| Greedy token parity | **PASS (20/20)** | `artifacts/logs/phase12_5_4_token_ids.csv` |
| Incremental Decode | **PASS** | `artifacts/logs/phase12_5_4_layer_accuracy.csv` (`step1..3` all pass) |
| Continuous Batching | **PASS** | `artifacts/logs/phase12_5_4_hardware.log` (multi-request scheduled together) |
| No silent Vendor fallback | **PASS** | `artifacts/logs/phase12_5_4_hardware.log` (`fallback_count=0`) |
| Performance measurements | **PASS (measured)** | `artifacts/logs/phase12_5_4_performance.csv` |

### Performance (measured)

- TTFT:
  - len1: `0.1653 s`
  - len7: `0.7414 s`
  - len17: `1.7108 s`
  - len65: `6.7071 s`
- TPOT:
  - len1: `0.1647 s/token`
  - len7: `0.1862 s/token`
  - len17: `0.1847 s/token`
  - len65: `0.1790 s/token`
- Throughput (batch 1/7/17/65): `1.4070 tokens/s`
- Peak MLU memory (max observed): `587,378,688 bytes`

### Outcome

Phase 12.5.4 numerical acceptance is **PASS** on real Vendor vLLM + MLU590-M9 card0 with frozen historical references.

<a id="phase12-5-4-root-cause-report"></a>

## Phase 12.5.4 Root Cause Report — Native KV Cache Numerical Investigation

### Environment

- Repo: `/app/shuwen/mlu-operator`
- Vendor image: `teleavatar-mlu:v0.1`
- Python: `/torch/venv3/pytorch_infer/bin/python3`
- Hardware: `MLU590-M9 card0`
- Model: `/app/shuwen/model/DeepSeek-V2-Lite`

### P0 preservation

- Pre-run HEAD/status and source+fixture hashes were recorded in:
  - `artifacts/logs/phase12_5_4_source_hashes.txt`
- Historical fixtures (`phase11_5`, `phase12_3`, `phase12_4`, `phase12_5 reference*`) were only read, not modified.

### First divergence (experimental)

Using the frozen references and real vendor run:

1. **Initial baseline (before fix)**: first failing tensor is `request0 / 0-layer0-key.bin`.
2. Mismatch localization showed:
   - key `nope` slice `[0:128]`: no mismatch
   - key `rope` slice `[128:192]`: all mismatch entries
3. `step0` still had `attention/logits PASS` because prefill length was 1 (single-token softmax degenerates to 1.0).
4. At `step1`, divergence propagated to attention/logits.

Conclusion: the first true divergence appears in the **rope segment of K-path**, before KV cache readback becomes a confounder.

### KV write/read investigation conclusion

- Added optional diagnostic dumps for write inputs (`NATIVE_DEEPSEEK_DUMP_WRITE=1`):
  - `layer{i}-key_write.bin`, `layer{i}-value_write.bin`
- For layers `0,1,2,13,26`, write-tail vs read-tail checks are all PASS:
  - `artifacts/logs/phase12_5_4_kv_write_accuracy.csv`
- Readback vs historical reference for the same layers are all PASS:
  - `artifacts/logs/phase12_5_4_kv_read_accuracy.csv`

This rules out KV cache scatter/gather as the first divergence root cause in this incident.

### Root cause

Two fixture-construction defects in runtime setup:

1. **RoPE table generation mismatch** (did not use checkpoint’s YARN rotary implementation).
2. **Attention softmax scale mismatch** (used `1/sqrt(192)` only, missing YARN `mscale_all_dim` factor).

These setup errors caused key rope-segment mismatch first, then attention/logits mismatch at decode.

### Fix (smallest surgical change)

- Added `integration/vllm_native_mlu/phase12_5/prepare_rope.py` to generate:
  - `cos.bin`, `sin.bin` from checkpoint `DeepseekV2YarnRotaryEmbedding`
  - `scale.txt` with YARN-adjusted softmax scale
- Added optional diagnostics in `integration/vllm_native_mlu/phase12_5/model.py`:
  - `NATIVE_DEEPSEEK_DUMP_WRITE=1` gated write-input dumps only.

No kernel source change and no package/environment upgrade.

### Regression after fix

- Full 27-layer run against frozen references:
  - `reference tolerance failures=0`
  - `phase12_5_4_layer_accuracy.csv`: all PASS
  - `phase12_5_4_logits_accuracy.csv`: all PASS
  - `phase12_5_4_token_ids.csv`: `20/20` greedy parity
- Component regressions:
  - router/rmsnorm, attention chain, phase12_3 metadata adapter, phase12_4 cache+projection all PASS.

Raw logs:
- `artifacts/logs/phase12_5_4_hardware.log`

<a id="phase12-6-acceptance"></a>

## Phase 12.6 — DeepSeek-V2-Lite MLU Production Inference Acceptance

### Environment
- Vendor image: `teleavatar-mlu:v0.1`
- Python: `/torch/venv3/pytorch_infer/bin/python3`
- Device: `MLU590-M9` (card 0)
- Model: `/app/shuwen/model/DeepSeek-V2-Lite`
- Runtime: Vendor vLLM + `NeurXDeepSeekV2ForCausalLM`

### Frozen baseline
- Phase 12.5.4 numerical acceptance remains unchanged.
- This phase uses independent runtime evidence files under `artifacts/logs/`.

### Gate results

| Gate | Result | Evidence |
|---|---|---|
| 12.6.1 Concurrent inference | PASS | `artifacts/logs/concurrency_accuracy.csv` |
| 12.6.2 KV cache lifecycle | BLOCKED | `artifacts/logs/kv_lifecycle.csv` |
| 12.6.3 Context scaling | PASS (1/7/17/65), BLOCKED (128) | `artifacts/logs/context_accuracy.csv` |
| 12.6.4 Reliability (>=1h) | PASS | `artifacts/logs/phase12_6_reliability_1h.log`, `artifacts/logs/phase12_6_reliability_1h.csv`, `artifacts/logs/phase12_6_reliability_1h_summary.json` |
| 12.6.5 Performance (TTFT/TPOT/tokens/s) | BLOCKED | `artifacts/logs/performance.csv` |

### Blocking reasons
1. Prompt length 128 is blocked by audited native model constraint (`max_model_len=128`) plus vLLM decode requirement (`max_tokens >= 1`), so `128 + 1` is invalid in current frozen contract.
2. Prefix reuse / request cancellation / explicit eviction are not exposed by current offline LLM API path and are recorded as BLOCKED in KV lifecycle evidence.
3. Context length 128 remains blocked by audited native model constraint (`max_model_len=128`) and vLLM decode minimum-token validation.
4. TTFT/TPOT timestamps are not exposed in collected offline metrics in this run path; throughput is measured, TTFT/TPOT are BLOCKED.

### Raw hardware log
- `artifacts/logs/phase12_6_reliability_1h.log`
- `artifacts/logs/hardware.log` (earlier short validation run)

### Summary snapshot
- `artifacts/logs/phase12_6_summary.json`
- `artifacts/logs/phase12_6_reliability_1h_summary.json`

### Overall phase result
- **BLOCKED** (KV lifecycle and performance gates remain blocked; context length 128 remains blocked by frozen audited model contract).
