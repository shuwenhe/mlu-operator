# Phase 12.5.4 Acceptance

## Required gates

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

## Performance (measured)

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

## Outcome

Phase 12.5.4 numerical acceptance is **PASS** on real Vendor vLLM + MLU590-M9 card0 with frozen historical references.
