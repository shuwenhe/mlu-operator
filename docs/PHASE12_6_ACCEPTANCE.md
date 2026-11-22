# Phase 12.6 — DeepSeek-V2-Lite MLU Production Inference Acceptance

## Environment
- Vendor image: `teleavatar-mlu:v0.1`
- Python: `/torch/venv3/pytorch_infer/bin/python3`
- Device: `MLU590-M9` (card 0)
- Model: `/app/shuwen/model/DeepSeek-V2-Lite`
- Runtime: Vendor vLLM + `NeurXDeepSeekV2ForCausalLM`

## Frozen baseline
- Phase 12.5.4 numerical acceptance remains unchanged.
- This phase uses independent runtime evidence files under `artifacts/logs/`.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| 12.6.1 Concurrent inference | PASS | `artifacts/logs/concurrency_accuracy.csv` |
| 12.6.2 KV cache lifecycle | BLOCKED | `artifacts/logs/kv_lifecycle.csv` |
| 12.6.3 Context scaling | PASS (1/7/17/65), BLOCKED (128) | `artifacts/logs/context_accuracy.csv` |
| 12.6.4 Reliability (>=1h) | PASS | `artifacts/logs/phase12_6_reliability_1h.log`, `artifacts/logs/phase12_6_reliability_1h.csv`, `artifacts/logs/phase12_6_reliability_1h_summary.json` |
| 12.6.5 Performance (TTFT/TPOT/tokens/s) | BLOCKED | `artifacts/logs/performance.csv` |

## Blocking reasons
1. Prompt length 128 is blocked by audited native model constraint (`max_model_len=128`) plus vLLM decode requirement (`max_tokens >= 1`), so `128 + 1` is invalid in current frozen contract.
2. Prefix reuse / request cancellation / explicit eviction are not exposed by current offline LLM API path and are recorded as BLOCKED in KV lifecycle evidence.
3. Context length 128 remains blocked by audited native model constraint (`max_model_len=128`) and vLLM decode minimum-token validation.
4. TTFT/TPOT timestamps are not exposed in collected offline metrics in this run path; throughput is measured, TTFT/TPOT are BLOCKED.

## Raw hardware log
- `artifacts/logs/phase12_6_reliability_1h.log`
- `artifacts/logs/hardware.log` (earlier short validation run)

## Summary snapshot
- `artifacts/logs/phase12_6_summary.json`
- `artifacts/logs/phase12_6_reliability_1h_summary.json`

## Overall phase result
- **BLOCKED** (KV lifecycle and performance gates remain blocked; context length 128 remains blocked by frozen audited model contract).
