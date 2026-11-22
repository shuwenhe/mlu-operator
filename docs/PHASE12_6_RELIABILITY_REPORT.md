# Phase 12.6.4 — Reliability Report

## Execution
- Hardware run executed on Vendor vLLM + MLU590-M9.
- Runtime: `teleavatar-mlu:v0.1`, `/torch/venv3/pytorch_infer/bin/python3`
- `PHASE12_6_RELIABILITY_SECONDS=3600`
- Raw log: `artifacts/logs/phase12_6_reliability_1h.log`
- Per-sample CSV: `artifacts/logs/phase12_6_reliability_1h.csv`
- Summary JSON: `artifacts/logs/phase12_6_reliability_1h_summary.json`

## Current status
- Result: **PASS**

## Observed data (1h run)
- Observed reliability window: 3609s
- Samples: 507
- Total requests: 1380
- Successful requests: 1380
- Failed requests: 0
- Generated tokens: 7696
- Token throughput (window average): 2.1324 tokens/s
- Schedule events: 2628
- Cache free events: 1406
- Block reuse observed: yes
- NaN occurrences: 0
- Inf occurrences: 0
- Native fallback count: 0 (no `native fallback` match in raw log)
- Runtime error count: 0
- Host RSS trend: 28,828,716 KB -> 28,831,952 KB (delta +3,236 KB)
- Peak MLU memory: 655,599,616 bytes (stable max)

## Notes
- Reliability gate is closed for the 1-hour criterion.
- Other Phase 12.6 gates (KV lifecycle/performance/context-128 constraints) remain governed by their own evidence and criteria.
