# Phase 13.4 — Standalone performance evidence

The corrected closure logs contain 3,696 layer and 128 full-model per-rank performance observations. The CSV retains latency, timing stages and device memory fields from those records. The exported full-model observations cover one prefill and three decode steps per prompt length. These are sparse replay observations, so they do not characterize production p95 latency or multi-TP speedup. The historical isolated O projection benchmark remains separate.

[Raw performance CSV](../artifacts/logs/phase13_4/performance.csv) · [Slowest-rank model times](../artifacts/logs/phase14/verification.txt) · [Production performance plan](MLU_OPERATOR_PERFORMANCE_ROADMAP.md)
