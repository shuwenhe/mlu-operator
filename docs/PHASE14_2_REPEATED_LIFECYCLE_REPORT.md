# Phase 14.2 — Repeated TP=2 Engine Lifecycle Acceptance

- Run timestamp: `2026-11-22T20:49:46+08:00`
- Evidence root: `artifacts/logs/phase14_2/repeated_lifecycle/run_20261122_204946`
- Model: DeepSeek-V2-Lite
- Hardware: 2 × MLU590-M9
- Runtime: frozen Vendor vLLM image (`sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`)

## Gate Result

Overall repeated lifecycle gate: **PASS**

## Acceptance Summary (10/10)

| Gate | Result |
|---|---|
| Fresh Engine creation | PASS (10/10) |
| Real TP=2 inference (`LLM.generate`) | PASS (10/10) |
| Frozen logits parity | PASS (10/10) |
| Greedy token parity | PASS (10/10) |
| Rank 0 cleanup | PASS (10/10, exit code 0) |
| Rank 1 cleanup | PASS (10/10, exit code 0) |
| EngineCore cleanup | PASS (10/10) |
| Parent cleanup | PASS (10/10) |
| ZeroMQ cleanup | PASS (10/10, socket count 0) |
| FD stability | PASS (no cumulative growth) |
| Thread stability | PASS (no unexplained growth) |
| Process stability | PASS (no orphan worker/engine processes observed at boundaries) |
| MLU resource stability | PASS (`cnmon` boundary checks clean) |

## Resource Trend

| Iteration | FD | Threads | Child Processes | ZMQ Sockets | MLU Memory | Result |
|---|---:|---:|---:|---:|---|---|
| Baseline | 6 | 2 | 0 | 0 | N/A | N/A |
| 1 | 10 | 3 | 1 | 0 | 0 (cnmon boundary clean) | PASS |
| 2 | 10 | 3 | 1 | 0 | 0 (cnmon boundary clean) | PASS |
| 3 | 10 | 3 | 1 | 0 | 0 (cnmon boundary clean) | PASS |
| 5 | 10 | 3 | 1 | 0 | 0 (cnmon boundary clean) | PASS |
| 10 | 10 | 3 | 1 | 0 | 0 (cnmon boundary clean) | PASS |

## FD Stability Conclusion

- In-process baseline: FD=6
- Iteration 1/2/3/5/10 after-cleanup: FD=10 (+4 from baseline)
- No monotonic growth across 10 cycles (stable at 10 after first-cycle warm retention)
- ZeroMQ sockets remained 0 at every cycle boundary
- No unreleased worker process handles (worker exit codes consistently `[0, 0]`)

## Numerical Regression Conclusion

- Frozen logits checks remained within tolerance across all cycles.
- Greedy token IDs matched frozen references in all cycles (`[254, 4334]` for the fixed prompt in this gate).
- No NaN/Inf reported in logged logits parity metrics.
- No CPU fallback evidence was observed in this gate.

## Evidence Files

All required deliverables are under:

`artifacts/logs/phase14_2/repeated_lifecycle/run_20261122_204946/`

- `repeated_lifecycle.csv`
- `repeated_lifecycle_summary.json`
- `numerical_results.csv`
- `fd_inventory.csv`
- `fd_diff.csv`
- `thread_inventory.csv`
- `process_inventory.csv`
- `zmq_cleanup.csv`
- `cnmon_snapshots.log`
- `cleanup.log`
- `resource_trend.md`
- `source_hashes.txt`
- `phase14_2.sha256`
