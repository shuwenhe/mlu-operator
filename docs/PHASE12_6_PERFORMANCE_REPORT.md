# Phase 12.6.5 — Performance Report

## Evidence files
- `artifacts/logs/performance.csv`
- `artifacts/logs/concurrency_accuracy.csv`
- `artifacts/logs/hardware.log`

## Current measured values
- Throughput (`tokens/s`) is recorded per scenario/run in `performance.csv`.
- Peak MLU memory is recorded per scenario/run in `performance.csv`.

## Gate status
- Result: **BLOCKED**
- Reason: TTFT/TPOT fields are unavailable in collected offline metrics for this run path, so mandatory TTFT/TPOT acceptance statistics (median and range over repeated runs) cannot be closed yet.

## Notes
- Concurrency output consistency checks passed across repeated runs (token hash stability).
- Performance gate remains blocked until TTFT/TPOT can be captured with real first-token/per-token timestamps under the same hardware environment.
