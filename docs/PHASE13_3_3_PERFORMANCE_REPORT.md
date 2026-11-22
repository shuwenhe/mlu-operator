# Phase 13.3.3 — O projection compatibility performance

The benchmark uses the frozen layer-0, length-17 prefill fixture and 100 repetitions per rank. Each sample aggregates the slowest rank for each metric. Host intervals are synchronized. In the ordered strategy the final rank folds inside the local GEMM launch; its measured GEMM/fold interval therefore includes both. Stage percentiles for GEMM/fold, CNCL, BF16 cast, O and the chain are retained in the CSV. Stage intervals can include peer wait and do not sum to the chain percentiles.

The chain adds Q, KV-A, its RMSNorm and KV-B to the replay O call. O takes the saved AV fixture. Speedup is relative to the same strategy at TP=1; the legacy chain is included for historical comparison under the same length and row count. Throughput is 17 logical tokens divided by chain latency, and efficiency is speedup divided by TP.

The ordered row path allocates one 64-lane FP32 state per rank. This adds 8.50 MiB at length 17 (68 MiB across eight ranks), and 32.50 MiB at length 65. At TP=8 and length 17 the two kinds of broadcast total 59.63 MiB of logical collective payloads per call before network amplification. Its observed allocator peak is recorded per timing sample and excludes externally allocated native packed weights.

The output-axis experiment gathers BF16 inputs and FP32 results, totaling 0.199 MiB of logical assembled payload at length 17. Its visible scratch estimate is 226,304 bytes per rank at TP=8. That estimate covers gathered tensors and the local FP32 output, excluding opaque collective temporaries. The experimental harness also keeps the row pack for the legacy diagnostic, so its measurements do not describe a production allocator footprint.

[Ordered per-rank timings](../artifacts/logs/phase13_3_3/performance.csv) · [Ordered slowest-rank summaries](../artifacts/logs/phase13_3_3/performance_summary.csv) · [Output-axis timings](../artifacts/logs/phase13_3_3/output_axis/performance_summary.csv)

| Path | TP | O p50 ms | Chain p50 ms | Chain p95 ms | Legacy chain p50 ms | Tokens/s | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordered row | 1 | 1.003 | 3.585 | 3.604 | 3.705 | 4741 | 1.00× | 100.0% |
| Ordered row | 2 | 1.566 | 3.283 | 3.363 | 2.545 | 5179 | 1.09× | 54.6% |
| Ordered row | 4 | 2.098 | 3.416 | 3.896 | 2.041 | 4976 | 1.05× | 26.2% |
| Ordered row | 8 | 2.655 | 3.728 | 6.933 | 1.651 | 4560 | 0.96× | 12.0% |
| Full-K output axis | 1 | 1.114 | 3.662 | 3.692 | 3.705 | 4642 | 1.00× | 100.0% |
| Full-K output axis | 2 | 1.104 | 2.807 | 3.071 | 2.545 | 6055 | 1.30× | 65.2% |
| Full-K output axis | 4 | 0.945 | 2.283 | 2.615 | 2.041 | 7446 | 1.60× | 40.1% |
| Full-K output axis | 8 | 0.908 | 2.084 | 2.834 | 1.651 | 8157 | 1.76× | 22.0% |

Both compatible paths are slower than the historical TP=8 chain at its measured median. The output-axis experiment reduces the compatibility cost, with different output weight ownership. The ordered row path preserves the audited input-channel shard contract.

The deferred-fold strategy performs one state-sized CNCL FP32 SUM (8.50 MiB logical input at length 17), then launches the frozen fold separately. At TP=2 it measured 2.764 ms chain p50, 2.828 ms p95 and 0.055 ms fold p50. These are timings from a failed numerical candidate. The TP=4/8 deferred variants are NOT RUN. [Deferred-fold timings](../artifacts/logs/phase13_3_3/deferred/performance_summary.csv)
