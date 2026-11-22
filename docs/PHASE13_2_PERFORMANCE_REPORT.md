# Phase 13.2 — Tensor Parallel Linear performance

The benchmark runs one Linear across a Vendor TP group. The reported throughput is the logical global `2 × M × K × N` operation count divided by distributed latency.

- **Local GEMM** is a synchronized wall-clock measurement of the per-rank GEMM path, including local padding or chunking when applicable.
- **Communication** independently measures the explicit Column gather or Row AllReduce, with device synchronization. The duration includes launch, synchronization and rank skew. Its ratio to end-to-end latency is a comparison of separate measurements and does not partition the end-to-end time.
- **Weight load** measures transfer of the already sliced CPU shard to its rank's MLU plus local bias transfer and native packing, where applicable. The timer starts after checkpoint slicing.
- **End-to-end Column** reports both rank-local output and the independently measured GEMM + explicit ordered gather. The global metric uses the latter. **End-to-end Row** includes rank-local GEMM, FP32 AllReduce, bias and residual addition.
- **Peak per card** is the PyTorch MLU allocator's peak plus the bytes of the live native BF16 packed buffer. Driver, CNCL and unrelated device allocations are outside this measurement.
- Aggregate times average the slowest rank for each repetition. Speedup divides matching TP=1 latency by measured TP latency. Efficiency divides that speedup by TP. The detailed CSV retains results for each matrix size, dtype, mode and TP setting.

[Per-rank column samples](../artifacts/logs/phase13_2_column_parallel.csv) · [Per-rank row samples](../artifacts/logs/phase13_2_row_parallel.csv) · [Aggregates](../artifacts/logs/phase13_2_performance.csv)

## Measured global latency

All 64 configurations completed 100 repetitions per rank. Latencies are in ms. The TP=8 efficiency and throughput use the same global latency definition.

| Mode | Case | Dtype | TP=1 | TP=2 | TP=4 | TP=8 | TP=8 speedup | Efficiency | TP=8 TFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Column | Small | bfloat16 | 0.477 | 0.814 | 0.928 | 0.996 | 0.48× | 6.0% | 0.034 |
| Column | Small | float32 | 0.178 | 0.632 | 0.719 | 0.666 | 0.27× | 3.3% | 0.050 |
| Column | Medium | bfloat16 | 13.382 | 7.416 | 4.250 | 2.692 | 4.97× | 62.1% | 0.798 |
| Column | Medium | float32 | 0.190 | 0.690 | 0.680 | 0.701 | 0.27× | 3.4% | 3.065 |
| Column | Large | bfloat16 | 401.016 | 199.098 | 101.324 | 51.507 | 7.79× | 97.3% | 1.334 |
| Column | Large | float32 | 1.043 | 1.392 | 0.761 | 0.748 | 1.39× | 17.4% | 91.865 |
| Column | Projection | bfloat16 | 85.491 | 43.153 | 22.122 | 11.725 | 7.29× | 91.1% | 1.282 |
| Column | Projection | float32 | 0.307 | 0.605 | 0.642 | 0.814 | 0.38× | 4.7% | 18.463 |
| Row | Small | bfloat16 | 0.435 | 0.548 | 0.703 | 0.670 | 0.65× | 8.1% | 0.050 |
| Row | Small | float32 | 0.135 | 0.294 | 0.336 | 0.367 | 0.37× | 4.6% | 0.091 |
| Row | Medium | bfloat16 | 13.025 | 7.533 | 4.587 | 3.170 | 4.11× | 51.4% | 0.677 |
| Row | Medium | float32 | 0.145 | 0.421 | 0.484 | 0.421 | 0.34× | 4.3% | 5.100 |
| Row | Large | bfloat16 | 397.630 | 209.065 | 110.759 | 61.985 | 6.41× | 80.2% | 1.109 |
| Row | Large | float32 | 1.042 | 2.412 | 1.665 | 1.945 | 0.54× | 6.7% | 35.336 |
| Row | Projection | bfloat16 | 87.618 | 43.436 | 23.581 | 12.868 | 6.81× | 85.1% | 1.168 |
| Row | Projection | float32 | 0.315 | 0.434 | 0.795 | 0.606 | 0.52× | 6.5% | 24.789 |

BF16 benefited most from sharding the large local GEMM. TP=8 accelerated Large by 7.79× for Column and 6.41× for Row, and Projection by 7.29× and 6.81×. Small incurred more overhead than the shorter local computation saved. The FP32 ATen path already had low single-card latency. Only the Large Column FP32 case improved at TP=8 in this run.
