# Phase 13.2 — Tensor Parallel Linear performance

The benchmark runs one Linear across a Vendor TP group. The reported throughput is the logical global `2 × M × K × N` operation count divided by distributed latency.

- **Local GEMM** is a synchronized wall-clock measurement of the per-rank GEMM path, including local padding or chunking when applicable.
- **Communication** independently measures the explicit Column gather or Row AllReduce, with device synchronization. The duration includes launch, synchronization and rank skew. Its ratio to end-to-end latency is a comparison of separate measurements and does not partition the end-to-end time.
- **Weight load** measures transfer of the already sliced CPU shard to its rank's MLU plus local bias transfer and native packing, where applicable. The timer starts after checkpoint slicing.
- **End-to-end Column** reports both rank-local output and the independently measured GEMM + explicit ordered gather. The global metric uses the latter. **End-to-end Row** includes rank-local GEMM, FP32 AllReduce, bias and residual addition.
- **Peak per card** is the PyTorch MLU allocator's peak plus the bytes of the live native BF16 packed buffer. Driver, CNCL and unrelated device allocations are outside this measurement.
- Aggregate times average the slowest rank for each repetition. Speedup divides matching TP=1 latency by measured TP latency. Efficiency divides that speedup by TP. The detailed CSV retains results for each matrix size, dtype, mode and TP setting.

[Per-rank column samples](../artifacts/logs/phase13_2_column_parallel.csv) · [Per-rank row samples](../artifacts/logs/phase13_2_row_parallel.csv) · [Aggregates](../artifacts/logs/phase13_2_performance.csv)
