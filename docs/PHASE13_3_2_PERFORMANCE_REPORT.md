# Phase 13.3.2 — Projection performance

The measured workload is layer-0 length-17 prefill, repeated 100 times per rank. The chain includes the five projection boundaries and residual, with frozen-replay attention activations supplied to O. The O stage communicates only its FP32 partials. Distributed RoPE, attention and cache are outside the benchmark.

Per-rank synchronized host timings, device stream intervals and native direct-launch device event timings are recorded separately. The direct launch measurement allocates outputs before placing the queue markers and omits the wrapper's padding, allocation and synchronization. CNCL and the composite chain use stage-level device stream intervals. The loader reports checkpoint slicing, H2D, hash verification and native packing together.

Summary p50 and p95 use the slowest rank for each repetition. Speedup divides the TP=1 p50 by the measured group p50. Efficiency divides speedup by TP. Throughput uses 17 logical tokens per chain call. Memory is tracked MLU PyTorch allocator peak plus resident packed buffers calculated from the native 128×128 allocation layout. The memory figures exclude CNCL and driver allocations.

[Rank samples](../artifacts/logs/phase13_3_2/performance.csv) · [Aggregates](../artifacts/logs/phase13_3_2/performance_summary.csv)

## Measured chain latency on MLU590-M9

| TP | p50 ms | p95 ms | tokens/s | Speedup | Efficiency | Aggregate tracked MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.705 | 3.945 | 4589 | 1.00× | 100.0% | 62.50 |
| 2 | 2.545 | 2.835 | 6679 | 1.46× | 72.8% | 65.00 |
| 4 | 2.041 | 2.387 | 8328 | 1.81× | 45.4% | 70.00 |
| 8 | 1.651 | 3.575 | 10298 | 2.24× | 28.1% | 98.00 |

The eight-card median improved by 2.24× in this fixture. TP=4 had a lower p95. These timings were collected before evaluating the final numerical gate, which failed for multi-rank BF16 output.
