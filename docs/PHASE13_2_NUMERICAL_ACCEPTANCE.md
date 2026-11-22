# Phase 13.2 — Numerical and hardware acceptance

The independent TP=1 reference is an unsharded FP32 ATen MLU GEMM with TF32 disabled. Both input precisions use the same deterministic index-based fixture. Reference BF16 operands are converted to FP32 before GEMM. The comparison covers FP32 outputs, including the native BF16 path's FP32 accumulation, FP32 CNCL reduction, and final bias/residual additions.

The comparison predicate uses `atol=2e-5` and `rtol=2e-5` against the FP32 reference. The absolute term handles small values near zero. The relative term allows FP32 rounding differences between native tiles, the unsharded reference and rank-order summation over up to 7,168 products. BF16 narrowing after the sum is outside this phase. The CSV records maximum absolute/relative error, mean absolute error, maximum FP32 ULP distance, bitwise mismatch count, tolerance mismatch count and the first flattened bitwise mismatch.

Each Column run checks the local result against its checkpoint-ordered output slice and checks the explicit gather against the full reference. Each Row run checks the partial against an independent local ATen GEMM, then compares the reduced result with the global reference for both replicated and pre-sharded input. The stability loop checks outputs on every repetition against the first validated results. Dimension rejection checks cover invalid shard sizes, malformed activation widths and malformed residual shape.

[Accuracy CSV](../artifacts/logs/phase13_2_accuracy.csv) · [Hardware log](../artifacts/logs/phase13_2_hardware.log)

## Results on 8 × MLU590-M9

| Gate | Result | Evidence |
|---|---|---|
| Vendor API compatibility | PASS | Frozen Vendor source audit, CNCL device group and MLUCommunicator on every multi-rank run |
| Column Parallel | PASS | Per-rank GEMM and ordered gather, including actual layer-0 gate/up |
| Row Parallel | PASS | Independent partial GEMM, replicated and pre-sharded inputs, post-reduction bias and residual |
| Weight sharding | PASS | 48 actual MLU checkpoint shard hashes and exact reconstruction at TP=1/2/4/8 |
| BF16 numerical parity | PASS | 318 comparisons; maximum absolute error 1.526e-5 |
| FP32 numerical parity | PASS | 300 comparisons; maximum absolute error 1.907e-5; FP32-sensitive probe passed on all eight ranks |
| TP=2 | PASS | Both modes, four shapes, both dtypes, both ranks |
| TP=4 | PASS | Both modes, four shapes, both dtypes, all four ranks |
| TP=8 | PASS | Both modes, four shapes, both dtypes, all eight ranks, CNCL MLU-Link |
| 100-run stability | PASS | 100 repetitions across 64 configurations; 24,000 per-rank timing rows |
| Performance | PASS | Global and local timings, communication, load, throughput, speedup, efficiency and peak tracked allocation |
| Resource cleanup | PASS | Rank cleanup recorded throughout; final CNMON reported no running processes |

All 618 comparisons had zero tolerance mismatches. The three host sharding unit tests passed. The malformed activation, residual and weight shape checks also passed on-device. The TP=8 FP32-sensitive probe distinguished FP32 results from BF16 and TF32 truncation and checked bias ownership by numerical output and per-rank SHA-256.

The sweep launcher stopped once after Column TP=1 emitted its cleanup record when the launcher file was edited concurrently. The 800 complete TP=1 Column samples were retained and the remaining launches resumed separately. The continuation used the same hardware and pinned image. An independent review checked the shard mappings, FP32 precision probe, wide tiling and post-reduction ordering.

[Architecture audit](PHASE13_2_TENSOR_PARALLEL_LINEAR.md) · [Weight sharding](PHASE13_2_WEIGHT_SHARDING.md) · [Performance report](PHASE13_2_PERFORMANCE_REPORT.md)
