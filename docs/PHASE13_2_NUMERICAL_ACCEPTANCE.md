# Phase 13.2 — Numerical and hardware acceptance

The independent TP=1 reference is an unsharded FP32 ATen MLU GEMM with TF32 disabled. Both input precisions use the same deterministic index-based fixture. Reference BF16 operands are converted to FP32 before GEMM. The comparison covers FP32 outputs, including the native BF16 path's FP32 accumulation, FP32 CNCL reduction, and final bias/residual additions.

The comparison predicate uses `atol=2e-5` and `rtol=2e-5` against the FP32 reference. The absolute term handles small values near zero. The relative term allows FP32 rounding differences between native tiles, the unsharded reference and rank-order summation over up to 7,168 products. BF16 narrowing after the sum is outside this phase. The CSV records maximum absolute/relative error, mean absolute error, maximum FP32 ULP distance, bitwise mismatch count, tolerance mismatch count and the first flattened bitwise mismatch.

Each Column run checks the local result against its checkpoint-ordered output slice and checks the explicit gather against the full reference. Each Row run checks the partial against an independent local ATen GEMM, then compares the reduced result with the global reference for both replicated and pre-sharded input. The stability loop checks outputs on every repetition against the first validated results. Dimension rejection checks cover invalid shard sizes, malformed activation widths and malformed residual shape.

[Accuracy CSV](../artifacts/logs/phase13_2_accuracy.csv) · [Hardware log](../artifacts/logs/phase13_2_hardware.log)
