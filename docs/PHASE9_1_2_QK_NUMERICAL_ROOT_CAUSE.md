# Phase 9.1.2 native QK numerical root cause

The serial FP32 dot accumulated the 192 products in feature order. The pinned PyTorch low-precision matmul follows an AVX2 reduction order with 64 accumulation lanes and a fixed horizontal fold. The change in FP32 addition order shifts some values across rounding boundaries at the first BF16 or FP16 conversion. The difference then reaches the scaled score. Causal mask placement matched in all device experiments.

The isolated 64-lane experiment grouped products at offsets 0, 64 and 128, folded eight groups of eight with the order documented by Phase 8.1, then applied the eight-lane horizontal reduction. Both casts still use the device's round-to-nearest conversion. The accepted experiment was promoted into the existing QK launcher. The launcher ABI, queue and cache-capacity contract are unchanged.

| Device strategy | BF16 scaled / masked mismatches | FP16 scaled / masked mismatches |
| --- | ---: | ---: |
| Serial FP32 | 33 / 20 | 334 / 169 |
| 16-element serial chunks | 22 / 10 | 248 / 139 |
| 32 interleaved lanes | 42 / 28 | 283 / 152 |
| Binary tree | 25 / 15 | 251 / 140 |
| AVX2-order 64 lanes | 0 / 0 | 0 / 0 |
| Scalar FMA | Assembly rejected | Assembly rejected |

Counts use `1e-5 + 1e-5 * abs(reference)` and include 564,160 values per dtype and score stage. CNAS rejected the scalar FMA instruction `fma.gpr.f32` on `mtp_592`. Phase 8.1's known AVX2 order is a better predictor of checkpoint parity than distance to FP64. For example, the tree had zero raw dot tolerance failures against rounded FP64 in both dtypes but failed the eager checkpoint score. The selected order had 21 BF16 and 1 FP16 raw diagnostic failures against rounded FP64, yet all checkpoint score boundaries matched bit-for-bit. The pinned low-precision PyTorch kernel does not expose its internal raw FP32 accumulator. The CSV includes comparisons against independently widened FP64 and the FP32 PyTorch matmul as raw diagnostics.

The original Phase 9.1 source and smoke harness are archived under `runtime/native/phase9/baseline_9_1` with SHA-256 checksums. The historical logs and captures remain in place. The experiment generator reproduces the source variants under `artifacts/build/phase9_1_2/experiments`.

[Phase 8.1 disassembly findings](PHASE8_1_AVX2_REDUCTION_INTEGRATION.md) · [Per-case stage error and bit patterns](../artifacts/logs/phase9_1_2_qk_accuracy.csv)
