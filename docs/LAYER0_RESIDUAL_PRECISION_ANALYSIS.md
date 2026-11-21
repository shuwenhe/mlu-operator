# DeepSeek-V2-Lite Layer 0 precision analysis

The reference executes Layer 0 from the local checkpoint's `modeling_deepseek.py` using its eager attention, YaRN, RMSNorm and MLP definitions, the fixed-token checkpoint weight fixtures, and an expanded K/V cache implementing the eager source append contract. BF16 is the checkpoint dtype. FP16 uses explicitly converted inputs and weights on both paths. Each slot runs prefill followed by a one-token decode, with its own K/V cache. The device runtime explicitly uses reference MLA and host elementwise operations alongside native MLU projections, RMSNorm and RoPE.

| Stage | Input → output dtype |
| --- | --- |
| Hidden input, RMSNorm and attention | Activation dtype → activation dtype, with FP32 normalization/reduction or attention arithmetic |
| Q, KV-A, KV-B, gate, up, down | Activation/weight dtype → FP32 accumulation → activation dtype |
| O projection and residual | Activation/weight dtype → FP32 accumulation → activation dtype, then activation-dtype operands → widened add → activation dtype |
| SwiGLU | Activation dtype → rounded SiLU → low-precision product |
| Final output | Activation-dtype operands → widened add → activation dtype |

Both prefill and decode run the same residual statements in the original decoder. Eager execution materializes O projection before the residual addition. A BF16 add is numerically equivalent to widening those already-rounded operands, adding in FP32 and rounding the result once. The PyTorch O projection FP32 trace is a diagnostic recomputation with widened inputs and weights. PyTorch does not expose the original low-precision linear operation's internal accumulators. FP32 residual trace likewise reconstructs the sum from the recorded low-precision operands.

## Residual microprobe

A is FP32 + FP32 followed by BF16 conversion. D widens the already-BF16 residual and adds it to the unrounded projection before BF16 conversion. B adds two BF16 operands. C first rounds the FP32 attention output and then adds. E invokes PyTorch with the checkpoint's materialize-then-add order. The residual fixture uses representable BF16 residual inputs, so A/D and B/C/E form two groups. Six vectors were evaluated. The BANG C probe executed both numeric orders on PCI `0000:53:00.0` and returned the same two sequences as the CPU study.

| Vector | A / D bits | B / C / E bits |
| --- | --- | --- |
| 0 | 3f80 | 3f80 |
| 1 | 3f82 | 3f82 |
| 2 | bf80 | bf80 |
| 3 | 3f20 | 3f20 |
| 4 | bf20 | bf20 |
| 5 | 3f81 | 3f82 |

The [accuracy CSV](../artifacts/logs/layer0_residual_accuracy.csv) contains 592 stage comparisons. Shapes and strides use contiguous row-major logical layout. Trace files transport values in FP32 carriers, and bit columns use the logical dtype. The gate is `abs(actual - expected) <= 1e-5 + 1e-5 * abs(expected)`. Relative reporting uses `max(abs(expected), 1e-6)` as its denominator. Nonfinite comparisons are counted as failures. Flat first-mismatch indices are zero-based. The matched-input BF16 experiments separately replay identical attention inputs and identical projected values.

## Final output regression

All counts are against original checkpoint source. Each prefill count aggregates two request slots, and each decode count covers one token per slot. All listed final outputs were finite.

| Dtype | Prefill length | Prefill failures / elements | Decode failures / elements | Max absolute error |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 2 (original) | 0 / 8192 | 4 / 4096 | 0.000244140625 |
| BF16 | 1 | 0 / 4096 | 0 / 4096 | 0 |
| BF16 | 7 | 9 / 28672 | 2 / 4096 | 0.00048828125 |
| BF16 | 32 | 616 / 131072 | 0 / 4096 | 0.00390625 |
| BF16 | 128 | 3723 / 524288 | 376 / 4096 | 0.00390625 |
| FP16 | 1 | 375 / 4096 | 361 / 4096 | 0.000244140625 |
| FP16 | 7 | 5251 / 28672 | 666 / 4096 | 0.0009765625 |
| FP16 | 32 | 27109 / 131072 | 1209 / 4096 | 0.0009765625 |
| FP16 | 128 | 119360 / 524288 | 783 / 4096 | 0.0009765625 |
