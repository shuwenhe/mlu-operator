# Phase 7 FFN precision analysis

The comparison replays Gate and Up on exact native post-attention norm words, then replays SiLU and multiplication on their actual low-precision operands. Down consumes the exact captured native SwiGLU words. Each projection uses the checkpoint weight under the same activation dtype. The independent FP64 comparison uses widened operands and returns a double-precision dot product. The historical comparison uses the preserved serial FP32 oracle and its own upstream input.

| Matched-input stage | BF16 values outside PyTorch tolerance | FP16 values outside PyTorch tolerance |
| --- | ---: | ---: |
| Gate projection | 118 | 557 |
| Up projection | 129 | 637 |
| SiLU | 0 | 0 |
| Elementwise product | 0 | 0 |
| Down projection | 15 | 129 |

Counts aggregate prefill 1/7/32/128 plus a single-token decode for both slots. SiLU and multiplication also had zero bit differences. Every native FP32 FFN projection passed against FP64 at `abs(actual - expected) <= 1e-5 + 1e-5 * abs(expected)`. When both outputs are rounded before comparison, the BF16 native-vs-FP64 gate and up have 79 and 90 tolerance failures, and FP16 has 404 and 442. FP64 therefore gives a mathematical accuracy reference and cannot serve as a replacement PyTorch parity target.

CSV rows identify stage dimensions, input and RHS strides in elements, activation/weight hashes, first threshold mismatch, first differing bit words and low-precision ULP distance. Relative errors use a denominator floor of `1e-6`. On FP32 or FP64 comparisons the ULP columns describe the corresponding low-precision outputs. Full-layer metrics separately compare the original checkpoint's eager Layer 0 with real weights, positions and masks.

**PASS** for the tested FFN pointwise operations on matched inputs. **FAIL** for matched-input rounded GEMM parity. **BLOCKED** for a hardware correction tied to a specific accumulator or tile-order change.

[Projection accuracy CSV](../artifacts/logs/phase7_projection_accuracy.csv)
