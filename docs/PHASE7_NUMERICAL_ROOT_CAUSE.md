# Phase 7 numerical root cause

**FAIL: native Layer 0 does not meet original PyTorch parity.** The historical sequential C++ oracle again reports 404 / 24,576 final output failures on the original BF16 fixture. The preserved original PyTorch trace for that fixture reports 4 / 12,288 across the unique chunks. These are separate oracles and sample counts.

The expanded test isolates each GEMM on the actual native activation and the same logical checkpoint weights. The independent FP64 dot products agree with all 224 native FP32 staging comparisons within the existing tolerance. Low-precision projections still differ from matched-input PyTorch. For BF16 the first projection bit difference is KV-B; Q and KV-A match on this matrix. Output projection and the FFN projections have matched-input tolerance failures. FP16 has matched-input Q failures already, including length 1. Comparing original-source layer tensors and matched-input projections separately identifies upstream error propagation.

Small pre-rounding differences can select adjacent representable BF16 or FP16 values. The comparison with rounded FP64 still produces disagreements, so choosing the mathematically more accurate dot product does not establish PyTorch parity. The native SiLU and product results were bit exact on matched inputs across the measured matrix. The original BF16 residual order was already independently verified, and Phase 7 left it intact.

The exact native and PyTorch reduction sequences, tile boundaries and pre-rounding behavior remain unverified. PyTorch low-precision linear returns the rounded output, so its internal accumulator values are unavailable in this capture. The CPU package is PyTorch 2.12.1+cpu with MKL and oneDNN available. Build flags alone do not identify which primitive executed a particular call. The evidence localizes the divergence to projection arithmetic and rounding boundaries. A particular reduction order or tiling defect remains a hypothesis.

**BLOCKED:** experimental correction, generated experimental instruction comparison, and deployment of a corrected Layer 0. No kernel was substituted or acceptance gate relaxed.

[Matched projection evidence](../artifacts/logs/phase7_projection_accuracy.csv) · [Original residual study](LAYER0_RESIDUAL_ROOT_CAUSE.md)
