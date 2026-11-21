# Phase 8.1 AVX2 reduction integration

The pinned CPU PyTorch wheel uses AVX2 `bf16_dot_with_fp32_arith` and `fp16_dot_with_fp32_arith` on the sampled checkpoint Gate calls. The experimental kernel implements the observed order for checkpoint K divisible by 64. It accumulates 64 FP32 lanes, folds the eight lane groups, and applies the final horizontal reduction. The candidate uses separately rounded multiply and add. The captured products used in the isolated experiment were all exactly representable in FP32, which made that substitution numerically equivalent on those operands. The compiler rejected scalar device FMA lowering on `mtp_592`; the isolated rejection and supported assembly are preserved.

The full-shape candidate reads the existing packed RHS and emits FP32 staging. Its separate executable uses a link wrapper to select the alternative GEMM for every native projection. Queue and registry lifetimes and the explicit Reference MLA path remain in the existing fixture. The wrapper propagates errors. Accepted production GEMM, RMSNorm and RoPE remain untouched and the recorded source hashes verify.

From the SDK container, use `bash runtime/native/phase8/select_backend.sh BACKEND /results CANDIDATE_DIR ARGS...`. The build command is `bash runtime/native/phase8/build_candidate.sh CANDIDATE_DIR /results`.

| Backend | Executable and scope |
| --- | --- |
| `native_default` | Original trace-enabled fixture with accepted GEMM and explicit Reference MLA |
| `native_avx2_compatible` | Separate Phase 8 fixture with all seven GEMMs wrapped and explicit Reference MLA |
| `cnnl_reference` | Adapter projection comparison. The BF16 Q selector test executed CNNL alone. This route takes adapter arguments and has no full-layer handler. |

The default and candidate projection records each compare against their own matched-input PyTorch reference, with activation and weight hashes available to identify identical operands across runs. The widened PyTorch FP32 column recomputes linear with FP32 operands. It neither captures nor substitutes for the low-precision CPU kernel's internal accumulator. FP64 is an independently widened mathematical comparison.

[Projection CSV](../artifacts/logs/phase8_1_projection_accuracy.csv) · [Backend verification](../artifacts/logs/phase8_1_backend_selection.log) · [CPU instruction samples](../artifacts/logs/phase8_pytorch_profile.log) · [Disassembly](../artifacts/logs/phase8_pytorch_disassembly.log)
