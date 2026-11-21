# DeepSeek-V2-Lite Layer 0 root cause

**Acceptance: FAIL.** The original BF16 failure reproduces on MLU590-M9. Historical comparison against the sequential C++ oracle reported 404 of 24,576 final values outside the existing tolerance over two repetitions. For the same fixed-token fixtures, an original-checkpoint PyTorch comparison reports 4 of 12,288 unique final values outside tolerance, all in decode. These are two different reference paths and sample counts. No numerical kernel fix was applied.

## Where the mismatch starts

For the original BF16 fixture, hidden input, input RMSNorm, Q and KV-A match bit for bit. KV-B is the first tensor with different bits (4 values; max absolute error 7.62939453125e-6). Attention matches bit for bit. The low-precision output projection differs by one bit pattern (absolute error 1.9073486328125e-6). The widened residual sum inherits that difference, which disappears when the sum rounds to BF16. The BF16 residual and post-attention RMSNorm match bit for bit across both slots and both chunks.

The first tolerance failures are the FFN gate and up projections (2 and 4 values respectively), with identical inputs. Down projection has 8 outside tolerance and final output has 4 (max absolute error 0.000244140625). Independent high-precision dot products agree with the MLU outcome for some gate/up elements and with PyTorch for others. For gate request 0 prefill at flat index 16452, PyTorch produces `3b53` and MLU/high-precision produce `3b54`. For gate request 1 decode at index 5194, PyTorch/high-precision produce `3c89` and MLU produces `3c88`. This isolates the BF16 fixture failure to projection numerics with identical inputs. The exact backend accumulation order and intermediate rounding steps remain unverified.

The checkpoint materializes a low-precision output projection and adds it to the low-precision residual. Native already follows that order. The identical-projection replay matches every BF16 residual bit. The MLU microprobe confirms that combining the FP32 projection with the residual before converting can produce a different output, so changing the residual would introduce a semantic regression.

Longer BF16 and FP16 fixtures have additional upstream differences. At BF16 length 128, KV-B already differs in bits, then 13 attention elements fail tolerance before O projection. At FP16 length 1, Q and KV-A already fail tolerance despite matching input norm. The original BF16 fixture diagnosis does not establish the exact cause of every expanded-matrix mismatch.

## Changed files and preserved evidence

- `runtime/native/r6_test.cpp` adds optional trace and regression modes while preserving the original fixed-fixture invocation.
- `runtime/native/p0/` contains the original-source PyTorch trace, precision comparator, dot audit, residual probe and a failing acceptance check for out-of-tolerance values.
- The initial fixture and log backup path is in `artifacts/logs/layer0_p0_backup_path.log`. Accepted GEMM, RMSNorm, RoPE and MLA sentinel hashes in `artifacts/logs/layer0_p0_accepted_sources.sha256` all verified.
- See [precision analysis](LAYER0_RESIDUAL_PRECISION_ANALYSIS.md), [runtime acceptance](MLU_SINGLE_LAYER_RUNTIME_ACCEPTANCE.md), [stage metrics](../artifacts/logs/layer0_residual_accuracy.csv), [dot audit](../artifacts/logs/layer0_dot_audit.csv) and [hardware command log](../artifacts/logs/layer0_residual_hardware.log).
