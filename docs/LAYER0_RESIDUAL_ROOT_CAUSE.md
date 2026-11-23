# Layer 0 numerical diagnosis and acceptance

- [DeepSeek-V2-Lite single-layer runtime acceptance](#mlu-single-layer-runtime-acceptance)
- [DeepSeek-V2-Lite Layer 0 root cause](#layer0-residual-root-cause)
- [DeepSeek-V2-Lite Layer 0 precision analysis](#layer0-residual-precision-analysis)

<a id="mlu-single-layer-runtime-acceptance"></a>

## DeepSeek-V2-Lite single-layer runtime acceptance

**Execution PASS, numerical acceptance FAIL.** MLU 0 (`0000:4f:00.0`) completed both full layer-zero BF16 repetitions. The test uses real local checkpoint embedding rows and layer-zero tensors. Each batch-size-one request prefilled two fixed tokens and decoded the third token. Request A uses 100000, 42, 100. Request B uses 777, 512, 16. The two queues submit projection work before either is synchronized. Each request has independent device storage for expanded FP32 pre-rotary KV data and separate host rotary history. This is a bounded three-token materialized test cache with an explicit cursor, not a paged production cache or compressed MLA cache.

The layer executes native input RMSNorm, native Q and connected KV projections, native RoPE, explicitly selected CPU reference MLA, native output projection and post-attention RMSNorm, native gate/up/down projections, and explicit host BF16 SwiGLU and residual operations. Host attention and elementwise code are selected by the test contract. They are not hidden runtime fallback. The reference uses independently accumulated FP32 GEMM with BF16 model boundaries and the same explicit CPU attention formula, rather than a full PyTorch layer differential. Rotary values at positions 0–2 follow checkpoint YaRN parameters. Previously accepted standalone PyTorch RoPE parity and the failed FP16 MLA fixture were preserved.

| Stage | Compared elements across both repetitions | Outside unchanged tolerance | Max absolute error |
| --- | ---: | ---: | ---: |
| Q | 36,864 | 0 | 1.62124634e-5 |
| Projected KV cache | 81,920 | 0 | 7.15255737e-7 |
| CPU reference MLA using native intermediates | 24,576 | 0 | 1.90734863e-6 |
| Output projection FP32 | 24,576 | 0 | 7.15255737e-7 |
| Attention residual BF16 | 24,576 | 4 | 0.000244140625 |
| Post-attention RMSNorm with matched inputs | 24,576 | 0 | 0 |
| Post-attention RMSNorm fed by independent chains | 24,576 | 4 | 0.000244140625 |
| Gate | 131,328 | 1,754 | 2.7670525e-5 |
| Up | 131,328 | 2,412 | 2.77534127e-5 |
| Down | 24,576 | 1,948 | 5.33424318e-5 |
| Final layer output | 24,576 | 404 | 0.00048828125 |

The threshold remains `1e-5 + 1e-5*abs(reference)`. The final failures repeat at the same points across both repetitions. The first failed gate occurs after casting O projection to BF16 and adding the residual. Matched-input RMSNorm agrees exactly, isolating the earlier rounding boundary. Do not ship this path as a numerically accepted full layer. Keep the failing fixture and investigate projection accumulation and BF16 midpoint behavior before enabling production dispatch.

The initial reference-MLA test also exposed a position-buffer sizing error. CNRT rejected the upload into a two-byte-per-element allocation. Fixing u32 to four bytes per element resolved it. The first full-layer run stopped at an intermediate parity failure. The final run collected every stage and returned exit 1 after explicit teardown. It recorded 163,053,568 packed-weight bytes, 98,304 cache bytes, 414,732 peak separately managed bytes, and **zero remaining tracked bytes**. Host CNMON showed zero MiB used on all eight cards after the test.

[Full accuracy CSV](../artifacts/logs/runtime_full_layer_accuracy.csv) · [Final execution log](../artifacts/logs/runtime_final_hardware.log) · [Initial failed parity data](../artifacts/logs/runtime_full_layer_first_failure.csv) · [Original failed MLA fixture](../artifacts/logs/phase5_mla_accuracy.csv) · [Cleanup](../artifacts/logs/runtime_cleanup.log)

### Layer 0 P0 checkpoint rerun

**FAIL.** The original historical oracle still reports 404 final mismatches over two repetitions. An original-source PyTorch trace of the same BF16 fixture finds 4 out-of-tolerance final values in decode and an exact BF16 attention residual. The first original-fixture tolerance failures occur in FFN projections. The expanded matrix also exposes earlier failures for converted FP16 and long BF16 chunks. See [root cause](#layer0-residual-root-cause) and [precision analysis](#layer0-residual-precision-analysis).

| Check | Result |
| --- | --- |
| Trace-enabled R6 compile and link | PASS |
| Native MLU execution of both dtypes and all four lengths | Executed; numerical FAIL |
| Original BF16 residual bit replay and independent MLU microprobe | PASS |
| Two interleaved slots and per-slot KV cache validation | PASS |
| Repeated execution and managed allocation cleanup | PASS |
| R3/R4 injected allocation rollback, queue sync fault and recovery | PASS |
| Original and expanded numerical acceptance | FAIL |
| Post-run physical MLU memory | PASS, all eight cards showed 0 MiB used |

Device traces ran on MLU590-M9 `0000:4f:00.0`; the microprobe ran on `0000:53:00.0`. Target `mtp_592`, driver 6.5.52, Ubuntu 24.04 SDK image `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`, CNCC 5.7.6, CNRT 7.7.3 and CNNL 2.2.18. R6 explicitly selects native projections and reference MLA. There is no implicit backend retry. Every R6 run reported `remaining=0`, R3/R4 exited zero and CNMON verified all eight cards. GLM53 remained exited and both Team containers remained running.

The [hardware log](../artifacts/logs/layer0_residual_hardware.log), [CPU trace log](../artifacts/logs/layer0_pytorch_trace.log) and [microprobe log](../artifacts/logs/layer0_residual_probe.log) contain the expanded compile and execution commands. From the workspace root, the reusable acceptance command is:

```bash
python3 runtime/native/p0/check_acceptance.py artifacts/logs/layer0_residual_accuracy.csv
```

It exited 1 with 284 of 576 original-source stage comparisons failing. Numerical tolerances remain unchanged. The accepted kernels and historical failure artifacts were preserved.

<a id="layer0-residual-root-cause"></a>

## DeepSeek-V2-Lite Layer 0 root cause

**Acceptance: FAIL.** The original BF16 failure reproduces on MLU590-M9. Historical comparison against the sequential C++ oracle reported 404 of 24,576 final values outside the existing tolerance over two repetitions. For the same fixed-token fixtures, an original-checkpoint PyTorch comparison reports 4 of 12,288 unique final values outside tolerance, all in decode. These are two different reference paths and sample counts. No numerical kernel fix was applied.

### Where the mismatch starts

For the original BF16 fixture, hidden input, input RMSNorm, Q and KV-A match bit for bit. KV-B is the first tensor with different bits (4 values; max absolute error 7.62939453125e-6). Attention matches bit for bit. The low-precision output projection differs by one bit pattern (absolute error 1.9073486328125e-6). The widened residual sum inherits that difference, which disappears when the sum rounds to BF16. The BF16 residual and post-attention RMSNorm match bit for bit across both slots and both chunks.

The first tolerance failures are the FFN gate and up projections (2 and 4 values respectively), with identical inputs. Down projection has 8 outside tolerance and final output has 4 (max absolute error 0.000244140625). Independent high-precision dot products agree with the MLU outcome for some gate/up elements and with PyTorch for others. For gate request 0 prefill at flat index 16452, PyTorch produces `3b53` and MLU/high-precision produce `3b54`. For gate request 1 decode at index 5194, PyTorch/high-precision produce `3c89` and MLU produces `3c88`. This isolates the BF16 fixture failure to projection numerics with identical inputs. The exact backend accumulation order and intermediate rounding steps remain unverified.

The checkpoint materializes a low-precision output projection and adds it to the low-precision residual. Native already follows that order. The identical-projection replay matches every BF16 residual bit. The MLU microprobe confirms that combining the FP32 projection with the residual before converting can produce a different output, so changing the residual would introduce a semantic regression.

Longer BF16 and FP16 fixtures have additional upstream differences. At BF16 length 128, KV-B already differs in bits, then 13 attention elements fail tolerance before O projection. At FP16 length 1, Q and KV-A already fail tolerance despite matching input norm. The original BF16 fixture diagnosis does not establish the exact cause of every expanded-matrix mismatch.

### Changed files and preserved evidence

- `runtime/native/r6_test.cpp` adds optional trace and regression modes while preserving the original fixed-fixture invocation.
- `runtime/native/p0/` contains the original-source PyTorch trace, precision comparator, dot audit, residual probe and a failing acceptance check for out-of-tolerance values.
- The initial fixture and log backup path is in `artifacts/logs/layer0_p0_backup_path.log`. Accepted GEMM, RMSNorm, RoPE and MLA sentinel hashes in `artifacts/logs/layer0_p0_accepted_sources.sha256` all verified.
- See [precision analysis](#layer0-residual-precision-analysis), [runtime acceptance](#mlu-single-layer-runtime-acceptance), [stage metrics](../artifacts/logs/layer0_residual_accuracy.csv), [dot audit](../artifacts/logs/layer0_dot_audit.csv) and [hardware command log](../artifacts/logs/layer0_residual_hardware.log).

<a id="layer0-residual-precision-analysis"></a>

## DeepSeek-V2-Lite Layer 0 precision analysis

The reference executes Layer 0 from the local checkpoint's `modeling_deepseek.py` using its eager attention, YaRN, RMSNorm and MLP definitions, the fixed-token checkpoint weight fixtures, and an expanded K/V cache implementing the eager source append contract. BF16 is the checkpoint dtype. FP16 uses explicitly converted inputs and weights on both paths. Each slot runs prefill followed by a one-token decode, with its own K/V cache. The device runtime explicitly uses reference MLA and host elementwise operations alongside native MLU projections, RMSNorm and RoPE.

| Stage | Input → output dtype |
| --- | --- |
| Hidden input, RMSNorm and attention | Activation dtype → activation dtype, with FP32 normalization/reduction or attention arithmetic |
| Q, KV-A, KV-B, gate, up, down | Activation/weight dtype → FP32 accumulation → activation dtype |
| O projection and residual | Activation/weight dtype → FP32 accumulation → activation dtype, then activation-dtype operands → widened add → activation dtype |
| SwiGLU | Activation dtype → rounded SiLU → low-precision product |
| Final output | Activation-dtype operands → widened add → activation dtype |

Both prefill and decode run the same residual statements in the original decoder. Eager execution materializes O projection before the residual addition. A BF16 add is numerically equivalent to widening those already-rounded operands, adding in FP32 and rounding the result once. The PyTorch O projection FP32 trace is a diagnostic recomputation with widened inputs and weights. PyTorch does not expose the original low-precision linear operation's internal accumulators. FP32 residual trace likewise reconstructs the sum from the recorded low-precision operands.

### Residual microprobe

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

### Final output regression

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
