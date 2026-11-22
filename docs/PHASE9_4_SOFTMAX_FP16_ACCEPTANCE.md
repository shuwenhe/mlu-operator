# Phase 9.4 native Softmax FP16 correction and attention acceptance

**PASS on MLU590-M9 card 0.** Both Phase 9.3 FP16 failures reproduced with the archived Softmax kernel. The corrected QK → Softmax → AV chain had zero out-of-tolerance outputs across all 32 checkpoint cases and eight unmasked synthetic chain cases. The chain kept intermediate tensors on the device. The separate AV regression also passed all eight documented synthetic cases.

## Baseline and cause

The baseline ran on PCI `0000:4f:00.0` with CNCC 5.7.6, CNRT 7.7.3, target `mtp_592`, and PyTorch 2.12.1 CPU AVX2 as the reference. The threshold was `1e-5 + 1e-5 * abs(reference)`. Request 0 at prefill length 128 reproduced both failures. QK supplies FP32 logits widened from FP16 scores. The Softmax output shape matches the input, with one row for each head and query.

| Mode | QK logits and Softmax shape | Softmax index, head/query/key | Reference FP16 | Baseline FP16 | AV output index, head/query/channel | AV absolute error |
| --- | --- | --- | ---: | ---: | --- | ---: |
| Prefill | 16 × 128 × 128 | (15, 68, 60) | 0.000429391861 | 0.000429630280 | (15, 68, 53) | 1.52587890625e-5 |
| Decode | 16 × 1 × 129 | (1, 0, 90) | 0.014945983887 | 0.014938354492 | (1, 0, 6) | 3.0517578125e-5 |

Identical QK and causal mask bits excluded input layout, scale, conversion and mask placement. A diagnostic kernel emitted the device exponentials and denominators. The reference shim called the exact AVX2 SLEEF function in the pinned PyTorch wheel. Reconstructing PyTorch's eight-lane denominator and normalization from that function reproduced every FP32 probability in both failing fixture tensors. Reconstructing normalization from the device numerator and denominator also reproduced the device result at the failing positions.

| Boundary | Prefill reference / device FP32 bits | Decode reference / device FP32 bits |
| --- | --- | --- |
| Exponential | `3b426c98` / `3b426c99` | `3f560d80` / `3f560d7f` |
| Denominator | `40dd0700` / `40dd0700` | `425fd5a5` / `425fd5a5` |
| Normalized probability | `39e12fff` / `39e13000` | `3c74d001` / `3c74d000` |

The first difference was the exponential polynomial's final addition. The pinned binary rounds `fma(r*r, coefficient, r)` before adding `1`. The baseline adds `1+r` before the multiply-add. Its FP32 probability lands on an FP16 midpoint in both cases. The device cast then rounds that value correctly, but to the neighboring FP16 probability. The resulting AV discrepancies were also produced by PyTorch when using the native probabilities.

## Correction and numerical verification

The sole production change is in `checkpoint_exp` in `runtime/native/phase9/softmax/softmax.mlu`. It evaluates `1.f + exp_mul_add(r*r, coefficient, r)`. An isolated two-expression candidate also tested exponent splitting. The final build kept the original split because the polynomial-order correction alone passed. It adds no workspace or precision upgrade.

| Dtype | Softmax stage | Baseline bit differences | Baseline worst absolute / relative error | Baseline max ULP | Corrected bit differences |
| --- | --- | ---: | --- | ---: | ---: |
| BF16 | FP32 | 61,655 | 1.1921e-7 / 3.3940e-7 | 4 | 0 |
| BF16 | Converted | 0 | 0 / 0 | 0 | 0 |
| FP16 | FP32 | 58,739 | 1.1921e-7 / 3.2965e-7 | 5 | 0 |
| FP16 | Converted | 3 | 7.6294e-6 / 5.5525e-4 | 1 | 0 |

Each row covers 564,160 checkpoint probabilities. The accuracy CSV records per-case maximum, mean and RMS errors, ULP histograms, row sums, extrema and zero/nonfinite counts. The corrected chain had zero unexpected NaNs or Infs. Existing fully masked unit cases retained PyTorch's expected NaNs. Additional extreme-logit checks matched PyTorch bitwise at FP32 and converted boundaries, including FP16 and BF16 subnormals, negligible probabilities rounding to zero, and extreme finite logits.

For each dtype and length 1, 7, 32 and 128, the chain ran both requests in prefill and decode. It ran each request alone and then both on separate queues. Each input was replayed after a warm-up, with three measured submissions. The final low-precision AV output remained bitwise stable per queue and every trailing buffer guard survived. The baseline reproduced two failing values in each queue configuration. The corrected chain reproduced none.

## Performance

Medians below are for FP16 with 128-token prefill and 129-key incremental decode. Kernel intervals use CNRT notifiers. Throughput uses synchronized host wall time across the active queues and counts query tokens. Buffer bytes include the trailing guards and exclude CNRT runtime overhead. Transfers, allocation and process startup are outside the timing interval.

| Mode | Queues | Softmax before → after | Chain before → after | Query tokens/s before → after |
| --- | ---: | --- | --- | --- |
| Prefill | 1 | 2.421 → 2.412 ms | 27.646 → 27.639 ms | 4,626 → 4,626 |
| Prefill | 2 | 2.438 → 2.434 ms | 38.101 → 38.083 ms | 5,275 → 5,277 |
| Decode | 1 | 60.40 → 60.12 µs | 460.00 → 459.94 µs | 2,075 → 2,069 |
| Decode | 2 | 60.40 → 60.20 µs | 459.92 → 459.04 µs | 3,960 → 3,932 |

Per-slot allocation was unchanged at 7,881,856 bytes for prefill and 1,373,188 bytes for decode. Single-queue corrected QK/AV medians were 21.251/3.979 ms for prefill and 334.64/65.02 µs for decode. The timing samples do not establish a performance improvement. [Per-queue timing samples](../artifacts/logs/phase9_4_performance.csv)

## Regression results and limits

- **PASS** for QK smoke and captured masked-score parity, Softmax checkpoint and mask tests, AV synthetic and checkpoint replay, Softmax and AV runtime dispatch, and R1–R5.
- **FAIL, unchanged** for the historical R6 parity test. It returned its baseline exit status and byte-for-byte output.
- **NOT RUN** for model-level MLA and KV cache acceptance. The test uses captured Q/K/V with complete device-resident attention handoffs.

The Phase 9.3 AV sources, dispatch and runtime digests matched exactly. QK, Phase 8.2 and archived Softmax acceptance evidence also verified. The pre-change Softmax digest is recoverable from the source snapshot in `artifacts/build/phase9_4`. To reproduce, use the Phase 8.2 SDK image and commands in the hardware and regression logs.

[Softmax accuracy](../artifacts/logs/phase9_4_softmax_accuracy.csv) · [Chain accuracy](../artifacts/logs/phase9_4_chain_accuracy.csv) · [Hardware log](../artifacts/logs/phase9_4_softmax_hardware.log) · [Regression log](../artifacts/logs/phase9_4_regression.log) · [Source hashes](../artifacts/logs/phase9_4_source_hashes.txt)
