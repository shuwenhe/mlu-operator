# Phase 9.3 AV hardware acceptance

| Check | Status |
| --- | --- |
| Native BF16 and FP16 compilation for `mtp_592` | PASS |
| Matched-input AV hardware acceptance | PASS |
| Native AV dispatch, request queues and buffer lifetime | PASS |
| Native QK → Softmax → AV versus the original checkpoint | FAIL, two FP16 outputs outside tolerance |
| Strict chained parity with frozen Softmax | BLOCKED by the input differences described below |
| QK, Softmax and runtime R1–R5 regressions | PASS |
| Legacy R6 layer parity | FAIL, pre-existing and byte-for-byte unchanged |
| Device-resident MLA integration and cache acceptance | NOT RUN |

## Scope and numerical contract

AV multiplies the downcast Softmax probabilities `[heads, queries, cache_length]` by V `[heads, cache_capacity, 128]` using FP32 accumulation and one round-to-nearest output cast. Masking is represented by the probabilities. A four-partial-sum FP32 reduction reproduces the pinned checkpoint's no-transpose GEMM order on this matrix. The generic AVX2 dot order investigated during QK is a different reduction path. [PyTorch CPU BLAS implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/BlasKernel.cpp)

The device test captures genuine FP16 and BF16 V, probabilities, and output from the original eager checkpoint, then verifies the reconstituted PyTorch matmul against the saved output. It poisons unused physical cache capacity with NaNs to detect accidental reads. Native compilation and CNRT execution used the Phase 8.2 SDK image and MLU590-M9 card 0 at PCI `0000:4f:00.0`. The standalone harness transfers operands and results for comparison. The runtime dispatch uses owned device buffers, request queues, and asynchronous completion.

The documented synthetic shape matrix yields 86,016 outputs per stage by summing the individual shape rows. The checkpoint matrix uses 16 heads, lengths 1, 7, 32 and 128, two request slots, and one decode step at each corresponding nonzero offset. The threshold remains `1e-5 + 1e-5 * abs(reference)` for FP32 and low-precision results.

## Results

For each dtype and path below, there were 704,512 checkpoint output values per stage. All eight synthetic cases passed both stages. Their low-precision outputs were bitwise exact. No unexpected NaN or Inf was observed.

| Reference path | Dtype | FP32 max abs | FP32 max rel | FP32 max ULP | Low-precision bit differences | FP32 / low-precision tolerance failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Matched checkpoint probabilities | BF16 | 3.5763e-7 | 3.9683e-4 | 5,120 | 0 | 0 / 0 |
| Matched checkpoint probabilities | FP16 | 3.5763e-7 | 6.0837e-4 | 9,216 | 0 | 0 / 0 |
| Matched native Softmax probabilities | BF16 | 3.5763e-7 | 3.9683e-4 | 5,120 | 0 | 0 / 0 |
| Matched native Softmax probabilities | FP16 | 3.5763e-7 | 6.0837e-4 | 9,216 | 0 | 0 / 0 |
| Native chain vs. original checkpoint | BF16 | 3.5763e-7 | 3.9683e-4 | 5,120 | 0 | 0 / 0 |
| Native chain vs. original checkpoint | FP16 | 3.5465e-6 | 6.0837e-4 | 9,216 | 3 | 0 / 2 |

FP32 uses a separately widened PyTorch matmul as its reference. The reference's original FP16/BF16 matmul does not expose its internal accumulator. Native AV is bitwise exact at the converted boundary when PyTorch receives the same probabilities and V bits, including the probabilities produced by the native Softmax.

## Chained FP16 discrepancy

Both tolerance failures were in request 0 with a prefill length of 128. Indices below use zero-based head, query and V channel.

| Mode | Index | Original checkpoint | Native AV | PyTorch with native probabilities | Absolute error |
| --- | --- | ---: | ---: | ---: | ---: |
| Prefill | (15, 68, 53) | 0.0209808349609 | 0.02099609375 | 0.02099609375 | 1.52587890625e-5 |
| Decode | (1, 0, 6) | 0.0483093261719 | 0.0482788085938 | 0.0482788085938 | 3.0517578125e-5 |

The affected rows each contain one of the already accepted one-ULP FP16 Softmax probability differences. The changed keys are 60 in prefill and 90 in decode. PyTorch reproduces both native AV outputs with those input bits. The Phase 9.2 implementation and acceptance evidence remain frozen, so strict chained parity stays open.

The serial candidate had 69 out-of-tolerance values across the synthetic and two matched-input replay paths. Applying QK's 64-lane AVX2 reduction to both dtypes reduced that count to 26. The four-partial-sum order eliminated these AV mismatches. The saved experimental CSVs and sources are under `artifacts/build/phase9_3`.

## Regression results

The frozen QK smoke tests passed and all 32 QK masked score tensors matched their captured bits. Softmax passed its full replay and dispatch tests, and its regenerated accuracy CSV was byte-for-byte identical to the Phase 9.2 artifact. R1–R5 passed. Running the legacy R6 reference-MLA fixture against both the pre-AV and AV-linked binaries returned the same failure status and byte-for-byte identical output. Frozen QK, Softmax and Phase 8.2 source checksums and both 40-entry capture manifests verified.

## Reproduction

Use the existing Phase 8.2 SDK image with card 0 exposed and `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for PyTorch. Run `runtime/native/phase9/av/capture.py` with the checkpoint path, historical `runtime-full-fixtures`, a per-case output directory, dtype and prefill length. The generated AV capture root is `artifacts/build/phase9_3` and includes `CAPTURE_SHA256SUMS`. Build QK, Softmax, and AV with their existing or new build scripts. Then run `runtime/native/phase9/av/replay_av.py QK_FIXTURES AV_FIXTURES QK_BINARY SOFTMAX_BINARY AV_BINARY`. The replay exits nonzero when either of its comparisons fails. The runtime build also emits `av_dispatch_test`, which accepts the PCI address and exercises two independent request slots.

[Execution plan](PHASE9_3_AV_EXECUTION_PLAN.md) · [Accuracy CSV](../artifacts/logs/phase9_3_av_accuracy.csv) · [Hardware execution log](../artifacts/logs/phase9_3_av_hardware.log) · [Source hash verification](../artifacts/logs/phase9_3_av_source_hashes.txt)
