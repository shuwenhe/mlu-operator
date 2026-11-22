# Phase 9.2 native Softmax hardware acceptance

**PASS at the checkpoint tolerance** `1e-5 + 1e-5 * abs(reference)`. CNCC 5.7.6 built both device variants for `mtp_592`. The test ran the frozen QK launcher and the new Softmax launcher on CNRT queues on MLU590-M9 card 0, PCI `0000:4f:00.0`. For each fixture, the test verified the device QK score bits against the saved checkpoint score bits before comparing Softmax.

## Checkpoint arithmetic

The original eager attention adds the mask in the BF16 or FP16 score dtype, rounds the sum at that dtype, widens the logits for FP32 last-dimension Softmax, and casts the probabilities back before AV. The pinned PyTorch 2.12.1 CPU build dispatches AVX2. It finds the FP32 row maximum, subtracts it, and uses the vector SLEEF FP32 exponential. The denominator accumulates eight lanes over contiguous blocks, includes the partial tail, and folds lanes in 4-, 2-, then 1-lane order. Rows shorter than eight elements use the short sequential reduction. It takes the reciprocal once in FP32 and multiplies each FP32 numerator by it. For finite rows, maximum selection is insensitive to lane order. [PyTorch CPU Softmax implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/SoftMaxKernel.cpp) · [SLEEF exponential implementation](https://github.com/shibatch/sleef/blob/master/src/libm/sleefsimdsp.c)

The device launcher accepts widened QK scores and optionally adds a broadcast low-precision mask before the reduction. It handles causal negative infinity and tail padding as zero probability in rows with finite maxima. Fully masked rows produce NaN probabilities, matching the checkpoint's eager `torch.softmax`. The test counts these separately from unexpected nonfinites and ignores NaN payload differences. The device exponential uses SLEEF-style range reduction and a compensated FP32 polynomial. The fast device intrinsic left two BF16 tolerance failures on the same fixtures. The revised path preserves the low-precision output within tolerance.

## Results

The 32 checkpoint cases cover BF16, independently captured FP16, two request slots, prefill lengths 1, 7, 32, 128, and single-token decode with nonzero KV offsets. Four additional unit cases cover broadcast additive bias, causal masking, a 131-column row with padding, fully masked rows and zero valid columns.

| Dtype | Stage | Checkpoint values | Bit differences | Tolerance failures | Max absolute error | Max relative error | Max ULP | Max row-sum error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | FP32 probabilities | 564,160 | 61,655 | 0 | 1.193e-7 | 3.395e-7 | 4 | 2.385e-7 |
| FP16 | FP32 probabilities | 564,160 | 58,739 | 0 | 1.193e-7 | 3.297e-7 | 5 | 2.385e-7 |
| BF16 | Converted probabilities | 564,160 | 0 | 0 | 0 | 0 | 0 | 0.003418 |
| FP16 | Converted probabilities | 564,160 | 3 | 0 | 7.630e-6 | 5.553e-4 | 1 | 0.0003663 |

No checkpoint probability was NaN or infinite. The unit cases matched the reference's expected NaN and negative-infinity positions. Low-precision row-sum error includes the checkpoint's output cast. The denominator is compared to a separate reconstructed diagnostic because the eager operator returns probabilities alone.

`Dispatch::attention_softmax` validates owned contiguous buffers, selects the dtype-specific launcher, pins the buffers through request completion, and submits on the request's device queue. The dispatch test exercises both dtypes, additive-mask ownership, tail padding and blocked early release. The accepted QK and Phase 8.2 source checksums and the 40-entry fixture checksum manifest verified.

## Reproduction

In the existing SDK container, build Softmax with `bash runtime/native/phase9/softmax/build.sh OUTPUT` and QK with `bash runtime/native/phase9/build_qk.sh OUTPUT`. Run `runtime/native/phase9/softmax/acceptance.py` using the Phase 9.1.2 capture root and the two generated test executables with `TORCH_DEVICE_BACKEND_AUTOLOAD=0`. The normal runtime build also emits `softmax_dispatch_test`, which takes the PCI address.

[Accuracy CSV](../artifacts/logs/phase9_2_softmax_accuracy.csv) · [Hardware execution log](../artifacts/logs/phase9_2_softmax_hardware.log)
