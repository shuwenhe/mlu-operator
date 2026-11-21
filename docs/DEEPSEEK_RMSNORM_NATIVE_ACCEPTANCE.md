# DeepSeek-V2-Lite native RMSNorm acceptance

## Implementation

[Native BANG C RMSNorm](../operators/rmsnorm/native/rmsnorm.mlu) implements the complete operator for FP16 and BF16. Its standalone interface takes packed device rows, same-dtype gamma, epsilon, and a caller-owned CNRT queue. It returns same-dtype output with optional FP32 normalized values and low-precision pre-gamma values. The existing CNNL wrapper and fixtures remain unchanged.

Each Block task owns rows and processes 128-lane tiles with 2,048 bytes of declared scratch before compiler overhead. The first pass widens inputs, squares in FP32 and sums in FP32 over the logical row width. The second pass scales with FP32 reciprocal square root of `mean + epsilon`, rounds the normalized values to the input dtype, widens the rounded values and same-dtype gamma, multiplies, then rounds the output. Epsilon is input from the fixture or caller, with 1e-6 for this checkpoint. Partial tiles pad scratch and transfer only logical lanes. Widths 512 and 2048 and irregular widths use the same path. Exact odd-size transfers still need device testing.

## Compile-only evidence

Both variants compiled with CNCC 5.7.6 targeting `mtp_592`, `-O2 -std=c++11 -ffp-contract=off`. Both objects contain `.cn_fatbin` and export separate launch symbols. Device assembly shows `.arch mtp_592`, FP32 scalar addition and `recipsqrt.nram.f32`. The FP16 path contains two `cvt.nram.rn.f16.f32` conversions. The BF16 path contains two `cvt.nram.rn.bf16.f32` conversions. The first conversion precedes gamma multiplication and the second produces output. This confirms code generation and explicit rounding instructions.

The [build script](../operators/rmsnorm/native/build.sh) links the native kernels and original CNNL wrapper into the diagnostics executable. The build used the existing Ubuntu 24.04 image `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756` in a disposable device-free container with no network. The executable was invoked only without arguments, which returned usage and exit 2 before any device API call. Transcript is `artifacts/logs/rmsnorm_native_build.log`.

## CPU numerical evidence

The preserved [CPU parity evidence](DEEPSEEK_RMSNORM_PARITY.md) contains 27,912 outputs compared with the exact checkpoint PyTorch CPU class, 13,956 per dtype, with zero absolute and relative error. The new stage comparator self-check recomputed the CPU intermediates and final output on all 10 fixtures and matched the prior CPU reference across all 27,912 final outputs. Fixtures include real gamma for widths 512 and 2048, zero rows, and synthetic widths 1 and 65.

No device metrics are available. The comparator reports maximum absolute error, maximum relative error with a 1e-6 denominator floor, mean absolute error, NaN, Inf and mismatches per dtype and stage. FP32 normalized and CNNL-contract outputs use tolerance `1e-5 + 1e-5*abs(reference)`. Pre-gamma and final low-precision outputs require exact bits, including signed zero. The CNNL output is also compared against the DeepSeek output for context. That cross-contract difference does not fail the CNNL baseline check.

## Authorized device procedure

After an exclusive MLU allocation is explicitly authorized, confirm that exactly one card is visible and pass the existing CNRT transfer probe in the same compatible SDK environment. Then build and run from the project root:

```sh
operators/rmsnorm/native/build.sh artifacts/build/rmsnorm-native
mkdir -p artifacts/build/rmsnorm-actual
export LD_LIBRARY_PATH="/usr/local/neuware/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
artifacts/build/rmsnorm-native/rmsnorm_diagnostics --authorized \
    artifacts/build/rmsnorm-actual artifacts/build/deepseek-fixtures/*norm*.fixture
PYTHONDONTWRITEBYTECODE=1 python3 integration/deepseek/rmsnorm_native_compare.py \
    artifacts/build/deepseek-fixtures --actual-dir artifacts/build/rmsnorm-actual
```

The diagnostic harness captures native FP32 normalized values, native rounded pre-gamma bits, native final bits and the preserved CNNL FP32 output for each fixture. It retains the original FP32 gamma for the CNNL baseline and explicitly converts gamma for the native model path. Treat differences near rounding boundaries as a reason to inspect FP32 reduction and reciprocal-square-root behavior before changing tolerances.

**Hardware acceptance remains BLOCKED.** No exclusive device was authorized for this work. No MLU launches, device numerical comparisons or performance measurements occurred. The device implementation's reduction and reciprocal-square-root fidelity and irregular-tail transfers remain unverified. Production vLLM, NeurX and device mappings were left untouched.
