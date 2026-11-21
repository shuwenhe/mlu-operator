# Standalone BANG C GEMM build and acceptance

## Build result

Both FP16 and BF16 kernels compiled for `mtp_592` with CNCC 5.7.6, MLVM 1.4 and Clang 11.1.0. Both objects contained `.cn_fatbin` and `.cnFatBinSegment`. The host launch symbols `mlu_bang_gemm_fp16` and `mlu_bang_gemm_bf16` were visible, and the reference/benchmark executable linked against CNRT in the compatible image. No CNNL call or CPU fallback is used by the device launchers.

The final compiler commands, including flags, are in [build.sh](../operators/gemm/native/build.sh). An initial build rejected the wrong CNRT invalid-argument enum. The deployed header defines `cnrtErrorArgsInvalid`; the corrected build succeeded. The final transcript is `artifacts/logs/bangc_gemm_build.log`.

| Kernel | Assembly result | Remaining validation |
| --- | --- | --- |
| FP16 | `.arch mtp_592`, `cvt.nram.f32.f16`, `mul.gpr.f32`, `add.gpr.f32` | Actual copies, output and FP32 numerical behavior. |
| BF16 | `.arch mtp_592`, `cvt.nram.f32.bf16`, `mul.gpr.f32`, `add.gpr.f32` | Actual copies, output and BF16 special-value behavior. |

Explicit FP32 multiply/add instructions establish the generated arithmetic path, with contraction disabled. The matrix-unit instructions observed in the earlier probes are out of this implementation until WRAM layout and precision are verified. [API semantics](BANGC_GEMM_API_SEMANTICS.md)

## Reproduction

Use a compatible SDK in an isolated development environment. This build was run with an immutable local image, read-only sources, no network and no MLU nodes. Source files and common FP32 reference are entirely in this workspace. For compile verification only:

```sh
docker run --rm --pull=never --runtime=runc --network=none --read-only \
  --cap-drop=ALL --security-opt=no-new-privileges --user 1001:1001 \
  --tmpfs /tmp:rw,nosuid,nodev,exec,mode=1777 \
  --mount "type=bind,src=$PWD,dst=/project,readonly" --entrypoint /bin/bash \
  sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756 \
  -ec 'test -z "$(find /dev -name "cambricon*")"; cd /tmp; /project/operators/gemm/native/build.sh /tmp/out'
```

## Correctness and hardware acceptance

The harness uses the independent increasing-K FP32 reference and encodes exact FP16/BF16 fixtures. It covers the five required cases and a decode-like 1×129×129 case that crosses both strip boundaries. Each dtype and shape includes repeated launches. The final timed transfer-and-launch path is compared against the reference. The report fields are max absolute, max relative with denominator floor 1e-6, mean absolute, NaN/Inf counts, and mismatches at `1e-5 + 1e-5*abs(reference)`. Add accumulation-sensitive and nonfinite input fixtures after basic hardware acceptance.

**Hardware execution BLOCKED.** No exclusive device or explicit authorization was established. The compile container was confirmed device-free. No output comparison, CNRT queue execution, numerical acceptance or benchmark occurred. In an approved environment, expose only the reserved card, verify driver/SDK compatibility and pass the existing CNRT transfer probe. Build into a writable directory and then run the harness as `gemm_native_test --authorized 0` with the compatible SDK on the library path. The launcher checks argument validity, and callers must check queue synchronization for execution errors. Verify odd-size transfer offsets and lengths on hardware before claiming arbitrary-tail support.

## Performance and CNNL comparison

The harness reports two warm-up launches and twenty timed launches per case. Kernel latency uses a CNRT notifier interval divided by twenty. End-to-end timing includes two host-to-device copies, one synchronized launch and the output copy, excluding allocation. TFLOPS derives from `2*M*N*K/(kernel_us*1e6)`.

No measured figures are available. The CNNL baseline uses heuristic-selected matrix-unit kernels and FP32 compute/output. The standalone path prioritizes an independent layout and arithmetic baseline. It makes no performance claim. Compare on the same reserved card, input residency, dtypes, shapes and timing scope after both paths pass the same oracle. Move to the matrix intrinsic only after the layout and accumulation gates are closed.
