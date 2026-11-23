# BANG C build and matrix instructions

- [MLU590-M9 BANG C compile probe](#bangc-compile-probe)
- [Standalone BANG C GEMM build and acceptance](#bangc-gemm-build-report)
- [BANG C GEMM API semantics](#bangc-gemm-api-semantics)
- [MLU590-M9 matrix instruction analysis](#gemm-matrix-instruction-analysis)

<a id="bangc-compile-probe"></a>

## MLU590-M9 BANG C compile probe

All five probes compiled to relocatable objects and emitted device assembly for `mtp_592`. The FP16 and BF16 matmul overloads lowered to typed device convolution instructions with FP32 destinations. These results establish compiler support for the tested calls and shapes. No kernel was executed.

### Environment and procedure

The compiler banner was `cncc 5.7.6 mlvm 1.4 clang version 11.1.0`, installed at `/usr/local/neuware/bin/cncc`. The banner's `x86_64-pc-linux-gnu` target is the host target. CNCC help identifies `compute_50` with `mtp_592`, and every generated device assembly file begins with `.arch mtp_592`.

The probe ran in a temporary, nonprivileged container from local image ID `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`. The container had no network, no Cambricon nodes, a read-only root and source mount, and temporary compiler output in `/tmp`. It did not run a serving command or interact with production workloads. Object and assembly outputs were inspected before the container exited. The local transcript is `artifacts/logs/bangc_compile_probe.log`.

[Probe source](../probes/bangc_compile.mlu) uses independently selected entry points. From the SDK environment with the probe file visible at `/probe/bangc_compile.mlu` and a writable current directory, the exact compile steps are:

```sh
cncc=/usr/local/neuware/bin/cncc
"$cncc" --version
for p in FP16_MATMUL BF16_MATMUL FP16_CONVERT BF16_CONVERT FP32_REDUCE; do
    "$cncc" --bang-mlu-arch=mtp_592 -O2 -std=c++11 \
        "-DPROBE_$p" -c /probe/bangc_compile.mlu -o "$p.o" || exit
    "$cncc" --bang-mlu-arch=mtp_592 --bang-device-only -O2 -std=c++11 \
        "-DPROBE_$p" -S /probe/bangc_compile.mlu -o "$p.s" || exit
    readelf -SW "$p.o"
done
```

### API results

The signatures below omit the common vendor qualifiers. They were checked in the deployed `/usr/local/neuware/lib/clang/11.1.0/include/bang_device_functions_decls.h`. Both compiler invocations exited 0 for every row, with no warnings or errors observed. Each host ELF relocatable object has `.cn_fatbin` and `.cnFatBinSegment` sections. The generated device assembly separately identifies the target and instruction.

| API | Header declaration | Compiled shape | Target and lowering | Remaining runtime check |
| --- | --- | --- | --- | --- |
| FP16 matmul | `void __bang_matmul(float *, const half *, const half *, uint32_t M, uint32_t K, uint32_t N)` | M=16, K=128, N=128 | `mtp_592`, `conv.nram.f32.f16.f16` | Verify packed RHS order and compare actual FP32 output and cancellation behavior against the CPU oracle. |
| BF16 matmul | `void __bang_matmul(float *, const bfloat16_t *, const bfloat16_t *, uint32_t M, uint32_t K, uint32_t N)` | M=16, K=128, N=128 | `mtp_592`, `conv.nram.f32.bf16.bf16` | Verify packing, FP32 accumulation behavior, range, rounding and subnormals. |
| FP16 to FP32 | `void __bang_half2float(float *, const half *, uint32_t elem_count)` | 128 elements | `mtp_592`, `cvt.nram.f32.f16` | Verify float bits for normal values, signed zero, subnormals and nonfinite inputs. |
| BF16 to FP32 | `void __bang_bfloat162float(float *, const bfloat16_t *, uint32_t elem_count)` | 128 elements | `mtp_592`, `cvt.nram.f32.bf16` | Verify float bits and special-value handling. |
| FP32 reduction | `void __bang_reduce_sum(float *, const float *, uint32_t elem_count)` | 128 elements | `mtp_592`, `reducesum.nram.f32` | Verify where the result is stored, legal lengths and alignment, and reduction order and error. The compile probe stores the first output lane. |

The matmul assembly places A and the output in NRAM, B in WRAM, and uses 64-byte-aligned local objects. The probe assumes a correctly packed RHS and performs no packing. The selected tile and vector lengths compiled, but dynamic shapes, tails, alternate address spaces, and the RMSNorm square/rsqrt pipeline were outside this test. No tested API was rejected. The FP32 destination opcode does not by itself verify the device's internal accumulation or rounding behavior.

### Comparison with CNNL

The workspace CNNL GEMM wrapper requests FP32 compute and output through `cnnlMatMul_v2`, with library-managed algorithm selection and packing requirements. The custom probe confirms direct instruction lowering for both input formats, while leaving packing, legal tiles, tails and numeric behavior to the kernel author. The CNNL RMSNorm wrapper invokes a complete operator. The conversion and reduction probes establish two stages of a future custom RMSNorm implementation.

The existing CNNL wrappers compiled to host objects, and the GEMM CPU reference passed. CNNL runtime acceptance for this workspace remains pending. These BANG C probes crossed the device code generation gate in a compatible userspace, but were not linked into a launcher or tested for runtime correctness or performance. The next hardware gate requires an explicitly authorized exclusive card and a passing CNRT transfer probe, followed by minimal known-output fixtures before full GEMM or RMSNorm work. [CNNL acceptance status](ACCEPTANCE_REPORT.md)

<a id="bangc-gemm-build-report"></a>

## Standalone BANG C GEMM build and acceptance

### Build result

Both FP16 and BF16 kernels compiled for `mtp_592` with CNCC 5.7.6, MLVM 1.4 and Clang 11.1.0. Both objects contained `.cn_fatbin` and `.cnFatBinSegment`. The host launch symbols `mlu_bang_gemm_fp16` and `mlu_bang_gemm_bf16` were visible, and the reference/benchmark executable linked against CNRT in the compatible image. No CNNL call or CPU fallback is used by the device launchers.

The final compiler commands, including flags, are in [build.sh](../operators/gemm/native/build.sh). An initial build rejected the wrong CNRT invalid-argument enum. The deployed header defines `cnrtErrorArgsInvalid`; the corrected build succeeded. The final transcript is `artifacts/logs/bangc_gemm_build.log`.

| Kernel | Assembly result | Remaining validation |
| --- | --- | --- |
| FP16 | `.arch mtp_592`, `cvt.nram.f32.f16`, `mul.gpr.f32`, `add.gpr.f32` | Actual copies, output and FP32 numerical behavior. |
| BF16 | `.arch mtp_592`, `cvt.nram.f32.bf16`, `mul.gpr.f32`, `add.gpr.f32` | Actual copies, output and BF16 special-value behavior. |

Explicit FP32 multiply/add instructions establish the generated arithmetic path, with contraction disabled. The matrix-unit instructions observed in the earlier probes are out of this implementation until WRAM layout and precision are verified. [API semantics](#bangc-gemm-api-semantics)

### Reproduction

Use a compatible SDK in an isolated development environment. This build was run with an immutable local image, read-only sources, no network and no MLU nodes. Source files and common FP32 reference are entirely in this workspace. For compile verification only:

```sh
docker run --rm --pull=never --runtime=runc --network=none --read-only \
  --cap-drop=ALL --security-opt=no-new-privileges --user 1001:1001 \
  --tmpfs /tmp:rw,nosuid,nodev,exec,mode=1777 \
  --mount "type=bind,src=$PWD,dst=/project,readonly" --entrypoint /bin/bash \
  sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756 \
  -ec 'test -z "$(find /dev -name "cambricon*")"; cd /tmp; /project/operators/gemm/native/build.sh /tmp/out'
```

### Correctness and hardware acceptance

The harness uses the independent increasing-K FP32 reference and encodes exact FP16/BF16 fixtures. It covers the five required cases and a decode-like 1×129×129 case that crosses both strip boundaries. Each dtype and shape includes repeated launches. The final timed transfer-and-launch path is compared against the reference. The report fields are max absolute, max relative with denominator floor 1e-6, mean absolute, NaN/Inf counts, and mismatches at `1e-5 + 1e-5*abs(reference)`. Add accumulation-sensitive and nonfinite input fixtures after basic hardware acceptance.

**Hardware execution BLOCKED.** No exclusive device or explicit authorization was established. The compile container was confirmed device-free. No output comparison, CNRT queue execution, numerical acceptance or benchmark occurred. In an approved environment, expose only the reserved card, verify driver/SDK compatibility and pass the existing CNRT transfer probe. Build into a writable directory and then run the harness as `gemm_native_test --authorized 0` with the compatible SDK on the library path. The launcher checks argument validity, and callers must check queue synchronization for execution errors. Verify odd-size transfer offsets and lengths on hardware before claiming arbitrary-tail support.

### Performance and CNNL comparison

The harness reports two warm-up launches and twenty timed launches per case. Kernel latency uses a CNRT notifier interval divided by twenty. End-to-end timing includes two host-to-device copies, one synchronized launch and the output copy, excluding allocation. TFLOPS derives from `2*M*N*K/(kernel_us*1e6)`.

No measured figures are available. The CNNL baseline uses heuristic-selected matrix-unit kernels and FP32 compute/output. The standalone path prioritizes an independent layout and arithmetic baseline. It makes no performance claim. Compare on the same reserved card, input residency, dtypes, shapes and timing scope after both paths pass the same oracle. Move to the matrix intrinsic only after the layout and accumulation gates are closed.

<a id="bangc-gemm-api-semantics"></a>

## BANG C GEMM API semantics

The deployed `bang_device_functions_decls.h` declares `void __bang_matmul(float *, const half *, const half *, uint32_t M, uint32_t K, uint32_t N)` and the same overload with two `const bfloat16_t *` operands. No transpose or leading-dimension arguments appear in these overloads. The existing compile probes lower the 16×128×128 M/K/N calls to `conv.nram.f32.f16.f16` and `conv.nram.f32.bf16.bf16`, with A and output in NRAM and B in WRAM, aligned to 64 bytes. This establishes compilation for this tile and FP32 destination types. It does not establish the logical WRAM layout or floating-point accumulation behavior.

Cambricon's GEMM tutorial stages and rearranges filter data before its convolution-based path. Its packing is evidence that physical operand layout matters. The example is not a verified packing contract for these exact FP16/BF16 `mtp_592` overloads. The deployed installed convolution sample similarly stages NRAM input and WRAM weights. Exact allowable dimensions, alignment minima, transpose mappings, packed strides, and accumulator guarantees remain open. Verify them with the target intrinsic manual and an authorized one-hot/asymmetric device fixture before integrating the matrix unit. [Cambricon GEMM laboratory](https://developer.cambricon.com/index/curriculum/expdetails/id/8/classid/8.html)

### Initial implementation contract

The independent kernel uses packed row-major A[M,K], B[K,N] and C[M,N] with explicit M, N and K. A and B are stored as 16-bit floats. Device conversion widens inputs to FP32, then the kernel uses explicit FP32 multiply and add in increasing K order. The compiled design caps scratch by using 128-element K slabs and 128-column output strips. Each task owns a row strip and writes only valid columns. No WRAM packing or transpose assumptions enter this baseline. It runs entirely on the MLU and has no CPU fallback.

The scratch budget is 2,048 bytes per task before compiler padding and spills. It comprises two 128-element 16-bit input buffers, two 128-element FP32 input buffers, and one FP32 accumulator strip. The tile size makes the conversion length fixed at 128. Input transfers and output stores use the exact logical tails. Check generated declarations in the build report. The total usable NRAM and deployment-specific alignment limits are not established by this allocation. [Cambricon BANG C operator guide](https://github.com/Cambricon/mlu-ops/blob/master/docs/BANG-C-OPS-Develop-Guide.md)

<a id="gemm-matrix-instruction-analysis"></a>

## MLU590-M9 matrix instruction analysis

CNCC 5.7.6 on `mtp_592` declares FP16 and BF16 overloads of `__bang_matmul(float *dst, const Low *src0, const Low *src1, uint32_t M, uint32_t K, uint32_t N)` in `/usr/local/neuware/lib/clang/11.1.0/include/bang_device_functions_decls.h`. The device assembly uses `conv.nram.f32.f16.f16` and `conv.nram.f32.bf16.bf16` for the verified 16×128×128 tile. A and C occupy NRAM, B occupies WRAM. The hardware probe confirms the 16-row A and output order. The overload has no transpose or stride arguments.

The installed SDK convolution example establishes the NRAM/WRAM staging model. The vendor GEMM lab describes rearranging weights for WRAM. Its data layout is not used as evidence for the exact modern overload. [Cambricon GEMM lab](https://developer.cambricon.com/index/curriculum/expdetails/id/8/classid/)

### Verified RHS layout

On host MLU 0 (`0000:4f:00.0`), [layout_probe.mlu](../operators/gemm/native/matrix/layout_probe.mlu) swept all 128 K positions in groups of 16 one-hot A rows. Two passes assigned exact base-128 digits to each physical WRAM word. Decoding the output recovered all 16,384 physical offsets for both dtypes. The maps were identical and every entry matched this word-offset rule:

```text
offset(k,n) = (8*floor((n mod 64)/4) + 4*floor(n/64) + (n mod 4))*128 + k
```

The output is FP32 and the later full kernel accumulates across K slabs in FP32. One-hot probes establish addressing. Cancellation and magnitude tests establish the observed numerical behavior for those fixtures. Neither establishes undocumented inner accumulator behavior for arbitrary exponent patterns. Compiler-emitted local arrays use 64-byte alignment. The verified tile uses 32 KiB WRAM, and exact supported shape/alignment limits beyond this tile were not determined.

The first probe build rejected host `_Float16` under CNCC. Integer bit encoding and explicit `-lstdc++` fixed the compilation and linking issues. Compile and execution commands are in the [layout log](../artifacts/logs/gemm_matrix_layout.log). [FP16 map](../artifacts/logs/matrix-layout-fp16.csv) · [BF16 map](../artifacts/logs/matrix-layout-bf16.csv)
