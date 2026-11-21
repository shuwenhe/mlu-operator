# MLU590-M9 BANG C compile probe

All five probes compiled to relocatable objects and emitted device assembly for `mtp_592`. The FP16 and BF16 matmul overloads lowered to typed device convolution instructions with FP32 destinations. These results establish compiler support for the tested calls and shapes. No kernel was executed.

## Environment and procedure

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

## API results

The signatures below omit the common vendor qualifiers. They were checked in the deployed `/usr/local/neuware/lib/clang/11.1.0/include/bang_device_functions_decls.h`. Both compiler invocations exited 0 for every row, with no warnings or errors observed. Each host ELF relocatable object has `.cn_fatbin` and `.cnFatBinSegment` sections. The generated device assembly separately identifies the target and instruction.

| API | Header declaration | Compiled shape | Target and lowering | Remaining runtime check |
| --- | --- | --- | --- | --- |
| FP16 matmul | `void __bang_matmul(float *, const half *, const half *, uint32_t M, uint32_t K, uint32_t N)` | M=16, K=128, N=128 | `mtp_592`, `conv.nram.f32.f16.f16` | Verify packed RHS order and compare actual FP32 output and cancellation behavior against the CPU oracle. |
| BF16 matmul | `void __bang_matmul(float *, const bfloat16_t *, const bfloat16_t *, uint32_t M, uint32_t K, uint32_t N)` | M=16, K=128, N=128 | `mtp_592`, `conv.nram.f32.bf16.bf16` | Verify packing, FP32 accumulation behavior, range, rounding and subnormals. |
| FP16 to FP32 | `void __bang_half2float(float *, const half *, uint32_t elem_count)` | 128 elements | `mtp_592`, `cvt.nram.f32.f16` | Verify float bits for normal values, signed zero, subnormals and nonfinite inputs. |
| BF16 to FP32 | `void __bang_bfloat162float(float *, const bfloat16_t *, uint32_t elem_count)` | 128 elements | `mtp_592`, `cvt.nram.f32.bf16` | Verify float bits and special-value handling. |
| FP32 reduction | `void __bang_reduce_sum(float *, const float *, uint32_t elem_count)` | 128 elements | `mtp_592`, `reducesum.nram.f32` | Verify where the result is stored, legal lengths and alignment, and reduction order and error. The compile probe stores the first output lane. |

The matmul assembly places A and the output in NRAM, B in WRAM, and uses 64-byte-aligned local objects. The probe assumes a correctly packed RHS and performs no packing. The selected tile and vector lengths compiled, but dynamic shapes, tails, alternate address spaces, and the RMSNorm square/rsqrt pipeline were outside this test. No tested API was rejected. The FP32 destination opcode does not by itself verify the device's internal accumulation or rounding behavior.

## Comparison with CNNL

The workspace CNNL GEMM wrapper requests FP32 compute and output through `cnnlMatMul_v2`, with library-managed algorithm selection and packing requirements. The custom probe confirms direct instruction lowering for both input formats, while leaving packing, legal tiles, tails and numeric behavior to the kernel author. The CNNL RMSNorm wrapper invokes a complete operator. The conversion and reduction probes establish two stages of a future custom RMSNorm implementation.

The existing CNNL wrappers compiled to host objects, and the GEMM CPU reference passed. CNNL runtime acceptance for this workspace remains pending. These BANG C probes crossed the device code generation gate in a compatible userspace, but were not linked into a launcher or tested for runtime correctness or performance. The next hardware gate requires an explicitly authorized exclusive card and a passing CNRT transfer probe, followed by minimal known-output fixtures before full GEMM or RMSNorm work. [CNNL acceptance status](ACCEPTANCE_REPORT.md)
