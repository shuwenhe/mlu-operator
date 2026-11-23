# Native GEMM kernel design and accuracy

- [Native matrix-instruction GEMM design](#gemm-native-kernel-design)
- [Native GEMM hardware acceptance](#gemm-native-kernel-acceptance)

<a id="gemm-native-kernel-design"></a>

## Native matrix-instruction GEMM design

The independent implementation is under [operators/gemm/native/matrix](../operators/gemm/native/matrix). The scalar implementation, CNNL wrapper and RMSNorm remain unchanged.

- The host benchmark accepts packed row-major A[M,K] and logical row-major B[K,N]. Its packing routine emits zero-padded RHS tiles in N-tile-major and then K-tile order, using the [hardware-verified packing formula](BANGC_BUILD_AND_INSTRUCTIONS.md#gemm-matrix-instruction-analysis). The device launcher accepts already packed B and returns row-major FP32 C. It uses a caller-owned CNRT queue and buffers. It rejects null required arguments and nonpositive dimensions. Callers must provide nonoverlapping correctly sized device storage and synchronize before reading output.
- Each Block task owns one or more 16×128 output tiles in a grid-stride loop with up to 32 tasks. The K slab width is 128. A rows enter NRAM, the prepacked RHS enters WRAM, and the matrix instruction writes a partial FP32 tile. FP32 vector add combines subsequent slabs. One output task owns each output element across the full K reduction.
- The code zero-fills A when M or K has a tail and only copies valid rows and columns back. Packing pads K and N tails. The declared scratch is 20 KiB NRAM (4 KiB A, two 8 KiB FP32 tiles) and 32 KiB WRAM before compiler overhead. No split-K, double buffering, SRAM staging or asynchronous overlap is used.
- Build with [build.sh](../operators/gemm/native/matrix/build.sh) from the existing Ubuntu 24.04 SDK image. It compiles both device objects and device assembly at `--bang-mlu-arch=mtp_592 -O2 -std=c++11 -ffp-contract=off`, then links the benchmark with CNRT and CNNL. It prints the exact commands. CNCC currently warns that `__bang_write_zero` is deprecated on this target. This build used the tested intrinsic for tails and kept the warning in the transcript.

The next performance investigation should quantify strided A staging, WRAM transfer reuse, and host packing cost before selecting another tile or adding buffering. A production path needs a reusable device-resident weight packer or cached weights, profiling across larger M/N/K families, verified memory tooling, and an explicit stride/output contract for inference integration. This implementation is a measured matrix-instruction baseline.

<a id="gemm-native-kernel-acceptance"></a>

## Native GEMM hardware acceptance

**PASS for the 14 native FP16/BF16 correctness cases on MLU 0.** CNNL also passed the same 14 cases using the same inputs. Both paths repeated the five required shapes per dtype in the timed run on MLU 2 (`0000:6f:00.0`). The device builds emit the expected typed matrix instructions and fatbins. This is actual execution evidence.

| Check | Native | CNNL |
| --- | ---: | ---: |
| Correctness cases on MLU 0 | 14 | 14 |
| Maximum absolute, relative and mean absolute error | 0 | 0 |
| Mismatches at existing `1e-5 + 1e-5*abs(reference)` gate | 0 | 0 |
| NaN/Inf count (combined) | 0 | 0 |
| Adjacent output canary errors | 0 | 0 |
| Timed cases on MLU 2 | 10 | 10 |

The benchmark uses deterministic LCG seeds 7 and 43 and dyadic values exact in both input formats. Correctness includes five requested shapes, a 1×1×1 small-magnitude case and a 17×129×257 paired-sign cancellation case using larger magnitude A and small B. Reference accumulation is independent serial FP32. The result applies to those fixtures. Broader dynamic ranges, memory-checker coverage and arbitrary alignment beyond tested packed matrices remain open. No correctness tolerance was loosened.

The one-card test container maps the selected compute/IPCM node and shared control using the already authorized disposable SDK context. The existing eight-card CNRT smoke gate was reused. Neither Team container, GLM53, NeurX, nor the earlier operator implementations were modified. Post-run CNMON reported 0 MiB used on every card.

[MLU 0 accuracy CSV](../artifacts/logs/gemm_native_matrix_accuracy.csv) · [Build and correctness transcript](../artifacts/logs/gemm_matrix_build_and_check.log) · [MLU 2 rerun](../artifacts/logs/gemm_matrix_performance_run.log)
