# GEMM weight packing and performance

- [Device-resident packed GEMM weights](#gemm-device-resident-weight-design)
- [Packed-weight memory and queue lifecycle](#gemm-device-memory-lifecycle)
- [Packed-weight hardware acceptance](#gemm-weight-packing-acceptance)
- [Native matrix GEMM performance](#gemm-native-performance-report)
- [Device-resident GEMM weight reuse performance](#gemm-weight-reuse-performance)
- [MLU590-M9 GEMM hardware accuracy and performance](#mlu590-m9-gemm-hardware-accuracy-performance)

<a id="gemm-device-resident-weight-design"></a>

## Device-resident packed GEMM weights

The new [packed-weight interface](../operators/gemm/native/prepacked/packed_weight.h) owns immutable device RHS storage. It reuses the existing matrix-instruction GEMM launchers without altering them.

| API | Contract |
| --- | --- |
| `pack_rhs_host` | Takes packed logical row-major host `B[K,N]`, FP16 or BF16 words. Allocates on the current device, packs on the CPU and completes one synchronous H2D copy before returning. |
| `pack_rhs_device` | Takes device row-major `B[K,N]` and a queue for the current device. Allocates packed storage and enqueues device packing. It optionally brackets packing with caller-provided notifiers. B must stay live until the queue completes. |
| `gemm_prepacked` | Enqueues the existing matrix GEMM with packed RHS and known K/N/dtype. A is device row-major `[M,K]`; output is device FP32 `[M,N]`. |
| `release_packed_weight` | Frees the packed storage on its owning current device and clears the caller's handle on success. The caller must complete all referencing queues first. |

The handle records device ordinal, dimensions, byte count and input type. There is no global cache, automatic device switch or implicit queue synchronization in the asynchronous path. The ABI checks input pointer alignment to scalar type alignment, positive shapes, valid dtype, queue presence and current-device identity. Callers provide valid host/device ranges, nonoverlapping buffers and queues belonging to that device. The runtime handles launch/queue errors. Device pointer provenance and queue ownership are not inferred from raw addresses.

Tiles are 128×128, ordered by N tile then K tile. The device packer bulk-loads valid rows, zero-fills K/N tails, transposes in NRAM and copies each column into the hardware-verified WRAM slot. It uses two 32 KiB NRAM tiles per task. Its logical mapping agrees with the prior one-hot hardware map:

```text
offset(k,n) = (8*floor((n mod 64)/4) + 4*floor(n/64) + n mod 4)*128 + k
```

Persistent storage is `32768 * ceil(K/128) * ceil(N/128)` bytes. The host route uses one temporary host buffer of that size. The device route needs the original `2*K*N` device bytes until packing completes, in addition to the persistent allocation. Existing model-resident row-major weights can be passed directly. The benchmark uploads them once to separate preparation and invocation costs.

Build from the existing authorized Ubuntu 24.04 SDK context with [build.sh](../operators/gemm/native/prepacked/build.sh). It records matrix and packer compilation at `--bang-mlu-arch=mtp_592 -O2 -std=c++11` and the host/link flags. No SDK installation, NeurX change, or production integration is required.

<a id="gemm-device-memory-lifecycle"></a>

## Packed-weight memory and queue lifecycle

Create with a null handle and retain sole release ownership. On success the handle identifies the selected CNRT device and owns one allocation. Select that device before invoking or releasing. Host packing returns ready for use. Device packing returns after submission. Use the same queue for dependent work or wait for the packing queue before publishing to another queue. Multiple queues may read a fully prepared weight while the caller keeps it live. Concurrent release, writes to the underlying storage and sharing a handle across devices are outside the contract.

Synchronize all referencing queues successfully before calling release. A failed synchronization requires explicit queue recovery before freeing memory. On device preparation errors after allocation, the caller still owns any populated handle and must resolve outstanding queue work. On host preparation failure the wrapper attempts immediate cleanup, and a non-null output signals a failed free requiring caller handling. Release on the wrong current device returns `cnrtErrorDeviceInvalid` and preserves ownership. Destroy queues only after pending operations are resolved. No implicit device synchronization is introduced by GEMM or release. The host preparation path uses a synchronous transfer.

Hardware lifecycle tests exercised both preparation routes, multiple live handles, synchronized read sharing on two queues, 32 additional one-word allocation/readback/release cycles, null release, invalid shape/dtype, null queue, misaligned activation pointer and a nonempty output handle. Wrong-device release returned 100005 and preserved the pointer. A linker-wrapped allocation failure returned 100100 and left the handle null in all four cases. This is injected error-path evidence. Physical exhaustion and recovery from a device fault were not attempted.

For the 3×193×65 fixture, logical RHS is 25,090 bytes and packed RHS is 65,536 bytes. Device preparation temporarily retains both. For the Q projection, each logical or packed RHS uses 12,582,912 bytes. Host packing additionally uses that amount of temporary host storage. Actual driver pool peaks were not instrumented. Post-test CNMON is recorded separately to check for residual allocated memory at its reporting granularity.

[Lifecycle and build log](../artifacts/logs/gemm_prepacked_build.log) · [Cleanup snapshot](../artifacts/logs/gemm_prepacked_cleanup.log)

<a id="gemm-weight-packing-acceptance"></a>

## Packed-weight hardware acceptance

**PASS for tested FP16/BF16 fixtures.** The final build used CNCC 5.7.6, CNRT 7.7.3, CNNL 2.2.18 and host driver 6.5.52. It ran on MLU 0 (PCI `0000:4f:00.0`), with MLU 1 (`0000:53:00.0`) visible for wrong-device rejection. Only MLU 0 executed GEMM. The device timing run used MLU 2 (`0000:6f:00.0`).

For both dtypes, the harness compared every packed word, including all padding, in both preparation paths against the existing host layout reference. That is 32768 words per path for 3×193×65 and 98304 for 1×129×257. The prior hardware one-hot maps independently establish the fixed tile mapping. The check covered 524,288 word comparisons with zero differences.

The same handles served 1-, 10-, and 100-call groups with distinct deterministic A matrices (LCG seeds 7+i, RHS seed 43). The test used two live weights and two independent queues, including both queues reading the same immutable weight. Each result was compared against the independent serial FP32 oracle, with the unchanged tolerance `1e-5 + 1e-5*abs(reference)`. Both dtypes had zero maximum absolute, maximum relative and mean absolute error, mismatches, nonfinite outputs and output canary errors. Canaries detect adjacent writes for these fixtures and do not establish complete memory safety.

The benchmark separately verified its final result against the same oracle in each timed trial, including the DeepSeek-shaped projection. No end-to-end DeepSeek inference was run.

[Accuracy CSV](../artifacts/logs/gemm_prepacked_accuracy.csv) · [Build and correctness transcript](../artifacts/logs/gemm_prepacked_build.log) · [Original tile hardware analysis](BANGC_BUILD_AND_INSTRUCTIONS.md#gemm-matrix-instruction-analysis)

<a id="gemm-native-performance-report"></a>

## Native matrix GEMM performance

Measured on MLU 2, `mtp_592`, driver 6.5.52, CNCC 5.7.6, CNRT 7.7.3 and CNNL 2.2.18. Each backend uses identical logical inputs, shape, dtype, queue and timing protocol. Five warmups precede 30 individual device-event samples. P95 is nearest rank. Transfer-inclusive wall time measures H2D for A and each backend's B representation, one synchronized launch and D2H for C. Allocations, algorithm setup and the one-time host packing are outside both timing intervals. The CSV reports host packing separately and includes both latency distributions and effective TFLOPS, calculated as 2MNK / kernel seconds / 1e12.

| M×N×K | FP16 kernel p50, BANG / CNNL (µs) | BF16 kernel p50, BANG / CNNL (µs) |
| --- | ---: | ---: |
| 127×193×65 | 3.60 / 2.76 | 3.56 / 2.76 |
| 32×64×128 | 2.86 / 2.32 | 2.92 / 2.32 |
| 128×256×512 | 7.16 / 5.24 | 7.16 / 5.24 |
| 256×128×512 | 7.28 / 4.02 | 7.28 / 4.04 |
| 1024×1024×1024 | 168.60 / 14.74 | 168.42 / 15.22 |

At 1024³, native reached 12.74–12.75 TFLOPS and CNNL reached 141.10–145.69 TFLOPS. The irregular case reached 0.885–0.895 native TFLOPS versus 1.155 for CNNL. The old scalar BANG kernel took about 1.99 ms at the irregular shape in a separate run with different fixtures and timing aggregation. Its approximate improvement is 550×. The new same-input comparison is the table above. Host RHS packing ranged about 39–7,081 µs across the native cases, which favors reusing packed weights. The benchmark records warm-up counts. Individual warm-up latencies and hardware utilization counters were not collected. The remaining large-shape gap merits profiling A row copies, repeated WRAM loads and tile occupancy. No equivalent end-to-end win is claimed.

### Independent eight-card scaling

BF16 resident 128×256×512, five warmups and 100 timed launches per card. One host thread and independent queue per card. The common wall interval ends with the last synchronized queue. Initial copies, readback and cleanup are outside the timed interval. Each card uses a distinct constant RHS and validates its own output. Mean per-card throughput is aggregate divided by card count. Individual card timing and inter-card communication were not measured.

| Cards | Aggregate TFLOPS | Mean per card | Speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.602 | 4.602 | 1.000× | 100.0% |
| 2 | 6.362 | 3.181 | 1.383× | 69.1% |
| 4 | 12.563 | 3.141 | 2.730× | 68.3% |
| 8 | 24.333 | 3.042 | 5.288× | 66.1% |

This is independent job throughput. No distributed GEMM or CNCL was introduced.

[Matched timing CSV](../artifacts/logs/gemm_native_matrix_performance.csv) · [Scaling CSV](../artifacts/logs/gemm_native_matrix_scaling.csv) · [Scaling command log](../artifacts/logs/gemm_matrix_scaling.log)

<a id="gemm-weight-reuse-performance"></a>

## Device-resident GEMM weight reuse performance

Measured on MLU 2 (`0000:6f:00.0`), driver 6.5.52, CNRT 7.7.3 and CNNL 2.2.18. The local checkpoint records `model.layers.0.self_attn.q_proj.weight` as BF16 `[3072,2048]`. Logical GEMM therefore uses N=3072 and K=2048. M=1 and M=128 are decode-like and prefill-like test batch sizes, respectively. The benchmark uses matched synthetic values at those dimensions for FP16 and BF16, plus the irregular 3×193×65 case. The same logical input data feeds all four paths. No full model inference was run.

### BF16 Q projection

Median amortized microseconds per GEMM. Each trial includes all RHS preparation for the reuse count, transfers, synchronized invocation and release, divided by the number of calls. Repacking also allocates and releases the packed RHS each call. Other paths retain it through the trial.

| M | Path | 1 call | 10 calls | 100 calls | Kernel p50 / p95 (µs) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Host pack every call | 19528.68 | 19496.10 | 19503.74 | 24.28 / 25.20 |
| 1 | Host pack once | 19451.77 | 1997.68 | 248.93 | 24.30 / 24.88 |
| 1 | Device pack once | 1645.28 | 217.61 | 74.91 | 24.12 / 24.68 |
| 1 | CNNL resident | 1237.49 | 160.79 | 52.95 | 8.78 / 9.40 |
| 128 | Host pack every call | 20093.54 | 20116.80 | 20307.43 | 161.54 / 162.32 |
| 128 | Host pack once | 20345.17 | 2749.93 | 1011.44 | 161.32 / 162.60 |
| 128 | Device pack once | 2406.40 | 960.15 | 822.99 | 160.86 / 162.52 |
| 128 | CNNL resident | 1881.48 | 798.10 | 685.72 | 14.38 / 15.04 |

For M=1, BF16 Q projection preparation took 19.32 ms on the host-resident path and 1.58 ms on the device-resident path. Device packing itself took about 349 µs, with about 1.20 ms for the initial logical RHS upload. The same-shape initial scalar device pack took about 38 ms. Bulk row transfers and NRAM transpose reduced its measured pack time by roughly 109×. On the small irregular case, device packing took about 87 µs and remained slower than the roughly 27 µs host pack. Packed RHS reuse materially reduces repeated preparation cost. CNNL remained faster on the measured Q projection, with approximately 112.0 TFLOPS against 10.0 native TFLOPS for BF16 M=128.

### Timing scope and evidence

- Device kernel p50 and nearest-rank p95 use 30 individual CNRT notifier samples after five resident warmups. They exclude transfers and preparation. Effective TFLOPS uses 2MNK divided by kernel p50 seconds and 10^12.
- Transfer-inclusive invocation measures A H2D, one synchronized GEMM and C D2H. RHS is already prepared in this interval. Weight packing, initial RHS H2D and total preparation are separate columns. Amortized p50/p95 use nine independently timed reuse trials. The p95 of nine trials is the maximum. CNNL descriptor creation and ordinary reusable A/C/workspace allocations precede the timing scopes.
- The 72-row final CSV combines host and CNNL measurements from the first run with the optimized device-pack measurements from the second run. The same card, fixtures, matrix kernels and protocol were used for both. The scalar-packer experiment is preserved separately. These are point measurements, with no inference-level throughput claim.
- After hardware tests, all eight cards reported 0 MiB allocated at CNMON granularity, GLM53 remained exited, and both Team containers remained running.
- Outstanding work includes testing exhaustion under an isolated memory budget, running a memory checker, and integrating the explicit queue and handle lifetime contract into a serving adapter. No production source was changed.

[Performance CSV](../artifacts/logs/gemm_prepacked_performance.csv) · [Original scalar pack experiment](../artifacts/logs/gemm_prepacked_scalar_device_experiment.csv) · [Host/CNNL run log](../artifacts/logs/gemm_prepacked_performance_run.log) · [Optimized device run log](../artifacts/logs/gemm_prepacked_device_final_run.log) · [Checkpoint tensor metadata](../artifacts/logs/gemm_prepacked_checkpoint.log) · [Build transcript](../artifacts/logs/gemm_prepacked_build.log)

<a id="mlu590-m9-gemm-hardware-accuracy-performance"></a>

## MLU590-M9 GEMM hardware accuracy and performance

**PASS for the tested packed row-major FP16/BF16 fixtures on host MLU 2.** The native BANG C harness passed six shapes per dtype, including 32×64×128, 128×256×512, 256×128×512 and 127×193×65. The CNNL harness passed seven shapes per dtype. Both reported zero maximum absolute, maximum relative and mean absolute error and zero mismatches on their respective CPU FP32 oracles. Compiler output for BANG C shows separate FP32 multiply and add after input conversion. CNNL requests FP32 compute and output.

The shared shape below uses the same card, dtype, packed layout, M/N/K and effective FLOP count. Each harness uses its own deterministic fixture and timing scope. This is a same-shape observation, with no claim of a matched-input end-to-end benchmark.

| 127×193×65 | BANG C kernel µs | CNNL kernel µs | BANG C TFLOPS | CNNL TFLOPS |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 1985.71 | 2.754 | 0.00160468 | 1.15702 |
| BF16 | 1986.41 | 2.680 | 0.00160412 | 1.18897 |

BANG C used two warmups and 20 timed launches. CNNL used one warmup and 20 timed launches. TFLOPS uses 2MNK divided by kernel seconds and 1e12. For the common shape, BANG C transfer-inclusive single-call wall time was 2049.11 µs FP16 and 2063.79 µs BF16. CNNL measured H2D and D2H separately and reports a session wall time including allocation, setup and repeated launches. That session number is not comparable with the BANG C single-call transfer-inclusive interval. CNNL peaked at 143.664 FP16 and 143.933 BF16 TFLOPS at 1024 cubed. The standalone BANG C correctness kernel uses scalar FP32 arithmetic and is not a competitive substitute for the vendor matrix path.

[BANG C measured data](../artifacts/logs/gemm_mlu8_bang.csv) · [CNNL measured data](../artifacts/logs/gemm_mlu8_cnnl.csv) · [Commands and build evidence](../artifacts/logs/gemm_mlu8_hardware.log)
