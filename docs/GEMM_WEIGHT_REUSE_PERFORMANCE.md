# Device-resident GEMM weight reuse performance

Measured on MLU 2 (`0000:6f:00.0`), driver 6.5.52, CNRT 7.7.3 and CNNL 2.2.18. The local checkpoint records `model.layers.0.self_attn.q_proj.weight` as BF16 `[3072,2048]`. Logical GEMM therefore uses N=3072 and K=2048. M=1 and M=128 are decode-like and prefill-like test batch sizes, respectively. The benchmark uses matched synthetic values at those dimensions for FP16 and BF16, plus the irregular 3×193×65 case. The same logical input data feeds all four paths. No full model inference was run.

## BF16 Q projection

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

## Timing scope and evidence

- Device kernel p50 and nearest-rank p95 use 30 individual CNRT notifier samples after five resident warmups. They exclude transfers and preparation. Effective TFLOPS uses 2MNK divided by kernel p50 seconds and 10^12.
- Transfer-inclusive invocation measures A H2D, one synchronized GEMM and C D2H. RHS is already prepared in this interval. Weight packing, initial RHS H2D and total preparation are separate columns. Amortized p50/p95 use nine independently timed reuse trials. The p95 of nine trials is the maximum. CNNL descriptor creation and ordinary reusable A/C/workspace allocations precede the timing scopes.
- The 72-row final CSV combines host and CNNL measurements from the first run with the optimized device-pack measurements from the second run. The same card, fixtures, matrix kernels and protocol were used for both. The scalar-packer experiment is preserved separately. These are point measurements, with no inference-level throughput claim.
- After hardware tests, all eight cards reported 0 MiB allocated at CNMON granularity, GLM53 remained exited, and both Team containers remained running.
- Outstanding work includes testing exhaustion under an isolated memory budget, running a memory checker, and integrating the explicit queue and handle lifetime contract into a serving adapter. No production source was changed.

[Performance CSV](../artifacts/logs/gemm_prepacked_performance.csv) · [Original scalar pack experiment](../artifacts/logs/gemm_prepacked_scalar_device_experiment.csv) · [Host/CNNL run log](../artifacts/logs/gemm_prepacked_performance_run.log) · [Optimized device run log](../artifacts/logs/gemm_prepacked_device_final_run.log) · [Checkpoint tensor metadata](../artifacts/logs/gemm_prepacked_checkpoint.log) · [Build transcript](../artifacts/logs/gemm_prepacked_build.log)
