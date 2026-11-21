# MLU590-M9 RMSNorm hardware accuracy

**PASS for the ten acceptance fixtures on host MLU 0.** A disposable context mapped only compute node 0, IPCM 0 and shared control. The diagnostic executable required exactly one CNRT-visible device. FP16 and BF16 covered widths 65, 127, 193, 512 and 2048, including checkpoint gamma and irregular synthetic rows.

| Check | FP16 | BF16 |
| --- | ---: | ---: |
| Elements per stage | 8,770 | 8,770 |
| FP32 normalized maximum absolute error | 1.1920929e-7 | 1.1920929e-7 |
| FP32 normalized maximum relative error | 1.0625677e-7 | 1.0625677e-7 |
| Pre-gamma bit mismatches | 0 | 0 |
| Final bit mismatches | 0 | 0 |
| NaN/Inf on gated outputs | 0 | 0 |
| CNNL FP32-contract maximum absolute error | 2.3841858e-7 | 2.3841858e-7 |

The only nonzero BANG C normalized errors were at width 127, with mean absolute error 1.955533e-8 for that fixture in each dtype. Buffer canary and cleanup checks returned success. This establishes observed behavior on these fixtures. Full memory instrumentation and arbitrary-input parity remain unverified. The independent CPU reference for these fixtures was previously verified against the checkpoint PyTorch CPU class. CNNL passed its own FP32 contract. Its output differs from the DeepSeek low-precision ordering as expected.

Repeat validation with the benchmark path passed as well. With five warmups and 30 measured launches, BF16 kernel latency ranged 5.553–50.555 µs and transfer-inclusive latency 37.338–86.381 µs. FP16 ranges were 5.599–50.563 µs and 35.958–86.464 µs. Timings exclude allocation, with diagnostics disabled in the timed native launch. The baseline CNNL RMSNorm call is a correctness baseline without matching timing instrumentation.

[Device accuracy CSV](../artifacts/logs/rmsnorm_mlu8_accuracy.csv) · [Device commands and compile evidence](../artifacts/logs/rmsnorm_mlu8_hardware.log) · [Timing log](../artifacts/logs/rmsnorm_mlu8_performance.log)
