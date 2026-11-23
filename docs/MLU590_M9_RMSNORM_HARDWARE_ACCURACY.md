# RMSNorm hardware accuracy and reproduction

- [MLU590-M9 RMSNorm hardware accuracy](#mlu590-m9-rmsnorm-hardware-accuracy)
- [RMSNorm one-card reproduction](#deepseek-rmsnorm-hardware-acceptance)

<a id="mlu590-m9-rmsnorm-hardware-accuracy"></a>

## MLU590-M9 RMSNorm hardware accuracy

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

<a id="deepseek-rmsnorm-hardware-acceptance"></a>

## RMSNorm one-card reproduction

Obtain an explicit exclusive reservation from the workload owner, including the physical PCI ID. Use an approved compatible SDK context exposing only that card and its required control nodes. Confirm `cnmon info` and the identity of the reserved card, then pass the standalone CNRT selection, transfer, synchronization and cleanup probe. The diagnostics require one visible CNRT ordinal. The command-line flag does not itself authorize or allocate a device. Inspect authorized memory-checker diagnostics for odd widths and end-of-allocation accesses before accepting bounds safety.

Run from the project root in the compatible SDK environment. Use a fresh fixture and result directory to avoid overwriting preserved fixtures.

```sh
set -e
build=artifacts/build/rmsnorm-native
fixtures=artifacts/build/rmsnorm-hardware-fixtures
actual=artifacts/build/rmsnorm-hardware-actual
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/usr/local/neuware/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 integration/deepseek/rmsnorm_acceptance_fixtures.py artifacts/build/deepseek-fixtures "$fixtures"
operators/rmsnorm/native/build.sh "$build"
mkdir "$actual"
"$build/rmsnorm_diagnostics" --authorized "$actual" "$fixtures"/*.fixture
python3 integration/deepseek/rmsnorm_native_compare.py "$fixtures" --actual-dir "$actual"
```

Keep native normalized, pre-gamma and output buffers and the independent CNNL output from every case. Require finite values and passing FP32 tolerances, exact low-precision bits on fixtures where the numerical contract permits, unchanged guards, and no memory-checker errors. Compare final results with the CPU reference whose bits were checked against the exact PyTorch class.

After correctness passes, use `--authorized-benchmark` with a new result directory. It takes five warm-ups and thirty measured launches per shape and dtype. A queue notifier interval measures average device kernel latency with optional diagnostic buffers disabled. The independent end-to-end average includes two H2D copies, one launch, synchronization and one D2H copy per iteration. Allocation and reference computation are outside both intervals. The final timed result must agree with the diagnostic result. Record device/SDK identity, shape, dtype, warmups and repetitions with the CSV. The preserved CNNL call supplies an independent correctness baseline. Its kernel latency is not instrumented by this harness and no comparative speed claim is made.
