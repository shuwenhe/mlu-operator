# DeepSeek-V2-Lite RMSNorm hardware acceptance

**BLOCKED on hardware.** Host CNMON exited 0 and showed vLLM workers on all eight MLU590-M9 cards. No exclusive allocation or explicit authorization was established. The verification image had no Cambricon nodes. No device kernels, CNRT device transfers, CNNL numerical tests or benchmarks ran. Production workloads and mappings were left untouched.

## Live allocation recheck, 2026-11-21

The user reported that all eight cards were allocated to their work. The host check on `tza-0a09-ai01-em9` showed concurrent use on every card, so no card met the exclusive-use requirement for this test.

- Host `cnmon info` exited 0. Cards 0–7 were initialized. Each card had a resident vLLM worker using 93,556 MiB. Card 0 also listed seven other worker contexts with 0 MiB charged in the sample.
- Worker PIDs 3642753–3642760 were still present and their cgroups matched the running privileged `glm53_serve` container. Running `ta-mlu-team1` and `ta-mlu-team2` also retained explicit compute and IPCM mappings for cards 0–3 and 4–7 respectively, with the shared control node. These mappings show configured access, and the CNMON worker records show live use.
- Host glibc remains 2.31 and the installed host Neuware library directory exposes CNDev 3.12.4. The existing Ubuntu 24.04 image remains available for a future isolated SDK run, and the prior offline link evidence still applies. Host CNDEV health does not establish CNRT visibility or runtime compatibility in the intended single-card context. Those checks were deferred because exclusivity failed.
- Decision: **BLOCKED before launch**. No device was selected, no test container was started, and no mappings, processes or fixtures were changed. No FP16/BF16 device outputs, guard results, CNNL data or timings were collected.

## Static audit

| Area | Finding |
| --- | --- |
| ABI and device selection | The FP16 and BF16 entry points accept positive rows/width, positive finite epsilon, device pointers and a caller-owned queue. They reject null required pointers. The API relies on the caller to provide live, nonoverlapping buffers on the selected device. The launcher does not inspect allocation sizes or change the current device. |
| Queue lifecycle | The harness selects its only visible device, creates one native queue and checks launch status and queue synchronization before reading output. CNNL is evaluated through the preserved wrapper in a separate call, with its own queue and gamma contract. Native timing excludes CNNL. The harness destroys its own queue. |
| Buffers | For R rows and H columns, X, output and rounded diagnostics each require 2RH bytes. Gamma needs 2H bytes and normalized diagnostics 4RH bytes. The harness allocates the exact payload plus 128-byte canaries on both sides, and verifies all canaries after synchronization. Advancing the allocation base by 128 preserves any base alignment through 128 bytes. It does not align odd row starts. Canaries detect observed writes into adjacent guards only. They cannot prove absence of out-of-bounds reads or all illegal writes. |
| Tile bounds | Each task owns whole rows via a task-stride loop. For each tile, count is min(128, remaining width). The kernel transfers only count elements, zero-fills scratch tails, sums only logical elements, and divides by the full logical width. Row and offset products use signed 64-bit arithmetic. |
| Odd-width transfers | Widths 65 and 127 each use one partial tile. Width 193 uses 128+65. Packed odd rows can begin at offsets divisible only by the element size. Partial two-byte transfers and row-origin alignment still require device execution with the target SDK and a supported memory checker or equivalent tracing. Passing a compiler build or preserving canaries alone cannot close this gate. |
| Per-task storage | Two 128-element low-precision arrays and three 128-element FP32 arrays total 2,048 requested NRAM bytes before compiler layout. No WRAM or shared SRAM is requested by the source. Launch uses up to 32 Block tasks. Effective resource limits, spills and occupancy have not been validated on the card. |
| Numerical semantics | FP32 normalization and explicit narrow-before-gamma and output narrowing are present. The serial sum, device reciprocal square root, and low-precision conversion may differ from PyTorch near rounding boundaries. Fixture-level bit checks remain strict and mismatches require diagnosis. |

## Verified evidence

- **Build.** CNCC 5.7.6 generated both `mtp_592` fatbins and launch symbols. Assembly contains `recipsqrt.nram.f32` and both narrowing conversions per dtype. The updated diagnostics harness linked against the existing CNRT/CNNL bundle in the isolated Ubuntu 24.04 image. Its no-argument invocation exited 2 before the first explicit device API call. Build log is `artifacts/logs/rmsnorm_hardware_acceptance_build.log`.
- **CPU.** The separate acceptance fixture set covers widths 512, 2048, 65, 127 and 193 with both input dtypes. The new generator copies three existing cases per dtype and creates new 127/193 fixtures with distinct adjacent rows, a zero row and nonuniform gamma. The independent FP32 reference matched the exact checkpoint PyTorch CPU class across **17,540 outputs** (8,770 per dtype) with zero maximum absolute, maximum relative and mean absolute error. CPU stage self-check also passed. The earlier **27,912** CPU comparisons were preserved. New per-case evidence is in `artifacts/logs/rmsnorm_hardware_cpu_parity.csv`.
- **Device.** Normalized FP32 error, pre-gamma and final bit parity, bounds checks, CNNL results and both timing numbers are pending. The stage comparator reports max absolute, max relative with 1e-6 floor, mean absolute, nonfinite counts and mismatches. It scores the CNNL FP32 contract independently from the DeepSeek parity result.

## One-card procedure after authorization

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
