# Phase 7 Layer 0 implementation

**Status: diagnostic instrumentation PASS. Numerical closure FAIL.** The existing explicit Reference MLA runtime executed Layer 0. Native MLU performs projections, normalization and rotation. Reference MLA, SiLU, multiplication and residual arithmetic remain explicit host stages. Persistent weights, queue completion and separate cache buffers remain under the existing runtime.

The recoverable snapshot is recorded as `source_snapshot` in the hardware log. It contains the initial Git revision, status, binary diff, source archive and original precision CSV. Existing fixture binaries and historical reports were used read-only. The accepted source hash check passed before and after the investigation.

The implementation sequence was to reproduce the historical fixture, extend trace-only instrumentation, replay all seven projections on identical low-precision operand words, compare the connected layer against the original checkpoint, and rerun device cleanup checks. The numerical evidence did not justify replacing the accepted GEMM or changing rounding order, so the experimental kernel and production integration stages remain BLOCKED.

- `runtime/native/r6_test.cpp` adds opt-in FP32 projection outputs and historical accumulators, actual KV-B input and rounded SiLU output to the existing fixture trace. Ordinary invocation still exercises the historical gate.
- `runtime/native/phase7/projections.py` independently replays all projection inputs through PyTorch and FP64, emits low-precision input words, and reports differences and ULP distance.
- `runtime/native/phase7/performance.py` extracts measured trace totals and tracked memory, with unmeasured scopes explicitly marked.

The hardware log captures the complete build and device commands. The replay log records each CPU-only comparison command. Native trace directories are recorded in the SDK results directory's `phase7_trace_manifest` and contain actual FP32 staging and low-precision values represented as FP32 carriers. The replay path log points to exact activation words, matched PyTorch words and FP64 outputs. BF16 checkpoint weight words are in the preserved fixture; FP16 weights are deterministically derived by conversion, with a per-dtype SHA256 reported for the exact replay words. The packing interface uses transposed logical `[K,N]` rows followed by the accepted packed RHS tile format. Full inference and MoE work remain outside this phase.

[Hardware log](../artifacts/logs/phase7_hardware.log) · [Reference replay log](../artifacts/logs/phase7_reference_replay.log) · [Adapter contracts](DEEPSEEK_OPERATOR_MAPPING.md)
