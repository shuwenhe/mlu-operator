# Phase 14.1 — Vendor engine integration

The isolated module is `integration/vllm_native_mlu/phase14_1`. Its distribution metadata exposes the official `vllm.general_plugins` entry point. Both spawned Vendor workers registered the lazy TP=2 architecture, resolved `NeurXDeepSeekV2ForCausalLM` and loaded the checkpoint through the official loader. Each worker materializes only its own MLA slices on-device. Non-MLA decoder work remains replicated.

The Vendor allocation is mapped without a copy to two BF16 views with eight heads. Scheduler spans, physical IDs and rank agreement are checked before numerical collectives. The native MLA keeps the frozen layout and exact Ordered O conversion sequence. The initial real prefill found that the current replay helper returns three values. The TP=2 adapter now directly invokes the accepted Ordered O primitive and exposes its actual FP32 output; it does not invoke the replay helper's legacy diagnostic SUM.

The model's optional traces record worker, local rank, stream, operator, dtype and shapes. The first numerical run additionally records raw tensors, filtered to real scheduled forward calls. Supplemental TP=1 diagnostic boundaries are separate from frozen references. The production entrypoint and legacy model sources were left as found, including pre-existing worktree edits; the frozen TP=1 regression mounted the preserved HEAD source over those files only inside the test container.

## Reproduce

From the repository root on the pinned two-card host:

```bash
set -euo pipefail
driver=integration/vllm_native_mlu/phase14_1/run_tp2.sh
out=$(mktemp -d "$PWD/artifacts/logs/phase14_1/replay.XXXXXX")
NEURX_PHASE14_1_OUT="$out" bash "$driver" > "$out/worker.log" 2>&1
```

The driver uses the frozen image digest, read-only checkpoint, frozen prompt bytes and compiled native pager. It requests two generated tokens to exercise an actual decode step. Follow the accepted length-one gate with `--expanded` and the separate isolated lifecycle probes. The raw first attempt and its successful retry are preserved under distinct paths.

The frozen regression uses `run_tp1_regression.sh` with a fresh `NEURX_PHASE14_1_OUT` directory. The supplemental boundary collector consumes only the accepted dump and writes its own separate outputs. `boundary_audit.py` compares the successful two-card retry against the immutable and supplemental boundaries; `weight_audit.py` independently checks the official loader's recorded shard bytes.

For scheduler experiments, run the same two-card driver with `--cross-page` or `--negative invalid-block` (also `invalid-slot`, `divergent`, and `missing-metadata`), each in a separate fresh directory. Faults are armed on the actual native model through the declared acceptance RPC. Source snapshots, application diffs and SHA-256 evidence are in the Phase 14.1 artifact directory.

The isolated `--negative-suite` runs the four scheduler metadata faults in a single same-process Vendor EngineCore. Its acceptance driver aborts failed requests before trying the next prompt. The separate `NEURX_ENGINE_MULTIPROC=1` experiment exercises a standalone Vendor EngineCore process. The network-disabled fault container used local callable serialization to arm its trusted RPC fixture. The standard launcher leaves that setting disabled.
