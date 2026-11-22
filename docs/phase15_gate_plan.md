# Phase 15 Gate Plan (TP=8 Vendor vLLM + mlu-operator)

## Scope and constraints carried forward

- Keep Vendor vLLM scheduling/execution path (`LLM.generate`) authoritative.
- No second inference engine, no custom scheduler, no KV lifecycle duplication.
- No CPU fallback / no silent PyTorch fallback.
- Preserve Phase 12.5.4 references and all Phase 13.x / 14.x historical evidence.
- New implementation only under `integration/vllm_native_mlu/phase15/`.
- New evidence only under `artifacts/logs/phase15/`.
- Final docs at `docs/PHASE15_TP8_IMPLEMENTATION.md`.

## Stage A status (this audit)

- Stage A deliverables produced:
  - `phase15_architecture_audit.md`
  - `phase15_source_inventory.txt`
  - `phase15_baseline_hashes.txt`
  - `phase15_gate_plan.md`
- Stage A production-code mutation: **none**

## Precondition matrix from existing evidence

| Prerequisite | Status | Evidence |
|---|---|---|
| Phase 13.3.2 native projection acceptance | PASS | `artifacts/logs/phase13_3_2/*` |
| Phase 13.3.3 ordered O TP=1/2/4/8 + output-axis alt | PASS | `artifacts/logs/phase13_3_3/*`, `docs/PHASE13_3_3_NUMERICAL_ACCEPTANCE.md` |
| Phase 13.4 TP8 all-layer/full-model closure | PASS | `docs/PHASE13_4_FINAL_EVIDENCE_CLOSURE.md`, `closure3_summary.json` |
| Phase 14.1 real TP=2 through `LLM.generate` | PASS | `artifacts/logs/phase14_1/*`, `docs/PHASE14_1_IMPLEMENTATION.md` |
| Phase 14.2 repeated lifecycle remediation | PASS (report) | `docs/PHASE14_2_REPEATED_LIFECYCLE_REPORT.md` |
| Direct query of runtime Torch/vLLM versions in current shell | BLOCKED | current shell lacks those modules |
| Direct query of CNRT/CNNL/CNCL/CNCC versions in current shell | BLOCKED | toolchain CLIs/libs not directly available in shell |

## Stage B — eight-card hardware initialization gate

### Goal
Validate rank-to-device and CNCL readiness for TP=8 on MLU590-M9.

### Required checks
1. Enumerate `/dev/cambricon_dev0..7` and ensure all are accessible.
2. Launch real TP=8 distributed initialization via Vendor path (not isolated kernel-only path).
3. Verify rank map `0..7 -> mlu:0..7`.
4. Verify no orphan workers and clean teardown after init/run.

### Evidence outputs
- `artifacts/logs/phase15/hw_init/`:
  - `env_snapshot.txt`
  - `rank_device_map.json`
  - `cncl_init.log`
  - `cleanup.log`

### Gate decision
- PASS only if all 8 ranks initialize and teardown cleanly through Vendor runtime.

## Stage C — TP=8 model integration gate

### Goal
Introduce TP=8 architecture registration and model runner path in `integration/vllm_native_mlu/phase15/` reusing proven phase13/14 building blocks.

### Required checks
1. Official ModelRegistry registration path for TP=8 architecture.
2. Native model construction via Vendor worker/model runner.
3. Correct TP=8 shard loading for DeepSeek-V2-Lite.
4. KV cache mapping consistent with scheduler metadata and block-table semantics.
5. CNCL collectives and Ordered O path retained (no fallback).

### Evidence outputs
- `artifacts/logs/phase15/integration/`:
  - `registration.log`
  - `model_load.log`
  - `weight_shard_audit.json`
  - `kv_contract_audit.json`

### Gate decision
- PASS only if real vendor worker path reaches native TP=8 model execution entry.

## Stage D — real `LLM.generate` TP=8 acceptance gate

### Goal
Run full production flow:
`User Request -> vLLM -> Vendor Worker/Model Runner -> mlu-operator -> CNRT/CNNL/CNCL -> 8x MLU590-M9`

### Required checks
1. Real prompt prefill + decode via `vllm.LLM.generate()` with TP=8.
2. Frozen logits parity and greedy token parity against accepted references.
3. Tolerance: `abs(error) <= 1e-5 + 1e-5 * abs(reference)`.
4. No metadata divergence acceptance and no hidden fallback.
5. Rank cleanup / FD / socket / process stability.

### Evidence outputs
- `artifacts/logs/phase15/e2e/`:
  - `tp8_worker.log`
  - `numerical.csv`
  - `token_parity.json`
  - `scheduler_contract.log`
  - `resource_stability.csv`
  - `final_gate_report.json`

### Gate decision
- PASS only if all numerical + lifecycle + cleanup checks pass on TP=8 real engine run.

## Final closure deliverable

- `docs/PHASE15_TP8_IMPLEMENTATION.md` must include:
  - exact call chain and file map
  - reproducible commands
  - acceptance table with PASS/FAIL per gate
  - explicit statement on blocked/non-blocked prerequisites
  - pointer list to all phase15 logs

