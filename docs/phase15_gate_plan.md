# Phase 15 architecture and acceptance plan

- [Phase 15 Architecture Audit (Stage A, Read-Only)](#phase15-architecture-audit)
- [Phase 15 Gate Plan (TP=8 Vendor vLLM + mlu-operator)](#phase15-gate-plan)

<a id="phase15-architecture-audit"></a>

## Phase 15 Architecture Audit (Stage A, Read-Only)

### 1) Repository baseline

- Git HEAD: `8e8ec638862621ea76e933c246f4f246ba6fd7d8`
- Branch: `main` (tracking `origin/main`)
- Existing uncommitted tracked edits (preserved):
  - `integration/vllm_native_mlu/phase12_5/model.py`
  - `integration/vllm_native_mlu/phase12_5/run.py`
- Large pre-existing untracked evidence/doc trees exist under `artifacts/logs/phase13_*`, `artifacts/logs/phase14_*`, and `docs/PHASE13_*`, `docs/PHASE14_*`.

### 2) Runtime / platform facts (observed in this session)

| Item | Observed value | Status |
|---|---|---|
| Host Python | `Python 2.7.18` (`/usr/bin/python`) | OBSERVED |
| Host Python3 | `Python 3.8.10` (`/usr/bin/python3`) | OBSERVED |
| Torch (host shell) | import failed (`No module named torch`) | BLOCKED (not in current shell env) |
| Torch MLU (host shell) | import failed (`No module named torch_mlu`) | BLOCKED (not in current shell env) |
| vLLM (host shell) | import failed (`No module named vllm`) | BLOCKED (not in current shell env) |
| CNMON | `6.5.52` | OBSERVED |
| MLU driver (from cnmon) | `v6.5.52` | OBSERVED |
| CNRT version command | unavailable (`cnrt-cli` missing) | BLOCKED |
| CNNL version command | unavailable in host path | BLOCKED |
| CNCL version command | unavailable in host path | BLOCKED |
| CNCC version command | unavailable in host path | BLOCKED |
| Runtime image digest (frozen launcher) | `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756` | OBSERVED (from launcher/docs evidence) |

### 3) Device inventory (observed)

- Enumerated accelerator nodes: `/dev/cambricon_dev0` … `/dev/cambricon_dev7`
- Auxiliary nodes present: `/dev/cambricon_ctl`, `/dev/cambricon_gdr`, `/dev/cambricon_ipcm0..7`
- `cnmon info` reports MLU590-M9, with card 0 shown and topology table listing MLU0..MLU7.
- Available device IDs for TP=8 mapping: `0..7`.

### 4) Baseline evidence verification against repository content

#### Phase 13.3.2
- Evidence files present (`tp1.log`, `tp2.log`, `tp4.log`, `tp8.log`, `accuracy.csv`, `hardware.log`): **PASS**

#### Phase 13.3.3
- Ordered path logs present for TP=1/2/4/8 and accuracy reports: **PASS**
- Output-axis alternative logs present (`output_axis/tp1..tp8.log`, `accuracy.csv`): **PASS**
- Frozen tolerance statement present in acceptance doc: **PASS**

#### Phase 13.4
- Corrected closure summary exists and asserts:
  - `all_27_layer_gates_passed_on_8_ranks = true`
  - `all_27_layers_seen_in_full_model = true`
  - rank-device map `0..7 -> mlu:0..7`
- Corresponding corrected replay logs exist: **PASS**

#### Phase 14.1
- TP=2 plugin registration path exists (`vllm.general_plugins` entry point + `ModelRegistry.register_model`): **PASS**
- Real `LLM.generate()` driver and TP2 model path exist: **PASS**
- TP=2 worker logs, cross-page logs, and fault-suite logs exist: **PASS**
- Fault rejection evidence (`negative_rejected`, metadata preflight failures) present: **PASS**

#### Phase 14.2
- Lifecycle executor implementation and replay harness exist: **PASS**
- Repeated lifecycle report exists and states overall PASS (10/10): **PASS**
- Fresh run evidence bundle exists under `artifacts/logs/phase14_2/repeated_lifecycle/run_20261122_204946/`: **PASS**

### 5) Exact production call chain (current TP=2 proven path)

1. **Model registration**
   - `integration/vllm_native_mlu/phase14_1/neurx_native_tp2-0.1.dist-info/entry_points.txt`
   - `integration/vllm_native_mlu/phase14_1/plugin.py`
   - `ModelRegistry.register_model("NeurXDeepSeekV2ForCausalLMTP2", "...phase14_1.model:NeurXDeepSeekV2ForCausalLM")`

2. **Engine entry**
   - `integration/vllm_native_mlu/phase14_1/engine.py`
   - Calls real `vllm.LLM(..., tensor_parallel_size=2, hf_overrides={"architectures":[ARCH]})`
   - Uses real `LLM.generate(...)`

3. **Vendor scheduler/executor**
   - Scheduler/Engine core path shown in captured vendor sources:
     - `artifacts/logs/phase14_2/vendor_core.py.txt`
     - `artifacts/logs/phase14_2/vendor_mlu_core.py.txt`
   - Multiprocess executor and worker lifecycle path:
     - `artifacts/logs/phase14_2/vendor_multiproc_executor.py.txt`
   - Model execution dispatched via `model_executor.execute_model(...)` and `collective_rpc(...)`

4. **Worker / model runner / model**
   - Worker-side access in acceptance probes uses `worker.model_runner.model`
   - Native model class:
     - `integration/vllm_native_mlu/phase14_1/model.py`
   - KV cache contract:
     - `integration/vllm_native_mlu/phase14_1/cache.py`
     - `integration/vllm_native_mlu/phase14_1/contracts.py`

5. **Native MLA and kernels**
   - `phase14_1/model.py` uses:
     - `phase13_4.mla.NativeCalls` (native QK/Softmax/AV + rope + paging)
     - `phase13_3_3.OrderedOutput` (ordered O path)
     - CNCL collectives through TP group
   - CNRT/CNNL/CNCL are reached through loaded native shared libs and distributed backend (`cncl`).

### 6) Stage A conclusions / hard blockers

1. **Current shell environment is not the frozen Vendor runtime**, so Torch/vLLM/Torch-MLU versions cannot be directly re-queried here.
   - This is a **recording blocker only**, not a code blocker.
2. CNRT/CNNL/CNCL/CNCC versions are not directly queryable from current host shell tools; use frozen launcher/container evidence for authoritative runtime versions.
3. Existing dirty worktree must remain untouched (per constraints) during Phase 15 implementation.

<a id="phase15-gate-plan"></a>

## Phase 15 Gate Plan (TP=8 Vendor vLLM + mlu-operator)

### Scope and constraints carried forward

- Keep Vendor vLLM scheduling/execution path (`LLM.generate`) authoritative.
- No second inference engine, no custom scheduler, no KV lifecycle duplication.
- No CPU fallback / no silent PyTorch fallback.
- Preserve Phase 12.5.4 references and all Phase 13.x / 14.x historical evidence.
- New implementation only under `integration/vllm_native_mlu/phase15/`.
- New evidence only under `artifacts/logs/phase15/`.
- The final architecture is documented in [Stage D architecture](STAGE_D_ARCHITECTURE.md).

### Stage A status (this audit)

- Stage A deliverables produced:
  - `phase15_architecture_audit.md`
  - [Phase 15 source inventory](phase15_source_inventory.txt), which records original paths at baseline capture
  - [Phase 15 baseline hashes](phase15_baseline_hashes.txt)
  - `phase15_gate_plan.md`
- Stage A production-code mutation: **none**

### Precondition matrix from existing evidence

| Prerequisite | Status | Evidence |
|---|---|---|
| Phase 13.3.2 native projection acceptance | PASS | `artifacts/logs/phase13_3_2/*` |
| Phase 13.3.3 ordered O TP=1/2/4/8 + output-axis alt | PASS | `artifacts/logs/phase13_3_3/*`, [PHASE13 3 3 NUMERICAL ACCEPTANCE](TENSOR_PARALLEL_MLA.md#phase13-3-3-numerical-acceptance) |
| Phase 13.4 TP8 all-layer/full-model closure | PASS | [PHASE13 4 FINAL EVIDENCE CLOSURE](TENSOR_PARALLEL_MLA.md#phase13-4-final-evidence-closure), `closure3_summary.json` |
| Phase 14.1 real TP=2 through `LLM.generate` | PASS | `artifacts/logs/phase14_1/*`, [PHASE14 1 IMPLEMENTATION](MULTICARD_ENGINE_INTEGRATION.md#phase14-1-implementation) |
| Phase 14.2 repeated lifecycle remediation | PASS (report) | [PHASE14 2 REPEATED LIFECYCLE REPORT](ENGINE_LIFECYCLE.md#phase14-2-repeated-lifecycle-report) |
| Direct query of runtime Torch/vLLM versions in current shell | BLOCKED | current shell lacks those modules |
| Direct query of CNRT/CNNL/CNCL/CNCC versions in current shell | BLOCKED | toolchain CLIs/libs not directly available in shell |

### Stage B — eight-card hardware initialization gate

#### Goal
Validate rank-to-device and CNCL readiness for TP=8 on MLU590-M9.

#### Required checks
1. Enumerate `/dev/cambricon_dev0..7` and ensure all are accessible.
2. Launch real TP=8 distributed initialization via Vendor path (not isolated kernel-only path).
3. Verify rank map `0..7 -> mlu:0..7`.
4. Verify no orphan workers and clean teardown after init/run.

#### Evidence outputs
- `artifacts/logs/phase15/hw_init/`:
  - `env_snapshot.txt`
  - `rank_device_map.json`
  - `cncl_init.log`
  - `cleanup.log`

#### Gate decision
- PASS only if all 8 ranks initialize and teardown cleanly through Vendor runtime.

### Stage C — TP=8 model integration gate

#### Goal
Introduce TP=8 architecture registration and model runner path in `integration/vllm_native_mlu/phase15/` reusing proven phase13/14 building blocks.

#### Required checks
1. Official ModelRegistry registration path for TP=8 architecture.
2. Native model construction via Vendor worker/model runner.
3. Correct TP=8 shard loading for DeepSeek-V2-Lite.
4. KV cache mapping consistent with scheduler metadata and block-table semantics.
5. CNCL collectives and Ordered O path retained (no fallback).

#### Evidence outputs
- `artifacts/logs/phase15/integration/`:
  - `registration.log`
  - `model_load.log`
  - `weight_shard_audit.json`
  - `kv_contract_audit.json`

#### Gate decision
- PASS only if real vendor worker path reaches native TP=8 model execution entry.

### Stage D — real `LLM.generate` TP=8 acceptance gate

#### Goal
Run full production flow:
`User Request -> vLLM -> Vendor Worker/Model Runner -> mlu-operator -> CNRT/CNNL/CNCL -> 8x MLU590-M9`

#### Required checks
1. Real prompt prefill + decode via `vllm.LLM.generate()` with TP=8.
2. Frozen logits parity and greedy token parity against accepted references.
3. Tolerance: `abs(error) <= 1e-5 + 1e-5 * abs(reference)`.
4. No metadata divergence acceptance and no hidden fallback.
5. Rank cleanup / FD / socket / process stability.

#### Evidence outputs
- `artifacts/logs/phase15/e2e/`:
  - `tp8_worker.log`
  - `numerical.csv`
  - `token_parity.json`
  - `scheduler_contract.log`
  - `resource_stability.csv`
  - `final_gate_report.json`

#### Gate decision
- PASS only if all numerical + lifecycle + cleanup checks pass on TP=8 real engine run.

### Final closure deliverable

- The original plan called for the final implementation report to include:
  - exact call chain and file map
  - reproducible commands
  - acceptance table with PASS/FAIL per gate
  - explicit statement on blocked/non-blocked prerequisites
  - pointer list to all phase15 logs
