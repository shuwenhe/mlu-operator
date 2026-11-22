# Phase 15 Architecture Audit (Stage A, Read-Only)

## 1) Repository baseline

- Git HEAD: `8e8ec638862621ea76e933c246f4f246ba6fd7d8`
- Branch: `main` (tracking `origin/main`)
- Existing uncommitted tracked edits (preserved):
  - `integration/vllm_native_mlu/phase12_5/model.py`
  - `integration/vllm_native_mlu/phase12_5/run.py`
- Large pre-existing untracked evidence/doc trees exist under `artifacts/logs/phase13_*`, `artifacts/logs/phase14_*`, and `docs/PHASE13_*`, `docs/PHASE14_*`.

## 2) Runtime / platform facts (observed in this session)

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

## 3) Device inventory (observed)

- Enumerated accelerator nodes: `/dev/cambricon_dev0` … `/dev/cambricon_dev7`
- Auxiliary nodes present: `/dev/cambricon_ctl`, `/dev/cambricon_gdr`, `/dev/cambricon_ipcm0..7`
- `cnmon info` reports MLU590-M9, with card 0 shown and topology table listing MLU0..MLU7.
- Available device IDs for TP=8 mapping: `0..7`.

## 4) Baseline evidence verification against repository content

### Phase 13.3.2
- Evidence files present (`tp1.log`, `tp2.log`, `tp4.log`, `tp8.log`, `accuracy.csv`, `hardware.log`): **PASS**

### Phase 13.3.3
- Ordered path logs present for TP=1/2/4/8 and accuracy reports: **PASS**
- Output-axis alternative logs present (`output_axis/tp1..tp8.log`, `accuracy.csv`): **PASS**
- Frozen tolerance statement present in acceptance doc: **PASS**

### Phase 13.4
- Corrected closure summary exists and asserts:
  - `all_27_layer_gates_passed_on_8_ranks = true`
  - `all_27_layers_seen_in_full_model = true`
  - rank-device map `0..7 -> mlu:0..7`
- Corresponding corrected replay logs exist: **PASS**

### Phase 14.1
- TP=2 plugin registration path exists (`vllm.general_plugins` entry point + `ModelRegistry.register_model`): **PASS**
- Real `LLM.generate()` driver and TP2 model path exist: **PASS**
- TP=2 worker logs, cross-page logs, and fault-suite logs exist: **PASS**
- Fault rejection evidence (`negative_rejected`, metadata preflight failures) present: **PASS**

### Phase 14.2
- Lifecycle executor implementation and replay harness exist: **PASS**
- Repeated lifecycle report exists and states overall PASS (10/10): **PASS**
- Fresh run evidence bundle exists under `artifacts/logs/phase14_2/repeated_lifecycle/run_20261122_204946/`: **PASS**

## 5) Exact production call chain (current TP=2 proven path)

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

## 6) Stage A conclusions / hard blockers

1. **Current shell environment is not the frozen Vendor runtime**, so Torch/vLLM/Torch-MLU versions cannot be directly re-queried here.  
   - This is a **recording blocker only**, not a code blocker.
2. CNRT/CNNL/CNCL/CNCC versions are not directly queryable from current host shell tools; use frozen launcher/container evidence for authoritative runtime versions.
3. Existing dirty worktree must remain untouched (per constraints) during Phase 15 implementation.

