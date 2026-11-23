# Multi-card integration in the vendor engine

- [Native MLU production architecture audit](#mlu-operator-architecture-audit)
- [Phase 13.5.1 — Production Integration Architecture Audit](#phase13-5-1-production-architecture-audit)
- [Phase 13.5.1 — Integration Plan (No Production Code Changes Yet)](#phase13-5-1-integration-plan)
- [Phase 13.5 — DeepSeek-V2-Lite Native MLU TP=8 Production Integration Plan](#phase13-5-implementation)
- [Phase 14.1 — Vendor engine integration](#phase14-1-implementation)
- [Phase 14.1 — Numerical acceptance](#phase14-1-numerical-acceptance)

<a id="mlu-operator-architecture-audit"></a>

## Native MLU production architecture audit

### Current execution graph

The frozen Vendor engine starts from `LLM.generate()`, then passes scheduled requests through EngineCore, the MLU worker and model runner, model loading via the process-local ModelRegistry, model forward, logits and sampling. The worker receives the engine's block table and slot mapping through ForwardContext. The scheduler KV manager owns block allocation and release. TP workers use the Vendor CNCL device group and MLUCommunicator. Gloo CPU control groups are separate from device tensor collectives. [Vendor source inventory](../artifacts/logs/phase13_5/phase13_5_1_source_inventory.txt)

Phase 13.4 uses a standalone replay driver with real model weights, generated tokens and 27 layers on eight MLUs. It has a local Pages allocator and directly invokes the native MLA and replicated decoder helpers. It provides numerical evidence for those operations. In the real engine the corresponding pages and worker metadata must come from vLLM. [Corrected replay evidence](TENSOR_PARALLEL_MLA.md#phase13-4-final-evidence-closure)

### Operator inventory and gaps

| Surface | Available native path | Production gap |
|---|---|---|
| Dense GEMM and RMSNorm | Packed BF16 GEMM, FP32 accumulation, frozen native BF16 norm; Phase 13.2 Column/Row Linear and distributed collectives | Protect TP=1. Bind distributed weights and buffer lifetime to worker loading. |
| MLA Q, KV-A, norm, KV-B and RoPE | Q/KV-B shard by head, KV-A and its norm replicate. Shared rotary K and frozen YaRN tables. | Phase 13.4 local-head adapter needs a runner-facing model class. TP=2 and TP=4 require their own head counts and cache types. |
| QK, causal mask, Softmax and AV | Dynamic head-count native launchers used by eight-rank replay. | Validate actual scheduler batches and current worker streams. |
| MLA O | Ordered input-axis O and full-K output-axis O both passed isolated frozen numerical gates. Independent-fold Row SUM fails BF16 acceptance. | Select explicitly. Output-axis O requires BF16 input gather, full-K local output tiles and FP32 output gather before one BF16 cast and residual. |
| Paged KV | Frozen TP=1 worker layout uses 16 heads; Phase 13.4 uses two heads and independently verified page reuse. | Current production cache spec, reshape and launcher still have 16-head contracts. Connect local head counts to worker allocation, real block ownership and release. |
| Router projection, softmax and Top-K | Frozen native router projection, native FP32 softmax statistics and native rank/weight selection in `_moe`. | Measure accuracy/performance on worker batches and routing skew before changing kernels. |
| Expert dispatch and GEMM | Router IDs are copied to host; the current loop selects each token/expert pair and invokes native gate/up GEMMs then native SwiGLU and down GEMM. Shared experts have an explicit path. | Grouped, device-resident dispatch and batched expert GEMM are the main gaps. The reference path remains replicated across TP ranks. |
| Expert combine and reduction | Native local routed/shared expert combine with frozen weights. | No accepted EP token exchange or distributed expert combine in the NeurX path. Audit Vendor all-to-all modes later against the checkpoint's 64 routed experts, two shared experts and top-6 policy. |
| Embedding, final norm and logits | Frozen native embedding, final RMSNorm and chunked LM-head GEMM. Phase 13.4 replicates them and the remaining decoder work. | Validate real engine logits and greedy output before sharding or optimizing the tail. |

[Current model path](../integration/vllm_native_mlu/phase12_5/model.py) · [Native cache contract](../integration/vllm_native_mlu/phase12_5/cache.py) · [Phase 13.4 model replay](../integration/vllm_native_mlu/phase13_4/full_model.py)

### Worker registration and worktree findings

Phase 13.5.3 failed registry resolution in spawned worker processes. ModelRegistry is process-local. The archived Vendor source audit identifies the official `vllm.general_plugins` entry-point loader. Phase 13.5.4 subsequently reached the native constructor on **both** TP=2 workers, then rejected the frozen TP=1 contract. This is the latest observed stop. The current worktree has pre-existing edits that widen that constructor check and add import-time registration in `run.py`. The diff adds no per-rank native MLA/cache path, and no passing real-engine TP=2 evidence accompanies it. Archived TP=1 acceptance predates the current draft. [Registry audit](../artifacts/logs/phase13_5_4/worker_registration_audit.md) · [Latest TP=2 traceback](../artifacts/logs/phase13_5_4/tp2.log) · [Worktree snapshot](../artifacts/logs/phase14/unstaged.patch)

Use deterministic worker-side registration through the official installed general-plugin extension when this path is implemented. Keep the legacy TP=1 architecture entry clustered with its frozen initializer. The experimental TP=2 model should have a separate architecture name and request the engine's actual group and scheduler metadata. EP uses a separate expert ownership and token-dispatch plan and is outside the first engine milestone.

<a id="phase13-5-1-production-architecture-audit"></a>

## Phase 13.5.1 — Production Integration Architecture Audit

### Scope
- Repo audit root: `/app/shuwen/mlu-operator`
- Vendor runtime source roots (frozen image):
  - `/torch/venv3/pytorch_infer/lib/python3.12/site-packages/vllm`
  - `/workspace/vllm-v0.25.1+mlu0.16.0.pt212/vllm_mlu`
- Source inventory evidence: `artifacts/logs/phase13_5/phase13_5_1_source_inventory.txt`
- Source hash evidence: `artifacts/logs/phase13_5/phase13_5_1_source_hashes.txt`

### 1) Actual model registration and loading path

#### Local integration entry (already in use)
- Native registration call site: `ModelRegistry.register_model(...)` in [run.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_5/run.py).
- Runtime architecture override is passed via `hf_overrides={"architectures":[ARCHITECTURE]}` in [run.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_5/run.py) and [run_acceptance.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_6/run_acceptance.py).

#### Vendor vLLM resolution contract
- LLM user entry + generation API in `vllm/entrypoints/llm.py` (`class LLM`, `def generate`).
- Model registry contract in `vllm/model_executor/models/registry.py`:
  - `_ModelRegistry.register_model(...)`
  - `_ModelRegistry.inspect_model_cls(...)`
  - `_ModelRegistry.resolve_model_cls(...)`
- Worker-side model loading path in `vllm_mlu/v1/worker/gpu_model_runner.py`:
  - `MLUModelRunner.load_model(...)`
  - `self.model = model_loader.load_model(...)`

结论：当前“真实生产路径”是 `LLM(...) -> EngineCore/worker -> GPUModelRunner.load_model -> ModelRegistry resolve/inspect -> instantiated model class`。Phase 13.5 集成必须进入这个链路，不能只停留在 probe/replay。

### 2) Real execution path: load -> prefill/decode -> token output

#### End-to-end high-level path
1. `LLM(...)` initializes engine (`llm.py`).
2. `LLM.generate(...)` enters engine scheduling.
3. MLU worker is selected and constructed via `vllm_mlu/v1/worker/gpu_worker.py` as `MLUModelRunner`.
4. `MLUModelRunner.load_model(...)` loads the model.
5. Scheduling stage provides request/block/slot metadata; runner constructs attention metadata and executes forward.
6. Output logits and complete token selection (corresponding to `compute_logits` contract in repo [model.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_5/model.py)).

#### Prefill/decode metadata flow (actual contracts)
- Repo native model-side reading:
  - `get_forward_context().attn_metadata`
  - `vllm_mlu...get_common_metadata()`
  - 依赖 `block_table_tensor`、`slot_mapping`
  - 位置： [model.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_5/model.py)
- Vendor side metadata construction/passing:
  - `vllm_mlu/v1/attention/backends/utils.py` (`get_common_metadata`, `MLUCommonAttentionMetadata`)
  - `vllm_mlu/v1/worker/gpu_model_runner.py` (`_get_slot_mappings`, block table/slot mapping wiring)

结论：Phase 13.5 必须对齐 `MLUCommonAttentionMetadata + ForwardContext` 的现有 contract，不能自建“旁路 metadata 接口”。

### 3) TP process group, communicator, and CNCL ordering

- TP init contract (verified by repo probes):
  - `init_distributed_environment(..., backend="cncl")`
  - `initialize_model_parallel(..., backend="cncl")`
  - 见 [cncl_probe.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase13_1/cncl_probe.py), [probe.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase13_4/probe.py)
- Communicator implementation:
  - `MLUCommunicator` in `vllm_mlu/distributed/device_communicators/mlu_communicator.py`
  - 暴露 `all_reduce / all_gather / reduce_scatter`，底层 `torch.distributed` + CNCL group

Conclusion: Phase 13.5's ordered O must be embedded in existing TP collective order and must not disrupt the scheduler/runner's existing collective sequence.

### 4) KV cache lifecycle (allocation/write/read/free/reuse)

- Scheduler holds `KVCacheManager` (`vllm/v1/core/sched/scheduler.py` -> `KVCacheManager`).
- `KVCacheManager.free(...)` 是 release/reclamation contract（`vllm/v1/core/kv_cache_manager.py`）。
- Worker runner holds slot mapping / block table, cache addressing during prefill/decode:
  - `_get_slot_mappings(...)`
  - block table commit / mapping usage
  - `gpu_model_runner.py` 中多处 `slot_mapping` / `block_table` 路径（inventory 已记录行号）。
- Repo native model cache read/write contract：
  - `cache.write(k, v, metadata.slot_mapping[...])`
  - `cache.read(metadata.block_table_tensor[...], span.length)`
  - 见 [model.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_5/model.py)

Conclusion: Distributed KV lifecycle integration boundary is clear: scheduler allocation and reclamation, runner metadata orchestration, model metadata consumption. Phase 13.5 should not create an independent cache manager.

### 5) Ordered O (Phase 13.3.3) integration boundary

Existing verification implementation is in replay/probe path:
- [mla.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase13_4/mla.py): ordered FP32 accumulation + BF16 cast boundary evidence
- [probe.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase13_4/probe.py), [full_model.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase13_4/full_model.py): gate replay

Production integration boundary (must maintain Vendor contract):
1. Only replace internal numerical strategy in attention O aggregation path (local partial -> CNCL sum -> BF16 cast -> residual).
2. Input/output tensor shapes, dtype, and slot/block metadata interfaces remain consistent with existing worker/model paths.
3. Do not modify TP=1 production path behavior (can only be enabled under TP>1 conditions).

### 6) TP=1 regression protection boundary

- TP=1 regression entry points already exist in historical acceptance harnesses (e.g., [run_acceptance.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_6/run_acceptance.py) and 13.x probes).
- Mandatory requirements:
  - TP=1 code path logic unchanged (functionality and numerical baseline frozen).
  - TP=2/4/8 new integration must be explicitly gated by conditions.
  - Any TP=1 deviation should directly determine Phase 13.5 BLOCKED.

### 7) Blockers / risks identified

1. Vendor source ABI/signature risk  
   - `vllm_mlu` worker/runner interface is complex and version-dependent; risk of directly accessing internal private structures is high.  
   - Mitigation: Only do minimal replacement at existing metadata + runner call boundary.

2. Collective ordering risk  
   - If ordered O changes modify the relative order of reduce/allgather, there is risk of deadlock/numerical drift.  
   - Mitigation: Use existing communicator call order; add validation coverage for invalid metadata injection scenarios.

3. KV metadata consistency risk  
   - prefill/decode mixed batching 依赖 slot mapping 与 block table 一致性。  
   - 缓解：将 metadata 一致性和回收复用放在首个硬门槛之前验证。

4. Runtime-path mismatch risk  
   - 仅通过 probe/replay 通过不足以代表生产可用。  
   - 缓解：Phase13.5 必须以真实 `LLM.generate` engine 路径为主验收对象。

### 8) Static checks and independent source review

- Static check:
  - `py_compile` 写入 `__pycache__` 受权限限制（已记录并切换方案）。
  - 采用 AST 非写入语法检查：`AST_PARSE_OK 6`（审计模块语法通过）。
- Independent source review:
  - 采用“仓内调用点 + Vendor源码行号”双路径交叉审查；
  - 核对项：注册路径、worker load path、slot/block metadata、communicator collectives、KV free/reclaim；
  - 审查结论：当前边界可实现生产集成，前提是不破坏 TP=1 与 scheduler/metadata contracts。

### Audit status
- Phase 13.5.1 Production architecture audit: **COMPLETE**
- Production code modification: **NOT STARTED (as required)**

<a id="phase13-5-1-integration-plan"></a>

## Phase 13.5.1 — Integration Plan (No Production Code Changes Yet)

### Objective
在不破坏 Vendor runtime 合约与 TP=1 生产路径前提下，将已独立验证的 Native MLA（含 Phase 13.3.3 ordered O）接入实际 vLLM MLU 生产执行链路。

### Frozen constraints (hard gates)
1. 不修改 Phase 12.5.4 frozen references / fixtures / historical logs。
2. 不升级 Vendor vLLM / Neuware / 镜像。
3. 不引入 silent fallback。
4. 不改变 TP=1 生产行为。
5. 数值门限固定：`abs(error) <= 1e-5 + 1e-5 * abs(reference)`。

### Integration boundaries (from audit)
1. **Registration/Loading boundary**
   - 保持 `ModelRegistry` + `hf_overrides.architectures` 的官方路径。
   - 不创建绕过 registry 的模型注入路径。

2. **Worker/Runner boundary**
   - 仅在 `vllm_mlu` worker/model-runner 的真实前向路径中接入 Native MLA。
   - 不绕过 scheduler，不绕过 metadata builder。

3. **Metadata/KV boundary**
   - 仅消费既有 `block_table_tensor` / `slot_mapping` / forward context metadata。
   - 不实现并行的“第二套 KV 生命周期”。

4. **Ordered O boundary**
   - 在 TP>1 attention O 聚合内部启用 ordered FP32 accumulation。
   - 保持输入输出 contract（shape/dtype/layout）与现有链路一致。

### Execution plan (incremental)

#### Step 0 — Safety preconditions
- 保持当前证据链不覆盖写。
- 为 Phase13.5 单独输出目录：`artifacts/logs/phase13_5/`。

#### Step 1 — TP=1 regression guard first
- 建立 TP=1 快速回归门（prefill + decode + logits + tokens）。
- 任何 TP=1 偏移立即停止后续步骤并标记 BLOCKED。

#### Step 2 — TP=8 production wiring (minimal change)
- 将 Native MLA 接入 worker/model-runner 的真实执行路径（非 probe）。
- 先完成 prefill 单请求，再开放 decode，再开放 mixed batching。

#### Step 3 — Distributed KV lifecycle verification
- 验证 allocation/write/read/free/reuse。
- 注入 invalid metadata，验证一致错误退出且无 collective deadlock。

#### Step 4 — Numerical acceptance (frozen references)
- 长度：1/7/17/65；prefill + ≥3 decode。
- 边界：Q/K/V、RoPE、QK、Softmax、AV、O partial、ordered FP32、BF16 O、residual、logits、token IDs。
- 一旦出现“首个无法解释的 mismatch”，立即停机保留证据。

#### Step 5 — End-to-end engine acceptance
- 使用真实 `LLM.generate` 路径执行 TP=1/2/4/8。
- 覆盖单请求、并发、多长度混合、重复生成、engine restart。

#### Step 6 — Performance and reliability
- 性能：可支持配置下每组 ≥100 次，slowest-rank p50/p95。
- 可靠性：连续 ≥1h，验证无失败/无 NaNInf/无死锁/无 rank divergence/无内存异常增长。

#### Step 7 — Independent review before phase close
- 对边界、collective顺序、KV生命周期、BF16 rounding、TP=1保护做独立复审。

### Gate matrix
- G1 Registration/loading path intact: PASS/BLOCKED
- G2 TP=1 regression: PASS/FAIL
- G3 TP=8 production path executes prefill+decode: PASS/FAIL
- G4 Distributed KV lifecycle correctness: PASS/FAIL/BLOCKED
- G5 Numerical parity vs frozen references: PASS/FAIL
- G6 End-to-end multi-TP engine acceptance: PASS/FAIL/BLOCKED
- G7 Performance evidence complete: PASS/BLOCKED
- G8 1h reliability evidence complete: PASS/FAIL/BLOCKED

### Deliverable mapping
- Architecture audit: [PHASE13 5 1 PRODUCTION ARCHITECTURE AUDIT](#phase13-5-1-production-architecture-audit)
- Plan: [PHASE13 5 1 INTEGRATION PLAN](#phase13-5-1-integration-plan)
- Source inventory: `artifacts/logs/phase13_5/phase13_5_1_source_inventory.txt`
- Hash manifest: `artifacts/logs/phase13_5/phase13_5_1_source_hashes.txt`

### Current status
- 13.5.1 审计与计划：**COMPLETE**
- 生产代码集成：**NOT STARTED**

<a id="phase13-5-implementation"></a>

## Phase 13.5 — DeepSeek-V2-Lite Native MLU TP=8 Production Integration Plan

### Scope and constraints
- Repository: `/app/shuwen/mlu-operator`
- Frozen runtime/image: `teleavatar-mlu:v0.1` (`sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`)
- Hardware target: 8 × MLU590-M9
- Frozen tolerance: `abs(error) <= 1e-5 + 1e-5 * abs(reference)`
- No modification to Phase 12.5.4 frozen references/fixtures.
- No silent fallback and no PASS without hardware evidence.

### Preflight audit (completed before code modification)
- Audit log: `artifacts/logs/phase13_5/preflight_audit.log`
- SHA manifest: `artifacts/logs/phase13_5/preflight_sha256.txt`
- Verified presence of mandatory baseline evidence:
  - Phase 12.5.4 acceptance and source hashes
  - Phase 13.1 CNCL probe evidence
  - Phase 13.2 acceptance
  - Phase 13.3.1 architecture/mapping docs
  - Phase 13.3.2 failure evidence
  - Phase 13.3.3 ordered-O acceptance
  - Phase 13.4 corrected replay closure

### Integration strategy (incremental)
1. **Boundary audit and wiring design**
   - Locate production entry points used by Vendor vLLM worker init and model execution.
   - Define TP=8-only integration boundary; preserve TP=1 path behavior.
   - Confirm process-group/communicator ownership and stream ordering invariants.

2. **Production path integration (minimal/surgical)**
   - Integrate validated native MLA path into actual Vendor vLLM execution path:
     - Query/KV-A/KV-B projections
     - RoPE
     - two-head local attention
     - rank-local KV cache write/read
     - ordered O compatibility
     - FP32 accumulation, BF16 cast, residual add
   - Keep TP=1 production path unchanged and protected by regression gate.

3. **Distributed KV lifecycle validation**
   - Validate block/slot ownership and metadata consistency across ranks.
   - Validate prefill/decode mixed scheduling, block release/reuse.
   - Inject invalid metadata and verify coherent failure without collective deadlock.

4. **Numerical acceptance (frozen references)**
   - Context lengths: 1/7/17/65
   - Prefill + ≥3 decode steps
   - Compare required boundaries (Q/K/V, RoPE, QK/softmax/AV, O FP32 partial, ordered FP32 sum, BF16 O, residual, logits, token IDs).
   - Stop at first unexplained mismatch and preserve logs.

5. **End-to-end Vendor vLLM engine acceptance**
   - Real engine run (not isolated probe): TP=1/2/4/8
   - Single + concurrent + mixed requests, restart cycle
   - Deterministic token checks where supported.

6. **Performance and reliability**
   - Performance: ≥100 repetitions/config where supported; report slowest-rank p50/p95.
   - Reliability: continuous ≥1h hardware run, no deadlock/divergence/leak/fallback.

7. **Independent review**
   - Conduct a dedicated read-only review of integration boundaries, TP ownership, CNCL ordering, BF16 rounding, RoPE positions, KV lifecycle, failure handling, TP=1 protection.

### Planned outputs for this phase
- [Phase 13.5 frozen gate evidence](../artifacts/logs/phase13_5_2/final_gate_report.json)
- [Phase 14.2 repeated lifecycle evidence](ENGINE_LIFECYCLE.md#phase14-2-repeated-lifecycle-report)
- [Phase 13.5 production architecture audit](#phase13-5-1-production-architecture-audit)
- `artifacts/logs/phase13_5/` (raw logs, per-rank CSV, summaries, cleanup evidence, SHA256 manifests)

### Current status
- **Phase 13.5:** IN PROGRESS (audit complete, implementation not started in this document).

<a id="phase14-1-implementation"></a>

## Phase 14.1 — Vendor engine integration

The isolated module is `integration/vllm_native_mlu/phase14_1`. Its distribution metadata exposes the official `vllm.general_plugins` entry point. Both spawned Vendor workers registered the lazy TP=2 architecture, resolved `NeurXDeepSeekV2ForCausalLM` and loaded the checkpoint through the official loader. Each worker materializes only its own MLA slices on-device. Non-MLA decoder work remains replicated.

The Vendor allocation is mapped without a copy to two BF16 views with eight heads. Scheduler spans, physical IDs and rank agreement are checked before numerical collectives. The native MLA keeps the frozen layout and exact Ordered O conversion sequence. The initial real prefill found that the current replay helper returns three values. The TP=2 adapter now directly invokes the accepted Ordered O primitive and exposes its actual FP32 output; it does not invoke the replay helper's legacy diagnostic SUM.

The model's optional traces record worker, local rank, stream, operator, dtype and shapes. The first numerical run additionally records raw tensors, filtered to real scheduled forward calls. Supplemental TP=1 diagnostic boundaries are separate from frozen references. The production entrypoint and legacy model sources were left as found, including pre-existing worktree edits; the frozen TP=1 regression mounted the preserved HEAD source over those files only inside the test container.

### Reproduce

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

<a id="phase14-1-numerical-acceptance"></a>

## Phase 14.1 — Numerical acceptance

The fresh frozen TP=1 regression executed real Vendor `LLM.generate()`. All five requests matched frozen greedy tokens and the frozen CSV comparator counted zero tolerance failures. The original startup script emitted its known process-group destructor warning. The host reported no MLU processes before the two-card launch.

The corrected actual TP=2 length-one run produced greedy tokens `[254, 4334]`. All 1,628 recorded per-rank comparisons passed the frozen predicate `abs(error) <= 1e-5 + 1e-5*abs(reference)`, with zero missing boundaries and zero disagreements in replicated output hashes. Embedding, K/V, O BF16, layer outputs and logits use `native-dump-yarn-scale`. The pre-residual inputs and accepted O output supply the independent BF16 residual reference. Intermediate Q/KV-A/RMSNorm/KV-B, RoPE, QK, Softmax, AV and ordered FP32 O are checked against supplemental unsharded native captures produced from the same accepted frozen inputs. Each supplemental O capture was independently anchored to the accepted BF16 O bytes.

An independent source review identified a missing-metadata early-exit hazard. The repaired preflight exchanges context and error status before accepting profile-only calls, then validates local metadata and agrees on the schedule digest.

[Accuracy CSV](../artifacts/logs/phase14_1/tp2_accuracy.csv) · [Boundary summary](../artifacts/logs/phase14_1/boundary_summary.json) · [Corrected real-engine worker log](../artifacts/logs/phase14_1/tp2_retry/worker.log)

The expanded frozen engine run completed ten consecutive single requests and two requests in one batch. All 96 recorded rank/logit comparisons passed, and every reported greedy sequence matched its frozen starting length. The expanded workload covers prompts of lengths 1, 7, 17 and 65 and three actual decode steps each. Explicit release ran on both native models, and the Vendor executor reported graceful shutdown.

A separate lifecycle run with a real chunked scheduler reached absolute decode position 16 and wrote slot 32 on the second page. Scheduler steps 18, 19 and 20 contained both prefill and decode, and a block released from the prior request was reassigned. The mixed 7/65 prompts produced the frozen four-token sequences. Only the first four tokens of the additional 17-token lifecycle prompt have frozen numerical targets.

[Expanded engine log](../artifacts/logs/phase14_1/tp2_expanded/worker.log) · [Cross-page and mixed scheduler log](../artifacts/logs/phase14_1/cross-page/worker.log)
