# Phase 13.5.1 — Production Integration Architecture Audit

## Scope
- Repo audit root: `/app/shuwen/mlu-operator`
- Vendor runtime source roots (frozen image):
  - `/torch/venv3/pytorch_infer/lib/python3.12/site-packages/vllm`
  - `/workspace/vllm-v0.25.1+mlu0.16.0.pt212/vllm_mlu`
- Source inventory evidence: `artifacts/logs/phase13_5/phase13_5_1_source_inventory.txt`
- Source hash evidence: `artifacts/logs/phase13_5/phase13_5_1_source_hashes.txt`

## 1) Actual model registration and loading path

### Local integration entry (already in use)
- Native registration call site: `ModelRegistry.register_model(...)` in [run.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_5/run.py).
- Runtime architecture override is passed via `hf_overrides={"architectures":[ARCHITECTURE]}` in [run.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_5/run.py) and [run_acceptance.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_6/run_acceptance.py).

### Vendor vLLM resolution contract
- LLM user entry + generation API in `vllm/entrypoints/llm.py` (`class LLM`, `def generate`).
- Model registry contract in `vllm/model_executor/models/registry.py`:
  - `_ModelRegistry.register_model(...)`
  - `_ModelRegistry.inspect_model_cls(...)`
  - `_ModelRegistry.resolve_model_cls(...)`
- Worker-side model loading path in `vllm_mlu/v1/worker/gpu_model_runner.py`:
  - `MLUModelRunner.load_model(...)`
  - `self.model = model_loader.load_model(...)`

结论：当前“真实生产路径”是 `LLM(...) -> EngineCore/worker -> GPUModelRunner.load_model -> ModelRegistry resolve/inspect -> instantiated model class`。Phase 13.5 集成必须进入这个链路，不能只停留在 probe/replay。

## 2) Real execution path: load -> prefill/decode -> token output

### End-to-end high-level path
1. `LLM(...)` 初始化 engine（`llm.py`）。
2. `LLM.generate(...)` 进入 engine 调度。
3. MLU worker 由 `vllm_mlu/v1/worker/gpu_worker.py` 选择并构造 `MLUModelRunner`。
4. `MLUModelRunner.load_model(...)` 加载模型。
5. 调度阶段提供 request/block/slot 元数据，runner 构造 attention metadata 后执行前向。
6. 输出 logits 并完成 token 选择（对应 repo 中 [model.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_5/model.py) 的 `compute_logits` 合约）。

### Prefill/decode metadata flow (actual contracts)
- Repo native model侧读取：
  - `get_forward_context().attn_metadata`
  - `vllm_mlu...get_common_metadata()`
  - 依赖 `block_table_tensor`、`slot_mapping`
  - 位置： [model.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_5/model.py)
- Vendor side metadata construction/传递：
  - `vllm_mlu/v1/attention/backends/utils.py` (`get_common_metadata`, `MLUCommonAttentionMetadata`)
  - `vllm_mlu/v1/worker/gpu_model_runner.py` (`_get_slot_mappings`, block table/slot mapping wiring)

结论：Phase 13.5 必须对齐 `MLUCommonAttentionMetadata + ForwardContext` 的现有 contract，不能自建“旁路 metadata 接口”。

## 3) TP process group, communicator, and CNCL ordering

- TP init contract（repo probes已验证）：
  - `init_distributed_environment(..., backend="cncl")`
  - `initialize_model_parallel(..., backend="cncl")`
  - 见 [cncl_probe.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase13_1/cncl_probe.py), [probe.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase13_4/probe.py)
- Communicator实现：
  - `MLUCommunicator` in `vllm_mlu/distributed/device_communicators/mlu_communicator.py`
  - 暴露 `all_reduce / all_gather / reduce_scatter`，底层 `torch.distributed` + CNCL group

结论：Phase 13.5 的 ordered O 必须嵌入现有 TP collective 顺序，不得破坏 scheduler/runner 已有 collective 序。

## 4) KV cache lifecycle (allocation/write/read/free/reuse)

- Scheduler 持有 `KVCacheManager`（`vllm/v1/core/sched/scheduler.py` -> `KVCacheManager`）。
- `KVCacheManager.free(...)` 是 release/reclamation contract（`vllm/v1/core/kv_cache_manager.py`）。
- Worker runner 持有 slot mapping / block table，执行 prefill/decode 期间 cache addressing：
  - `_get_slot_mappings(...)`
  - block table commit / mapping usage
  - `gpu_model_runner.py` 中多处 `slot_mapping` / `block_table` 路径（inventory 已记录行号）。
- Repo native model cache read/write contract：
  - `cache.write(k, v, metadata.slot_mapping[...])`
  - `cache.read(metadata.block_table_tensor[...], span.length)`
  - 见 [model.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_5/model.py)

结论：分布式 KV 生命周期集成边界已明确：scheduler 分配与回收、runner metadata 编排、model 消费 metadata。Phase 13.5 不应新建独立 cache manager。

## 5) Ordered O (Phase 13.3.3) integration boundary

现有验证实现位于 replay/probe 路径：
- [mla.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase13_4/mla.py): ordered FP32 accumulation + BF16 cast boundary evidence
- [probe.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase13_4/probe.py), [full_model.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase13_4/full_model.py): gate replay

生产集成边界（必须保持 Vendor contract）：
1. 仅替换 attention O 聚合路径内部数值策略（local partial -> CNCL sum -> BF16 cast -> residual）。
2. 输入输出张量形状、dtype、slot/block metadata 接口保持与 worker/model 现有路径一致。
3. 不修改 TP=1 生产路径行为（只能在 TP>1 条件下启用）。

## 6) TP=1 regression protection boundary

- TP=1 回归入口已存在于历史 acceptance harness（例如 [run_acceptance.py](/app/shuwen/mlu-operator/integration/vllm_native_mlu/phase12_6/run_acceptance.py) 与 13.x probes）。
- 强制要求：
  - TP=1 代码路径逻辑不变（功能与数值基线冻结）。
  - TP=2/4/8 新集成必须被明确条件门控。
  - 任何 TP=1 偏移应直接判定 Phase13.5 BLOCKED。

## 7) Blockers / risks identified

1. Vendor source ABI/签名风险  
   - `vllm_mlu` worker/runner 接口复杂且版本绑定，若直接侵入内部私有结构风险高。  
   - 缓解：仅在现有 metadata + runner 调用边界做最小替换。

2. Collective ordering risk  
   - ordered O 改动若改变 reduce/allgather 的相对顺序，存在死锁/数值漂移风险。  
   - 缓解：沿用现有 communicator 调用顺序，新增验证覆盖 invalid metadata 注入场景。

3. KV metadata consistency risk  
   - prefill/decode mixed batching 依赖 slot mapping 与 block table 一致性。  
   - 缓解：将 metadata 一致性和回收复用放在首个硬门槛之前验证。

4. Runtime-path mismatch risk  
   - 仅通过 probe/replay 通过不足以代表生产可用。  
   - 缓解：Phase13.5 必须以真实 `LLM.generate` engine 路径为主验收对象。

## 8) Static checks and independent source review

- Static check:
  - `py_compile` 写入 `__pycache__` 受权限限制（已记录并切换方案）。
  - 采用 AST 非写入语法检查：`AST_PARSE_OK 6`（审计模块语法通过）。
- Independent source review:
  - 采用“仓内调用点 + Vendor源码行号”双路径交叉审查；
  - 核对项：注册路径、worker load path、slot/block metadata、communicator collectives、KV free/reclaim；
  - 审查结论：当前边界可实现生产集成，前提是不破坏 TP=1 与 scheduler/metadata contracts。

## Audit status
- Phase 13.5.1 Production architecture audit: **COMPLETE**
- Production code modification: **NOT STARTED (as required)**
