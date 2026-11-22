# Phase 13.5.1 — Integration Plan (No Production Code Changes Yet)

## Objective
在不破坏 Vendor runtime 合约与 TP=1 生产路径前提下，将已独立验证的 Native MLA（含 Phase 13.3.3 ordered O）接入实际 vLLM MLU 生产执行链路。

## Frozen constraints (hard gates)
1. 不修改 Phase 12.5.4 frozen references / fixtures / historical logs。
2. 不升级 Vendor vLLM / Neuware / 镜像。
3. 不引入 silent fallback。
4. 不改变 TP=1 生产行为。
5. 数值门限固定：`abs(error) <= 1e-5 + 1e-5 * abs(reference)`。

## Integration boundaries (from audit)
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

## Execution plan (incremental)

### Step 0 — Safety preconditions
- 保持当前证据链不覆盖写。
- 为 Phase13.5 单独输出目录：`artifacts/logs/phase13_5/`。

### Step 1 — TP=1 regression guard first
- 建立 TP=1 快速回归门（prefill + decode + logits + tokens）。
- 任何 TP=1 偏移立即停止后续步骤并标记 BLOCKED。

### Step 2 — TP=8 production wiring (minimal change)
- 将 Native MLA 接入 worker/model-runner 的真实执行路径（非 probe）。
- 先完成 prefill 单请求，再开放 decode，再开放 mixed batching。

### Step 3 — Distributed KV lifecycle verification
- 验证 allocation/write/read/free/reuse。
- 注入 invalid metadata，验证一致错误退出且无 collective deadlock。

### Step 4 — Numerical acceptance (frozen references)
- 长度：1/7/17/65；prefill + ≥3 decode。
- 边界：Q/K/V、RoPE、QK、Softmax、AV、O partial、ordered FP32、BF16 O、residual、logits、token IDs。
- 一旦出现“首个无法解释的 mismatch”，立即停机保留证据。

### Step 5 — End-to-end engine acceptance
- 使用真实 `LLM.generate` 路径执行 TP=1/2/4/8。
- 覆盖单请求、并发、多长度混合、重复生成、engine restart。

### Step 6 — Performance and reliability
- 性能：可支持配置下每组 ≥100 次，slowest-rank p50/p95。
- 可靠性：连续 ≥1h，验证无失败/无 NaNInf/无死锁/无 rank divergence/无内存异常增长。

### Step 7 — Independent review before phase close
- 对边界、collective顺序、KV生命周期、BF16 rounding、TP=1保护做独立复审。

## Gate matrix
- G1 Registration/loading path intact: PASS/BLOCKED
- G2 TP=1 regression: PASS/FAIL
- G3 TP=8 production path executes prefill+decode: PASS/FAIL
- G4 Distributed KV lifecycle correctness: PASS/FAIL/BLOCKED
- G5 Numerical parity vs frozen references: PASS/FAIL
- G6 End-to-end multi-TP engine acceptance: PASS/FAIL/BLOCKED
- G7 Performance evidence complete: PASS/BLOCKED
- G8 1h reliability evidence complete: PASS/FAIL/BLOCKED

## Deliverable mapping
- Architecture audit: `docs/PHASE13_5_1_PRODUCTION_ARCHITECTURE_AUDIT.md`
- Plan: `docs/PHASE13_5_1_INTEGRATION_PLAN.md`
- Source inventory: `artifacts/logs/phase13_5/phase13_5_1_source_inventory.txt`
- Hash manifest: `artifacts/logs/phase13_5/phase13_5_1_source_hashes.txt`

## Current status
- 13.5.1 审计与计划：**COMPLETE**
- 生产代码集成：**NOT STARTED**
