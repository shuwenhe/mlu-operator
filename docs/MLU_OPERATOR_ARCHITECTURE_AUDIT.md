# Native MLU production architecture audit

## Current execution graph

The frozen Vendor engine starts from `LLM.generate()`, then passes scheduled requests through EngineCore, the MLU worker and model runner, model loading via the process-local ModelRegistry, model forward, logits and sampling. The worker receives the engine's block table and slot mapping through ForwardContext. The scheduler KV manager owns block allocation and release. TP workers use the Vendor CNCL device group and MLUCommunicator. Gloo CPU control groups are separate from device tensor collectives. [Vendor source inventory](../artifacts/logs/phase13_5/phase13_5_1_source_inventory.txt)

Phase 13.4 uses a standalone replay driver with real model weights, generated tokens and 27 layers on eight MLUs. It has a local Pages allocator and directly invokes the native MLA and replicated decoder helpers. It provides numerical evidence for those operations. In the real engine the corresponding pages and worker metadata must come from vLLM. [Corrected replay evidence](PHASE13_4_FINAL_EVIDENCE_CLOSURE.md)

## Operator inventory and gaps

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

## Worker registration and worktree findings

Phase 13.5.3 failed registry resolution in spawned worker processes. ModelRegistry is process-local. The archived Vendor source audit identifies the official `vllm.general_plugins` entry-point loader. Phase 13.5.4 subsequently reached the native constructor on **both** TP=2 workers, then rejected the frozen TP=1 contract. This is the latest observed stop. The current worktree has pre-existing edits that widen that constructor check and add import-time registration in `run.py`. The diff adds no per-rank native MLA/cache path, and no passing real-engine TP=2 evidence accompanies it. Archived TP=1 acceptance predates the current draft. [Registry audit](../artifacts/logs/phase13_5_4/worker_registration_audit.md) · [Latest TP=2 traceback](../artifacts/logs/phase13_5_4/tp2.log) · [Worktree snapshot](../artifacts/logs/phase14/unstaged.patch)

Use deterministic worker-side registration through the official installed general-plugin extension when this path is implemented. Keep the legacy TP=1 architecture entry clustered with its frozen initializer. The experimental TP=2 model should have a separate architecture name and request the engine's actual group and scheduler metadata. EP uses a separate expert ownership and token-dispatch plan and is outside the first engine milestone.
