# Native MLU production roadmap

## Three highest-impact blockers

1. **The real worker reaches a single-card model contract.** An earlier spawned-worker registration failure is documented. The latest archived TP=2 launch reaches the native constructor on both workers and stops there. The worktree's relaxed guard does not bind TP weights or a local-head forward. Close this boundary through the official registry and loader.
2. **Worker-owned distributed KV cache is absent.** The standalone eight-rank harness has correct local KV behavior. The real model's cache spec, reshape and paging still target 16 heads. TP=2 needs eight physical heads per rank and engine-owned blocks, slot IDs and reclamation.
3. **The real multi-rank execution gates have no passing evidence.** There is no accepted TP=2 `LLM.generate()` proof for logits or token IDs, and no TP=4/8 engine scale or reliability baseline. Earlier MoE replay is credible correctness evidence, while host-driven expert dispatch remains a performance concern to quantify after P0.

[Architecture and operator inventory](MLU_OPERATOR_ARCHITECTURE_AUDIT.md)

## First milestone — real TP=2 generation under a separate architecture

1. Preserve the captured worktree diff and pin the frozen TP=1 runtime and prompt/reference hashes. Record the exact image and accepted `native-dump-yarn-scale` fixture lineage before running the engine.
2. Package a small `vllm.general_plugins` entry point in the same frozen environment. It registers a lazy import string for the experimental TP model in each spawned interpreter. Prove positive registration before ModelRegistry resolves the class in both workers. Keep one negative path with the plugin disabled to distinguish registration from construction. Keep the current working-tree guard draft separate from this implementation.
3. Use the normal model loader, MLUModelRunner and ForwardContext. Bind TP=2 weight slicing and select output-axis O explicitly. Specialize Q/KV/AV to eight local heads, verify block/page byte sizes, bind zero-copy views of worker allocations and retain the scheduler's identifiers. Reject unsupported metadata before numerical collectives. Reuse replicated, frozen non-MLA math initially.
4. First run frozen TP=1 `LLM.generate()` and confirm recorded logits and greedy tokens at the original tolerances. Then run real TP=2 with length 1 and one decode, expanding to lengths 7, 17 and 65 only after each smaller gate passes. Verify per-rank model identity, worker cache specs, actual native calls, tokens, logits, page ownership and shutdown. Retain the first failing trace and stop the sweep on a frozen numerical failure. The model length stays 128 including decode.
5. Once TP=2 passes, repeat the engine gates at TP=4 and TP=8. Exercise mixed scheduling, release/reuse, restart and coherent failure. Add performance and hour-long reliability only after the stable production execution path is known. Perform independent code and numerical review before closing the production gate.

## Remaining priorities

- **P1 MoE.** Audit routed-token counts and expert skew on real engine fixtures. Keep frozen router, softmax, top-k and mathematical combine as independent references. Prototype device dispatch/grouped GEMM, then evaluate EP rank ownership and Vendor all-to-all only when the baseline identifies a bottleneck.
- **P2 operators.** Rank GEMM, RMSNorm, RoPE, QK, Softmax, AV, O, paging and CNCL by measured contribution to end-to-end latency. Run parity before adopting each optimization.
- **P3 memory.** Measure checkpoint H2D, packed weights and allocator lifetime, KV pages, native scratch, fragmentation and reuse. Preserve the scheduler's metadata and cache contract.
- **P4 reliability.** Record repeated lifecycle, mixed requests and resource cleanup, followed by at least one hour of continuously accepted multi-rank inference. Inject worker failure only in an isolated test with a bounded timeout.

## Current gates

| Gate | Result |
|---|---|
| Native operator correctness and TP=8 MLA | PASS for archived isolated scope |
| TP=1 real vLLM regression | Historical PASS; current worktree regression NOT RUN |
| TP=2 real vLLM inference | BLOCKED at the latest archived constructor; current draft NOT RUN |
| TP=4 and TP=8 real vLLM inference | NOT RUN |
| MoE execution and KV lifecycle | Archived standalone/TP=1 evidence; production multi-TP NOT RUN |
| Production multi-TP performance and one-hour reliability | NOT RUN |
| Phase 14 independent review | NOT RUN |

[Performance roadmap](MLU_OPERATOR_PERFORMANCE_ROADMAP.md) · [Evidence verification](../artifacts/logs/phase14/verification.txt)
