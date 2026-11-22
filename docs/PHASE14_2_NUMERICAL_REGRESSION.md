# Phase 14.2 — Numerical regression

The lifecycle executor leaves the original TP=1 route and the Phase 14.1 model, Native MLA, Ordered O and KV cache unchanged. The TP=2 normal-shutdown acceptance ran the frozen length-one prompt through the production `LLM.generate()` path. Both prefill and incremental decode produced the frozen greedy tokens and passed four rank/logit comparisons.

The recovery driver uses the same frozen fixture and tolerance on a newly initialized engine after each fault. The separate TP=1 launch uses the pinned image, original checkpoint and original implementation snapshots. The final gate record reports the new run counts and distinguishes prior Phase 14.1 results from Phase 14.2 evidence.
