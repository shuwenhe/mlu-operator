# Phase 14.1 — Numerical acceptance

The fresh frozen TP=1 regression executed real Vendor `LLM.generate()`. All five requests matched frozen greedy tokens and the frozen CSV comparator counted zero tolerance failures. The original startup script emitted its known process-group destructor warning. The host reported no MLU processes before the two-card launch.

The corrected actual TP=2 length-one run produced greedy tokens `[254, 4334]`. All 1,628 recorded per-rank comparisons passed the frozen predicate `abs(error) <= 1e-5 + 1e-5*abs(reference)`, with zero missing boundaries and zero disagreements in replicated output hashes. Embedding, K/V, O BF16, layer outputs and logits use `native-dump-yarn-scale`. The pre-residual inputs and accepted O output supply the independent BF16 residual reference. Intermediate Q/KV-A/RMSNorm/KV-B, RoPE, QK, Softmax, AV and ordered FP32 O are checked against supplemental unsharded native captures produced from the same accepted frozen inputs. Each supplemental O capture was independently anchored to the accepted BF16 O bytes.

An independent source review identified a missing-metadata early-exit hazard. The repaired preflight exchanges context and error status before accepting profile-only calls, then validates local metadata and agrees on the schedule digest.

[Accuracy CSV](../artifacts/logs/phase14_1/tp2_accuracy.csv) · [Boundary summary](../artifacts/logs/phase14_1/boundary_summary.json) · [Corrected real-engine worker log](../artifacts/logs/phase14_1/tp2_retry/worker.log)

The expanded frozen engine run completed ten consecutive single requests and two requests in one batch. All 96 recorded rank/logit comparisons passed, and every reported greedy sequence matched its frozen starting length. The expanded workload covers prompts of lengths 1, 7, 17 and 65 and three actual decode steps each. Explicit release ran on both native models, and the Vendor executor reported graceful shutdown.

A separate lifecycle run with a real chunked scheduler reached absolute decode position 16 and wrote slot 32 on the second page. Scheduler steps 18, 19 and 20 contained both prefill and decode, and a block released from the prior request was reassigned. The mixed 7/65 prompts produced the frozen four-token sequences. Only the first four tokens of the additional 17-token lifecycle prompt have frozen numerical targets.

[Expanded engine log](../artifacts/logs/phase14_1/tp2_expanded/worker.log) · [Cross-page and mixed scheduler log](../artifacts/logs/phase14_1/cross-page/worker.log)
