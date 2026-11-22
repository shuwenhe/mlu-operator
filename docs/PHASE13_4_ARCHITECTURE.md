# Phase 13.4 — Distributed MLA

The isolated runtime will use contiguous rank-local Q and KV-B output weights, replicated KV-A and RMSNorm, and two physical KV heads per rank. The frozen attention library already accepts a runtime head count at its QK, Softmax and AV launcher boundaries. The frozen RoPE launcher uses 17 logical stripes. The adapter stages two active query stripes and shared K in that allocation. The two-head KV scatter/gather is built from the frozen paging source with its head count specialized to two.

Output-axis O is the primary projection. It gathers BF16 attention channels and then gathers FP32 output stripes, followed by one BF16 cast and residual. Experimental ordered O uses the Phase 13.3.3 implementation and must be explicitly selected. It has separate evidence. Worker metadata uses one digest collective before processing each schedule. No attention-head collective is needed in QK, Softmax or AV.

Validate local layout and existing fixture parity first, followed by single request, incremental decode, page boundaries, mixed work, page release/reuse, complete layer, all layers, and finally full model gates. Historical references remain mounted read-only.
