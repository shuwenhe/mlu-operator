# Phase 13.3.1 — Future MLA acceptance plan

## Numerical gates

Use frozen Phase 12.5.4 prompts and single-card reference for lengths 1, 7, 17, 65. Replay exact token sequences in prefill and incremental decode; do not regenerate references in place. Use BF16 for full native acceptance. FP32 remains a projection/collective check and an independent mathematical diagnostic where supported. A full FP32 native attention path requires an explicit supported contract.

| Stage | Evidence to compare |
|---|---|
| Q | Local FP32 projection vs the corresponding contiguous reference head slice, then assembled head order |
| KV latent | Replicated KV-A output, raw rotary slice and cast/norm of latent on every rank |
| KV expansion | Each local K-noPE and V slice against separate head-aligned reference boundaries |
| RoPE | Per-head Q and shared K for positions 0,7,16,17,64,65 and incremental tails; compare BF16 products/output and actual score scale |
| Scores | Raw/scaled/masked values and exact causal mask, including page-boundary decode and mixed prefill/decode |
| Softmax | FP32 max/denominator/probabilities and BF16 output; verify probability normalization and zero masked entries |
| Attention output | FP32 AV and BF16 post-cast local slices, merge ordering |
| Output projection | FP32 local partials vs independent shard GEMMs, replicated FP32 reduction, BF16 cast and residual |
| Complete MLA layer | Frozen 27-layer boundaries and cache writes/reads; capture one isolated layer first, then repeat on all layers and on mixed requests |

Record max absolute/relative error, mean absolute error, ULP distance, bitwise and tolerance mismatch counts, first mismatch index, rank/device/group, dtype, fixture hash and per-rank output consistency. For masks compare infinity placement separately from finite metrics. For the frozen dumped boundaries use the existing `abs(error) <= 1e-5 + 1e-5 * abs(reference)` predicate and check finiteness. Capture missing intermediate boundaries, such as raw QK or softmax statistics, with a separate instrumented TP=1 execution in a new evidence directory. Use exact checks for checkpoint/cache indexing, metadata, end-to-end replicated rank equality, and identical deterministic RoPE fixtures. For accumulated floating point use frozen per-boundary tolerances where defined; establish and record any new tolerance before distributed runs from unsharded independent references and accumulation order. Never widen a frozen tolerance. Add malformed shape/slot and mismatched worker schedule cases. Verify all rank exits, no fallback and cache block reuse after release.

## Performance gates

TP=1/2/4/8, lengths 1/7/17/65, prefill and repeated incremental decode, warm up and collect at least 100 timed repetitions for supported cases on the frozen image. Use slowest rank per step and report p50/p95, throughput, speedup and efficiency. Attribute Q, KV-A plus norm, KV-B, RoPE (including padded staging), QK/mask, Softmax, AV, cache scatter/gather, O local GEMM, and CNCL independently. Also time full MLA including layout and barriers, excluding model load unless measured separately. Record per-layer page bytes, local and aggregate cache, scratch and peak tracked allocator plus native bytes. Separate device-kernel and synchronized host timing; record overlap and synchronization scope. Phase 13.2 measured smaller workloads and most FP32 cases slower at TP=8; a speedup is a result to measure, never a gate substitute.

## Execution order

Head-aware layout and reference boundary captures precede local attention validation. Validate two local heads and emitted ordering, then cache ownership/reuse, then distributed O and MLA end-to-end. Preserve the Phase 12.5.4 source and fixtures and the Phase 13.1 and 13.2 hardware evidence throughout.

## Audit gates

| Gate | Result | Evidence |
|---|---|---|
| Evidence preservation | PASS | Initial HEAD/status and worktree patches, historical and current fixture/source hashes, four checkpoint shard hashes |
| Actual MLA architecture and forward trace | PASS | Local config, eager and native code, pinned Vendor source cross-check |
| Checkpoint mapping and reconstruction | PASS | 135 tensors, 1,080 rank records, byte-identical reconstruction |
| TP=8 head ownership and Linear API mapping | PASS | Two contiguous heads per rank, audited Column and Row boundaries |
| KV cache architecture and scheduler design | PASS | Frozen slot-major storage and block validation, proposed rank-local storage and shared logical ownership |
| RoPE compatibility audit | PASS | Source pairing, frozen device regression, BF16 table and FP32 ABI parity |
| Numerical and performance acceptance design | PASS | Fixtures, boundaries, stability checks and measurement scope specified |
| Independent review | PASS | Reviewed shard order, QK/AV entry points, RoPE pairing, cache bytes and residual rounding |
| Native TP MLA hardware execution | NOT RUN | Hardware gate belongs to a later implementation phase |
