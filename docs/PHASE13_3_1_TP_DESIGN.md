# Phase 13.3.1 — TP=8 MLA design

## Selected path

Replicate BF16 hidden states, pre-norm and KV-A/gamma. Column-shard Q and KV-B by contiguous complete heads. Each rank rotates two Q stripes plus the shared KV-A K stripe, materializes its two K/V heads, scans its full local history, computes scores and softmax over sequence positions, and emits [M,2,128] in token-major head order for the output GEMM. Heads are independent. Neither Q/K/V nor attention probabilities need a cross-rank reduction in the normal path. Debug gathers use global head order.

Row-parallel O consumes [M,256] with W [2048,256] and `input_is_parallel=True`. It returns the replicated FP32 sum [M,2048]. Apply the frozen BF16 cast then the existing BF16 residual outside the Linear; passing residual into Phase 13.2 Row would change rounding. O has no bias. Keep one data-plane collective per MLA layer, an FP32 CNCL AllReduce over 8,192 logical bytes per scheduled token per rank. All workers must enter the same ordered calls.

## KV latent alternatives

| Option | Memory and work | Collective, numerical and decode implications |
|---|---|---|
| A. Replicate KV-A and full latent | BF16 W ~2.25 MiB/rank/layer before packing; BF16 latent+rotary 1,152 bytes/token temporary per rank; duplicate projection and norm | No latent collective. Same full-width norm/reduction order and handoff as frozen path. Preferred first implementation. Measure redundant GEMM and pack memory. |
| B. Keep latent distributed | Can shard KV-A rows (64 latent + 8 rotary outputs per rank if evenly split), but head-local KV-B consumes all 512 latent features | Requires global RMS sum and access to the complete normalized latent or a 2-D/partial-product KV-B decomposition. Shared rotary must be assembled with correct pairing. Extra communication and rounding boundaries on decode. |
| C. Shard KV-A then gather | W shrinks to ~0.28125 MiB/rank before packing; gather returns full 576 outputs on each rank | At least one additional gather per layer, 1,152 logical bytes/token after BF16 narrowing or 2,304 at FP32; reproduce original narrowing before norm and rotary. Decode startup overhead can dominate. Benchmark only after A parity. |

The pinned Vendor projections support the A topology. This is source corroboration. Hardware gates apply to the later native two-head implementation.

## API and ABI plan

Use Phase 13.1 `get_tp_group()`, Vendor CNCL device group and `MLUCommunicator`. Reuse Phase 13.2 `ColumnParallelLinear(weight, sections, group, library, bias=None)` and `RowParallelLinear(weight, input_width, group, library, bias=None)`, shard spans, comparisons and runner. For KV-A use the accepted rank-local `PackedProjection` or its Phase 13.2 `LocalGemm` wrapper. Keep communications on the device TP group.

Frozen prepare/append/merge, RoPE, cache views and attention wrapper contain head-count assumptions. A new head-aware boundary must preserve FP32-to-BF16 layout casts, use two head lanes and verify ABI limits. The frozen RoPE kernel loops over exactly 17 stripes. First-stage reuse can stage two local Q stripes plus shared K in a correctly sized 17-stripe buffer, initialize the 14 unused stripes and discard their outputs. Never call that kernel on three allocated stripes. Benchmark padded RoPE separately; a head-aware launch in a later implementation needs independent parity and cannot change frozen source. Keep production TP=1 guard until end-to-end acceptance.
