# Tensor-parallel MLA, O projection and KV cache

- [Phase 13.3.1 — MLA KV cache design](#phase13-3-1-kv-cache-design)
- [Phase 13.3.1 — DeepSeek-V2-Lite MLA architecture audit](#phase13-3-1-mla-architecture-audit)
- [Phase 13.3.1 — MLA checkpoint weight mapping](#phase13-3-1-mla-weight-mapping)
- [Phase 13.3.1 — TP=8 MLA design](#phase13-3-1-tp-design)
- [Phase 13.3.2 — Native MLA projections](#phase13-3-2-implementation)
- [Phase 13.3.2 — Projection performance](#phase13-3-2-performance-report)
- [Phase 13.3.3 — Ordered O projection](#phase13-3-3-implementation)
- [Phase 13.3.3 — Numerical acceptance on MLU590-M9](#phase13-3-3-numerical-acceptance)
- [Phase 13.4 Final Numerical Evidence Closure](#phase13-4-final-evidence-closure)

<a id="phase13-3-1-kv-cache-design"></a>

## Phase 13.3.1 — MLA KV cache design

### Frozen worker contract

Phase 12.5.4 registers one native MLA cache entry per layer (27) in static forward context. Worker allocation uses `MLUFullAttentionSpec` and page reshape validates full-attention byte size. The observed block size is 16. One raw byte allocation per layer becomes two zero-copy BF16 views over S = local block capacity × 16 slots. K is [S,16,192], V is [S,16,128]. The V base is after S×16×192 BF16 words. K includes materialized non-rotary K and rotated shared K copied per head; V is materialized. The 512 latent and single 64 rotary vector are transient before expansion and rotation. The separate Vendor compressed path persists latent and rotary information; it does not define the native cache ABI.

The metadata builder uses the Vendor common metadata and rejects shared prefix pages. `validate_batch` checks request boundaries, lengths, absolute positions, exclusive/in-range blocks and int32 slot IDs. For absolute p, write to `block_table[request,p//16]*16+p%16`. Prefill writes scheduled tokens; incremental decode writes the new tail, then gathers full committed history by logical block order into head-major scratch. Padding slots after L are not consumed. The scheduler controls block reuse and reclamation. A request release recycles identifiers; it does not imply zeroing old bytes. [Frozen cache and allocator evidence](PHASE12_PRODUCTION_ACCEPTANCE.md#phase12-5-4-root-cause-report) · [Slot validation](../integration/vllm_native_mlu/phase12_3/scheduled_attention.py)

### Proposed TP ownership

Share logical request IDs, absolute positions, block tables, slot mapping and allocation/free events. Each rank independently owns physical storage for its two heads per layer. Use K [S,2,192] and V [S,2,128], 1,280 bytes/token/rank/layer or 20,480 bytes per block per layer. Frozen TP=1 uses 10,240 bytes/token/layer or 163,840 bytes per block. At 27 layers the eight-rank design uses 552,960 bytes per block per rank and 4,423,680 bytes across ranks, excluding allocator padding and scratch. Replicating a full materialized cache would multiply memory by eight. Replicated compressed latent+rotary would cost 18,432 bytes per block per rank, but requires a distinct decode kernel and revalidation of the materialization/rounding contract.

At worker init, require a common usable block count limited by the smallest rank capacity and bind every layer to the agreed numbering. Distribute one scheduler schedule per step, including request ordering, valid spans, table contents and slot IDs. Before numeric execution verify equal metadata digests across workers and local bounds; pointer equality across devices is irrelevant. Allow only exclusive pages initially. Free and reuse logical pages after all rank consumers complete; preserve the stream/completion dependency for K/V writes, reads and next-step reuse. Inject divergent bounds/order in an isolated negative test, fail coherently before subsequent collectives, and impose a timeout. Actual multi-rank worker metadata agreement and reclamation are future hardware gates.

<a id="phase13-3-1-mla-architecture-audit"></a>

## Phase 13.3.1 — DeepSeek-V2-Lite MLA architecture audit

### Verified architecture

The local checkpoint config and safetensors headers establish hidden width 2,048, 27 layers, 16 attention heads and BF16 weights. Each Q/K head is 192 channels (128 non-rotary + 64 rotary), each V head is 128 channels. The query LoRA rank is null. The KV latent rank is 512. Attention bias is disabled and the configured dropout is zero. The model config advertises 163,840 positions. The audited NeurX model accepts configured max length at most 128, the metadata/attention interfaces cap history at 131 and queries at 128, and the frozen RoPE tables contain 131 rows. Those are distinct limits.

Checkpoint `DeepseekV2Attention.__init__/forward` sets the per-head reshape and applies attention. `DeepseekV2DecoderLayer.forward` applies pre-attention norm and the residual outside attention. The installed Vendor class `MLUDeepseekV2MLAAttention` independently selects local heads, column Q/KV-B, replicated KV-A, and row output. Its compressed-cache kernels are a separate implementation surface. [Vendor source locations](../artifacts/logs/phase13_3_1/vendor_audit.txt)

### Execution order and native boundaries

Shapes below omit the request batch axis. M is the scheduled query count, L its committed history length. H=16 in the frozen path, H=2 in the proposed TP=8 path. Projection GEMMs use BF16 inputs/weights and return FP32 in NeurX; the checkpoint eager Linear returns activation dtype.

| Operation | Frozen shape and dtype | Current implementation and native entry | Phase 13.2 and planned placement |
|---|---|---|---|
| Input / pre-norm | BF16 [M,2048] to same; gamma [2048] | Decoder input norm, `NativeRMSNorm` | Replicated; no Linear |
| Query projection | [M,2048] × W [3072,2048] to FP32 [M,3072] | `_project`, `PackedProjection` | Column, BF16 GEMM, local [M,384] |
| Query LoRA A/B and Query RMSNorm | Absent | Eager branch skipped because query rank is null; no weights | Absent on every rank |
| KV latent projection | W [576,2048], FP32 [M,576] to 512 latent + 64 shared rotary K | `PackedProjection`; `phase5_cast_kv` narrows latent | Replicate GEMM and raw rotary term |
| KV latent norm | BF16 [M,512] to same; gamma [512] | `NativeRMSNorm`, epsilon 1e-6 | Replicated across full width |
| KV expansion | W [4096,512], FP32 [M,4096] = [M,16,256] | `PackedProjection` | Column, local [M,512]; each head owns K-nope then V |
| Q/K separation and RoPE | Q [M,16,192] splits 128+64; shared raw K [M,64]; BF16 rotary staging [17,M,64] | `phase10_prepare_bf16`, `worker_rope_bf16`, `phase10_append_bf16` | Local Q stripes and one shared K stripe; existing 17-stripe ABI needs safe staging |
| Cache scatter/gather | BF16 K [16,M,192], V [16,M,128] to slot-major storage and back as [16,L,D] | `PagedKV.write/read` | Local two-head physical cache; global logical metadata |
| Scores and mask | BF16 Q/K; FP32 raw QK [16,M,L], BF16 scaled, FP32 causal-masked | `NativeMaterializedAttention`, `phase9_launch_qk_bf16` and frozen scale | Local heads, full L per head; no Linear |
| Softmax | FP32 mask to FP32 probabilities and BF16 cast [16,M,L] | `phase9_softmax_bf16`, normalize across L | Local, no rank reduction |
| Value aggregation | BF16 P/V to FP32 accumulator then BF16 [16,M,128] | `phase9_av_bf16`; `phase10_merge` converts to token-major [M,2048] | Local attention, merge to [M,256] |
| Output projection | W [2048,2048], BF16 GEMM inputs, FP32 [M,2048] then BF16 | `PackedProjection`, `phase10_cast_bf16` | Row, local weight [2048,256], FP32 CNCL SUM then one BF16 cast |
| Residual | BF16 [M,2048] + [M,2048] | `phase11_residual` through `_add` | Replicated, after reduced output is rounded; no fused Row residual |

[Native model](../integration/vllm_native_mlu/phase12_5/model.py) · [Attention ABI](../integration/vllm_native_mlu/attention.py) · [Native layout](../runtime/native/phase10/layout.mlu)

### RoPE / YaRN compatibility

Theta=10,000, rotary width=64, factor=40, original context=4,096, beta_fast=32, beta_slow=1, mscale=mscale_all_dim=0.707. The checkpoint mixes interpolated and extrapolated inverse frequencies with the YaRN ramp. Its table amplitude ratio is 1. Score scale is `192**(-0.5) * (1 + 0.1 * 0.707 * log(40))**2`, about 0.1147213868.

Checkpoint `apply_rotary_pos_emb` deinterleaves adjacent pairs into halves before `rotate_half`. Native `worker_rope_bf16` reads even/odd lanes and uses the same signs, narrows each product to BF16, then sums and narrows. The native cursor is L−M and is validated against the scheduler's zero-based absolute positions. Prefill starts at 0; decode continues at the committed cursor even if its physical page changes. Shared K is rotated once per token per rank, then copied into the K tail of each local head. Frozen Phase 12.5.4 key-tail and decode acceptance are the existing device evidence. The audit rebuilds the BF16 tables and compares scale at its actual `ctypes.c_float` ABI; the two decimal spellings differ by one FP64 ULP. No enlarged positional contract is inferred. [RoPE replay](../artifacts/logs/phase13_3_1/rope_verified.txt) · [Frozen regression](PHASE12_PRODUCTION_ACCEPTANCE.md#phase12-5-4-root-cause-report)

<a id="phase13-3-1-mla-weight-mapping"></a>

## Phase 13.3.1 — MLA checkpoint weight mapping

The index and every referenced safetensors header identify exactly five BF16 MLA parameters per layer across 27 layers. The inventory contains 135 unique tensors and 1,080 rank records with file name, absolute byte range, original/local shape, axis, rank, offset, replication, reconstruction and SHA-256. Query A/B, Query RMSNorm and standalone K/V or rotary projection weights are absent. [Complete checkpoint inventory](../artifacts/logs/phase13_3_1_checkpoint_inventory.csv) · [Reconstruction evidence](../artifacts/logs/phase13_3_1/inventory_result.txt)

For rank r∈[0,7], own global heads 2r and 2r+1 in increasing order.

| Suffix under model.layers.L.self_attn | Global shape | TP=8 local and ownership |
|---|---|---|
| q_proj.weight | [3072,2048] | [384,2048], rows [384r,384r+384); each head has no-PE 128 then PE 64 |
| kv_a_proj_with_mqa.weight | [576,2048] | Replicate all rows; [0,512) produce latent, [512,576) produce one shared rotary K |
| kv_a_layernorm.weight | [512] | Replicate full gamma |
| kv_b_proj.weight | [4096,512] | [512,512], rows [512r,512r+512); each 256-row head stores K-noPE 128 then V 128 |
| o_proj.weight | [2048,2048] | [2048,256], columns [256r,256r+256); each head contributes 128 consecutive V channels |

Q and KV-B concatenate axis 0 in rank order. O concatenates axis 1 independently for each output row. A single full replica reconstructs KV-A and gamma. The byte audit reads candidate slices directly from the checkpoint and verifies exact reassembly against the original bytes for every layer. The rotated shared K has no separate K-B shard; it originates from the final 64 rows of KV-A. Keeping Q and KV-B as one Phase 13.2 Column section each preserves the checkpoint head-major packing.

<a id="phase13-3-1-tp-design"></a>

## Phase 13.3.1 — TP=8 MLA design

### Selected path

Replicate BF16 hidden states, pre-norm and KV-A/gamma. Column-shard Q and KV-B by contiguous complete heads. Each rank rotates two Q stripes plus the shared KV-A K stripe, materializes its two K/V heads, scans its full local history, computes scores and softmax over sequence positions, and emits [M,2,128] in token-major head order for the output GEMM. Heads are independent. Neither Q/K/V nor attention probabilities need a cross-rank reduction in the normal path. Debug gathers use global head order.

Row-parallel O consumes [M,256] with W [2048,256] and `input_is_parallel=True`. It returns the replicated FP32 sum [M,2048]. Apply the frozen BF16 cast then the existing BF16 residual outside the Linear; passing residual into Phase 13.2 Row would change rounding. O has no bias. Keep one data-plane collective per MLA layer, an FP32 CNCL AllReduce over 8,192 logical bytes per scheduled token per rank. All workers must enter the same ordered calls.

### KV latent alternatives

| Option | Memory and work | Collective, numerical and decode implications |
|---|---|---|
| A. Replicate KV-A and full latent | BF16 W ~2.25 MiB/rank/layer before packing; BF16 latent+rotary 1,152 bytes/token temporary per rank; duplicate projection and norm | No latent collective. Same full-width norm/reduction order and handoff as frozen path. Preferred first implementation. Measure redundant GEMM and pack memory. |
| B. Keep latent distributed | Can shard KV-A rows (64 latent + 8 rotary outputs per rank if evenly split), but head-local KV-B consumes all 512 latent features | Requires global RMS sum and access to the complete normalized latent or a 2-D/partial-product KV-B decomposition. Shared rotary must be assembled with correct pairing. Extra communication and rounding boundaries on decode. |
| C. Shard KV-A then gather | W shrinks to ~0.28125 MiB/rank before packing; gather returns full 576 outputs on each rank | At least one additional gather per layer, 1,152 logical bytes/token after BF16 narrowing or 2,304 at FP32; reproduce original narrowing before norm and rotary. Decode startup overhead can dominate. Benchmark only after A parity. |

The pinned Vendor projections support the A topology. This is source corroboration. Hardware gates apply to the later native two-head implementation.

### API and ABI plan

Use Phase 13.1 `get_tp_group()`, Vendor CNCL device group and `MLUCommunicator`. Reuse Phase 13.2 `ColumnParallelLinear(weight, sections, group, library, bias=None)` and `RowParallelLinear(weight, input_width, group, library, bias=None)`, shard spans, comparisons and runner. For KV-A use the accepted rank-local `PackedProjection` or its Phase 13.2 `LocalGemm` wrapper. Keep communications on the device TP group.

Frozen prepare/append/merge, RoPE, cache views and attention wrapper contain head-count assumptions. A new head-aware boundary must preserve FP32-to-BF16 layout casts, use two head lanes and verify ABI limits. The frozen RoPE kernel loops over exactly 17 stripes. First-stage reuse can stage two local Q stripes plus shared K in a correctly sized 17-stripe buffer, initialize the 14 unused stripes and discard their outputs. Never call that kernel on three allocated stripes. Benchmark padded RoPE separately; a head-aware launch in a later implementation needs independent parity and cannot change frozen source. Keep production TP=1 guard until end-to-end acceptance.

<a id="phase13-3-2-implementation"></a>

## Phase 13.3.2 — Native MLA projections

Use an isolated projection module and rank-local safetensors slice reads. Reuse Phase 13.2 Column, LocalGemm and Row plus the frozen NativeRMSNorm. Preserve FP32 Q and KV outputs, BF16 latent narrowing before native RMSNorm, and FP32 Row SUM before one BF16 cast and residual.

1. Verify shard bounds and identity for each milestone.
2. Capture supplemental unsharded boundary diagnostics from exact frozen activation fixtures. Retain historical references in place. For O input use the frozen single-card attention helper and frozen K/V, solely in the test harness.
3. Run real MLU Q, KV-A, RMSNorm, KV-B and O in sequence, compare every boundary and then run stability and timing.

### Interfaces and fixture scope

`MLAProjection` keeps the five layer-local checkpoint parameters isolated from the model entrypoint. It reads safetensors slices by audited axes and validates BF16 metadata. The hardware probe executes layer 0. It verifies all 40 TP=8 device weight hashes against the Phase 13.3.1 inventory. The inventory and independent source review cover all 27 layers.

The probe reuses the exact frozen prompt bytes and layer-0 embedding, K/V and attention fixtures. Supplemental TP=1 intermediate boundaries are stored under `artifacts/build/phase13_3_2/` because the frozen run did not persist Q, raw KV-A, latent gamma output or KV-B boundaries. Independent FP32 MLU ATen GEMMs cross-check the native TP=1 GEMM outputs. Existing frozen tables and the frozen attention helper serve only to recover the O input in that isolated harness. Historical O output is read from the frozen Phase 12.5.4 reference. Production RoPE and attention, scheduling, cache and the model guard are unchanged.

```bash
set -o pipefail
runner=integration/vllm_native_mlu/phase13_3_2/run_probe.sh
logs=artifacts/logs/phase13_3_2
bash "$runner" 1 reference > "$logs/reference.log" 2>&1
for tp in 1 2 4 8; do
    bash "$runner" "$tp" test > "$logs/tp$tp.log" 2>&1
    printf 'TP=%s launcher status=%s\n' "$tp" "$?"
done
python3 integration/vllm_native_mlu/phase13_3_2/report.py "$logs"
```

Use fresh log paths for a subsequent experiment. The probe writes rank results and releases resources before returning a nonzero status on numerical mismatches.

<a id="phase13-3-2-performance-report"></a>

## Phase 13.3.2 — Projection performance

The measured workload is layer-0 length-17 prefill, repeated 100 times per rank. The chain includes the five projection boundaries and residual, with frozen-replay attention activations supplied to O. The O stage communicates only its FP32 partials. Distributed RoPE, attention and cache are outside the benchmark.

Per-rank synchronized host timings, device stream intervals and native direct-launch device event timings are recorded separately. The direct launch measurement allocates outputs before placing the queue markers and omits the wrapper's padding, allocation and synchronization. CNCL and the composite chain use stage-level device stream intervals. The loader reports checkpoint slicing, H2D, hash verification and native packing together.

Summary p50 and p95 use the slowest rank for each repetition. Speedup divides the TP=1 p50 by the measured group p50. Efficiency divides speedup by TP. Throughput uses 17 logical tokens per chain call. Memory is tracked MLU PyTorch allocator peak plus resident packed buffers calculated from the native 128×128 allocation layout. The memory figures exclude CNCL and driver allocations.

[Rank samples](../artifacts/logs/phase13_3_2/performance.csv) · [Aggregates](../artifacts/logs/phase13_3_2/performance_summary.csv)

### Measured chain latency on MLU590-M9

| TP | p50 ms | p95 ms | tokens/s | Speedup | Efficiency | Aggregate tracked MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.705 | 3.945 | 4589 | 1.00× | 100.0% | 62.50 |
| 2 | 2.545 | 2.835 | 6679 | 1.46× | 72.8% | 65.00 |
| 4 | 2.041 | 2.387 | 8328 | 1.81× | 45.4% | 70.00 |
| 8 | 1.651 | 3.575 | 10298 | 2.24× | 28.1% | 98.00 |

The eight-card median improved by 2.24× in this fixture. TP=4 had a lower p95. These timings were collected before evaluating the final numerical gate, which failed for multi-rank BF16 output.

<a id="phase13-3-3-implementation"></a>

## Phase 13.3.3 — Ordered O projection

The isolated `ordered.mlu` kernel reuses the frozen BANG C BF16 product, lane updates, packing permutation and fold. Each rank keeps its Phase 13.2 row-sharded packed BF16 weight and handles one contiguous, 128-aligned K region. The first rank initializes state, intermediate ranks load and update it, and the final rank folds once. CNCL broadcasts the unfurled FP32 state after each preceding rank and broadcasts the FP32 result from the last rank. The BF16 conversion and residual addition remain after that result. The production Row path and TP=1 model guard remain frozen.

The probe reports each rank's existing Row FP32 partial and CNCL SUM as a control diagnostic, captures the compatible FP32 and BF16 values, and replays the frozen O boundary using the exact Phase 13.3.2 saved AV inputs. That control SUM is independent from the ordered output path. The latter uses multiple collectives and serial rank work because a single folded-output SUM cannot carry the required recurrence.

### Reproduction

From the repository root on the MLU host, use a fresh evidence target for any additional execution.

```bash
set -euo pipefail
d=$(mktemp -d "$PWD/artifacts/logs/phase13_3_3/replay.XXXXXX")
export NEURX_1333_EVIDENCE_DIR="$d"
image=sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756
src=integration/vllm_native_mlu/phase13_3_3
python3 "$src/test_rounding.py"
docker run --rm --pull=never --network=none --read-only \
  --tmpfs /tmp:rw,exec,mode=1777 --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/project,readonly" \
  --mount "type=bind,src=$d,dst=/results" -w /project \
  --entrypoint /bin/bash "$image" "$src/build.sh"
for tp in 1 2 4 8; do
  bash "$src/run_probe.sh" "$tp" test > "$d/tp$tp.log" 2>&1
done
python3 "$src/report.py" "$d"
```

The runner uses the saved `artifacts/build/phase13_3_2/boundary_*.pt` records and reads the original checkpoint and frozen Phase 12.5.4 reference files in place. The independent review compared the recurrence/fold text to the frozen source, verified K tile alignment and checked flow ordering. The final tests use integer views for bitwise stability and record mismatches before collective cleanup.

Set and export `NEURX_1333_EVIDENCE_DIR` to the new evidence directory when repeating these commands. Build into the same directory. The initial ordered run is kept alongside supplementary `verified_tp*.log` traces from the final bitwise and raw-tensor checks. The exporter prefers those supplemental traces when present.

For the optional strategy comparisons, use `--output-axis --no-capture` or `--deferred --no-capture` on the same runner. Store their stdout/stderr in separate subdirectories and export each directory independently with `report.py`. The numerical gate is enforced in TP order for each strategy. A deferred run that returns nonzero is evidence that independent lane states cannot reproduce the frozen path. Preserve its log and stop that sweep at the failing TP.

The frozen-path regression commands are `python3 integration/vllm_native_mlu/phase13_3_2/test_mapping.py` and `bash integration/vllm_native_mlu/phase13_3_2/run_probe.sh 1 test`. Running the latter with TP=2 is the native red control. The repository GEMM CPU regression used the compile flags from `scripts/test.sh` with a Phase 13.3.3 executable and CSV destination.

<a id="phase13-3-3-numerical-acceptance"></a>

## Phase 13.3.3 — Numerical acceptance on MLU590-M9

The ordered row compatibility run advanced from TP=1 through TP=2, TP=4 and TP=8 only after each configuration passed. Each exercises layer 0 with the original BF16 weights, frozen prompt bytes, lengths 1/7/17/65, prefill and the three saved decode steps. The O input is read from the saved Phase 13.3.2 AV boundary, and the historical O target is read from Phase 12.5.4. The criterion remains `abs(error) <= 1e-5 + 1e-5*abs(reference)`.

| Gate | Status | Evidence |
|---|---|---|
| Root cause and midpoint | PASS | Source fold comparison, IEEE midpoint and cancellation tests, old TP=8 snapshot, real old TP=2 failure |
| TP=1 compatibility | PASS | FP32 O, historical BF16 O and residual compare bit-exact |
| TP=2 BF16 O and residual | PASS | Both ranks bit-exact for all 16 fixtures |
| TP=4 BF16 O and residual | PASS | Four ranks bit-exact for all 16 fixtures |
| TP=8 BF16 O and residual | PASS | Eight ranks bit-exact for all 16 fixtures |
| Frozen numerical tolerance | PASS | Zero O/residual violations and zero NaN/Inf |
| 100-run stability | PASS | Bitwise local FP32 partial, ordered FP32, BF16 O and residual checks for each rank |
| Resource cleanup | PASS | All accepted launchers exit 0; rank cleanup and final CNMON recorded |
| Frozen evidence integrity | PASS | 9,599 distinct files verified, including all four checkpoint shards |
| Independent review | PASS | Frozen recurrence/fold, rank/stream ordering, stability comparison and alternative isolation |
| Projection compatibility | PASS | Isolated hardware experiment; production integration is outside this result |

There are 960 rank/boundary comparisons, 720 on the compatible path and 240 on the legacy FP32 control. All 720 ordered comparisons have zero bitwise differences. The CSV contains errors and SHA-256 for each result. The ordered `tensors/` capture contains the old local FP32 partial and CNCL SUM as controls plus ordered FP32, O BF16 and residual BF16. The original case at length 7, decode step 2 and index 1134 returns FP32 `0x3d5e8002` and BF16 `0x3d5f` at TP=8. The legacy control matches the midpoint `0x3d5e8000`.

The Phase 13.3.2 host mapping tests and the repository C GEMM reference test passed. The frozen Phase 13.3.2 hardware probe passed again at TP=1. Its TP=2 negative control reproduced 18 tolerance-mismatch occurrences on each rank and exited nonzero after rank cleanup. Legacy TP=4 and TP=8 reruns were NOT RUN under the stop-on-failure rule. All four ordered configurations passed.

[Accuracy CSV](../artifacts/logs/phase13_3_3/accuracy.csv) · [Host bit patterns](../artifacts/logs/phase13_3_3/host_rounding.log) · [Hardware log](../artifacts/logs/phase13_3_3/hardware.log) · [Source hashes](../artifacts/logs/phase13_3_3/source_hashes.txt)

The full-K output-axis alternative also matched the frozen FP32 and BF16 boundaries bit-for-bit at TP=1/2/4/8. The separate deferred-fold strategy passed TP=1 and failed at TP=2, with three O and three residual tolerance violations per rank. It still folds exactly once. This isolates the missing sequential rounding in independently initialized lane accumulators. Deferred TP=4/8 were NOT RUN after that failure.

[Output-axis accuracy](../artifacts/logs/phase13_3_3/output_axis/accuracy.csv) · [Deferred-fold negative control](../artifacts/logs/phase13_3_3/deferred/accuracy.csv)

<a id="phase13-4-final-evidence-closure"></a>

## Phase 13.4 Final Numerical Evidence Closure

### Scope
- Repository: `/app/shuwen/mlu-operator`
- Runtime image: `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756` (`teleavatar-mlu:v0.1`)
- Python: `/torch/venv3/pytorch_infer/bin/python3`
- Hardware: MLU TP=8 (`mlu:0..7`)
- Frozen fixture policy:
  - required: `artifacts/build/phase12_5_4/native-dump-yarn-scale`
  - forbidden: `artifacts/build/phase12_5_4/native-dump-yarn`

### What was closed in this pass
1. Verified 27-layer execution evidence from corrected replay outputs.
2. Recorded runtime-resolved fixture absolute path + SHA256 per rank.
3. Added explicit O boundaries:
   - local FP32 O partial
   - CNCL FP32 SUM
   - BF16 O output
   - BF16 residual output
4. Verified rank/device mapping for all 8 ranks.
5. Verified cleanup events for all ranks and captured final CNMON snapshot attempt.

### Evidence (non-overwriting)
- Corrected closure layered replay: `artifacts/logs/phase13_4/corrected_replay_closure/tp8_test_gate8_closure3.log`
- Corrected closure full-model replay: `artifacts/logs/phase13_4/corrected_replay_closure/tp8_full_model_closure3.log`
- Closure summary: `artifacts/logs/phase13_4/corrected_replay_closure/closure3_summary.json`
- CNMON snapshot: `artifacts/logs/phase13_4/corrected_replay_closure/cnmon_final_state.log`
- Preserved previous failed closure attempt: `artifacts/logs/phase13_4/corrected_replay_closure/tp8_test_gate8_closure2.log`

### Required distinction
#### A) Original diagnostic failure cause
- The original divergence investigation used wrong fixture lineage (`native-dump-yarn`) in older launcher paths.
- The corrected acceptance path explicitly enforces `native-dump-yarn-scale` (see precheck/audit evidence).

#### B) Corrected fixture validation result
- `closure3_summary.json` reports:
  - `fail_event_count = 0`
  - `all_27_layer_gates_passed_on_8_ranks = true`
  - `all_27_layers_seen_in_full_model = true`
  - `fixture_recorded_for_all_ranks = true`
- Runtime fixture path + SHA256 is recorded on all ranks in accuracy rows (`fixture_path`, `fixture_sha256`).

#### C) Remaining genuine failures
- No numerical tolerance failure was observed in corrected closure replay logs.
- No runtime gate failure was observed in corrected closure replay logs.

### Explicit gate outcomes for this closure
- Corrected fixture validation: **PASS**
- Layer-level numerical acceptance (TP8, all 27 layers): **PASS**
- Full 27-layer prefill/decode acceptance: **PASS**
- End-to-end logits acceptance (`model_logits` rows): **PASS**
- End-to-end token-id acceptance (`token` rows): **PASS**
- Rank/device mapping (0..7 -> mlu:0..7): **PASS**
- Rank cleanup events (0..7): **PASS**
- Final CNMON state capture command: **PASS (command exit code 0; command output empty in this image)**

### Notes
- No production kernel logic was modified for numerical “compensation”.
- Existing frozen references and tolerances were preserved.
- Prior failed closure logs were preserved and not overwritten.
