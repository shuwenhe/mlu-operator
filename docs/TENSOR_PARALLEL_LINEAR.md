# Tensor-parallel linear operators and weight sharding

- [Phase 13.1 — Vendor vLLM MLU distributed architecture](#phase13-1-vendor-distributed-audit)
- [Phase 13.2 — Tensor Parallel Linear](#phase13-2-tensor-parallel-linear)
- [Phase 13.2 — Checkpoint sharding](#phase13-2-weight-sharding)
- [Phase 13.2 — Numerical and hardware acceptance](#phase13-2-numerical-acceptance)
- [Phase 13.2 — Tensor Parallel Linear performance](#phase13-2-performance-report)

<a id="phase13-1-vendor-distributed-audit"></a>

## Phase 13.1 — Vendor vLLM MLU distributed architecture

The audit uses the Phase 12 Vendor image digest `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`. The hardware run uses eight MLU590-M9 devices. The runtime reports CNCL 1.30.10, CNRT 7.7.3 and driver 6.5.52.

### Worker and process groups

- `MLUPlatform` advertises `cncl` and resolves the device communicator to `MLUCommunicator`.
- `MLUWorker.init_device` maps the local rank to a visible MLU and sets it before calling the base worker distributed initializer. It then calls `initialize_model_parallel_mlu`, which manages additional Vendor groups for attention sequence parallelism, embedding/logits, dense MLP, and MoE TP/EP. Vendor MoE group creation consults `enable_expert_parallel`.
- Base vLLM `GroupCoordinator` creates device and CPU groups. The MLU communicator implements AllReduce using `torch.distributed.all_reduce` on the device group. Fixed-size AllGather and ReduceScatter inherit implementations that call `all_gather_into_tensor` and `reduce_scatter_tensor`. The Vendor variable-size paths have separate code.
- The experiment calls the same base vLLM group code with a TP=8 configuration, disables custom AllReduce, and validates the actual TP backend, communicator and local MLU on each rank. It exercises the three fixed-size collectives without loading model weights. Vendor-specific auxiliary groups and the full engine-worker lifecycle are covered by the source audit only.

[Vendor source hashes and symbol locations](../artifacts/logs/phase13_1_source_audit.log)

### Implications for NeurX DeepSeek-V2-Lite

- **Linear layers.** Vendor DeepSeek imports its ColumnParallel, RowParallel and replicated linear paths. For the native model, column-parallel checkpoint loading must slice output channels, including the separate gate/up portions. Row parallel must slice inputs and reduce partial outputs before applying any replicated bias or residual.
- **MLA and cache.** Vendor attention reshapes Q/KV using the local head count and has compressed latent-cache decode paths. The checkpoint has 16 attention heads, giving two per rank at TP=8. The frozen native attention path uses materialized K/V. A native TP path must size per-rank cache pages for two heads and preserve consistent scheduler slot/block ownership across ranks.
- **MoE.** The checkpoint specifies 64 routed and two shared experts. Eight-way EP can assign eight routed experts to each rank. Routing and combine must preserve checkpoint expert IDs and top-k weights across ranks. Shared-expert execution and its final contribution require a separate test.
- **Acceptance.** The collective fixture can establish device communication and rank participation. Weight ownership, prefill/decode logits, greedy tokens, cache lifecycle, and model performance require subsequent hardware runs against the frozen single-card reference.

### Reproduction

Run on the MLU host where the eight devices and pinned local Docker image are available.

```bash
set -o pipefail
bash integration/vllm_native_mlu/phase13_1/run_cncl_probe.sh 2>&1 | tee artifacts/logs/phase13_1_cncl_final.log
```

### Hardware result — 2026-11-22

**PASS for the TP communication probe.** The launcher exited 0. Ranks 0–7 each used their matching MLU ordinal and reported `cncl` and `MLUCommunicator`. AllReduce, AllGather and ReduceScatter matched the expected tensors exactly on every rank for FP32 and BF16, a total of 48 checks. CNCL reported MLU-Link rings. This is a correctness fixture using 2×16 elements per rank for AllReduce and AllGather and 16×16 input per rank for ReduceScatter.

Startup retries resolved hostname resolution in the isolated container with loopback rendezvous, Python cache initialization with writable temporary cache paths, and TP initialization with an active vLLM config. The experiment used the frozen image and left the Phase 12.5.4 model untouched. The RMSNorm/GEMM scaling CSVs and the Phase 12.5.4 token fixture retained their pre-run SHA-256 hashes.

[Per-rank results](../artifacts/logs/phase13_1_cncl_accuracy.csv) · [Raw hardware log](../artifacts/logs/phase13_1_cncl_final.log)

<a id="phase13-2-tensor-parallel-linear"></a>

## Phase 13.2 — Tensor Parallel Linear

### Compatibility audit and implementation plan

Frozen Vendor image `8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`.

| Source | Interface and constraint |
|---|---|
| Installed `vllm/model_executor/layers/linear.py` (SHA-256 `1f1ac36402338d32643d4d707386852caba188c9fe04cd5fbc1d4a6b739f4507`) | `ColumnParallelLinear.weight_loader(self,param,loaded_weight)` narrows output axis 0 by rank; `MergedColumnParallelLinear.weight_loader(self,param,loaded_weight,loaded_shard_id)` slices each projection separately; `RowParallelLinear.weight_loader(self,param,loaded_weight)` narrows input axis 1; `ReplicatedLinear.weight_loader(self,param,loaded_weight)` copies an exact-shape parameter. |
| Installed `vllm/model_executor/parameter.py` (SHA-256 `ff6054fbd19ec932c548d562c6f4cc4506383b71cbe411fcfd554c8b1d87b510`) | `load_column_parallel_weight(loaded_weight)`, `load_row_parallel_weight(loaded_weight)`, and merged loader use explicit output/input axes and rank offsets. |
| Vendor `vllm_mlu/model_executor/layers/linear.py` (SHA-256 `fba89b1bc1685b20042678dda61d7ccdbb5175c99d94316e947b1647803f86fe`) | `UnquantizedLinearMethod.apply(self,layer,x,bias=None,residual=None)` dispatches `mlu_ops.matmul`. Vendor row forward accepts a TP group and fuses bias/residual on rank zero before summation. The phase requires post-reduction addition. |
| Installed `vllm/distributed/parallel_state.py` and Vendor `vllm_mlu/distributed/device_communicators/mlu_communicator.py` | Reuse `GroupCoordinator.all_reduce(input_)`, `all_gather(input_,dim=-1)` and its CNCL device group. Vendor `MLUCommunicator.all_reduce(input_)` mutates using `dist.all_reduce`; gather inherits `all_gather_into_tensor`. |
| `operators/gemm/include/mlu_gemm.h`, `operators/gemm/native/matrix/matrix_gemm.h`, `integration/vllm_native_mlu/phase12_4/projection.cpp` | Host `mlu_gemm` owns transfers/queue. Device `phase124_pack(queue,kn,k,n,packed)`, `phase124_project(queue,x,out,packed,rows)`, and `phase124_release(packed)` use BF16 operands and FP32 output on the active CNRT queue. Packed K and N require multiples of 64 up to 10944. Use local zero padding and trim output. The bridge accepts at most 128 rows per launch. Wider output weights use multiple local native packs. FP32 uses ATen MLU matmul with TF32 disabled. |

1. Add isolated Column Linear and verify local checkpoint slices, packed output order and explicit gather.
2. Add Row Linear after the Column hardware smoke. Reduce FP32 partials and add bias/residual once per replicated result.
3. Verify hashes and exact reconstruction for deterministic fixtures and selected model checkpoint parameters using shard-only reads.
4. Run independent single-device reference and TP=1/2/4/8 acceptance with 100 iterations per configuration, then summarize timings, peak allocation and numerical metrics.

### Reproduction

The runner uses the audited Phase 12.5.4 `artifacts/build/phase12_5_4/nativelibs/projection/libprojection.so` (SHA-256 `c399d3e4af1b43891502791179b4027bed0993d3175df3ca6bbd52f5d4fbeb49`). The frozen image already contains the SDK. To reproduce the bridge from source in that image, run `bash integration/vllm_native_mlu/phase12_4/build_projection.sh /results/phase13_2_projection` with the repository read-only and a writable results mount. Supply the separate binary path through `NEURX_TP_PROJECTION_LIBRARY` as a path inside that container.

```bash
set -o pipefail
runner=integration/vllm_native_mlu/phase13_2/run_probe.sh
log=artifacts/logs/phase13_2
bash "$runner" 1 reference 2>&1 | tee "${log}_reference.log"
bash "$runner" 8 checkpoint 2>&1 | tee "${log}_checkpoint_tp8.log"
for tp in 1 2 4 8; do
    for mode in column row; do
        bash "$runner" "$tp" "$mode" 2>&1 | tee "${log}_${mode}_tp${tp}.log"
    done
done
bash "$runner" 8 precision 2>&1 | tee "${log}_precision_tp8.log"
for tp in 1 8; do
    bash "$runner" "$tp" wide_checkpoint 2>&1 | tee "${log}_wide_checkpoint_tp${tp}.log"
done
```

The TP=1 device process group is CNCL and vLLM omits a device communicator for its singleton group. TP=2/4/8 require `MLUCommunicator`. Vendor Gloo CPU groups provide control-plane rendezvous. Linear tensor collectives use the CNCL device group. Runtime code lives in the new Phase 13.2 module, so the production model and its TP guard remain frozen.

Review artifacts are available in the [exact installed API signatures](../artifacts/logs/phase13_2_vendor_signatures.log) and the [initial Git snapshot and source hashes](../artifacts/logs/phase13_2_source_hashes.txt).

<a id="phase13-2-weight-sharding"></a>

## Phase 13.2 — Checkpoint sharding

Each weight is logically `[N,K]`. For column sharding, a projection section of width `S` at offset `B` contributes `S / TP` rows starting at `B + rank × (S / TP)`. Divide each packed section separately. The local tensor concatenates sections in checkpoint order. A gather for each section recovers section-major order. Column bias follows the same spans.

Row sharding reads input columns at `rank × (K / TP)` with width `K / TP`. Row bias is replicated and applied to the reduced result. The optional residual is likewise added on each rank after reduction. Replicated inputs are narrowed on-device. Pre-sharded inputs must have the exact local width.

Positive dimensions must divide the TP size on the sharded axis. In a packed column projection every section must divide independently. Native BF16 alignment pads each local shard's K and N to 64 and crops output to the logical local N. Logical output widths above 10,944 are tiled into separate local GEMMs. Padding and GEMM tiling preserve the checkpoint spans. The native packed buffer allocates complete 128×128 storage tiles.

The hardware fixture generates weight values directly at rank-local indexes. The real checkpoint probe separately uses safetensors slice reads for six layer-0 Attention and MLP projections and verifies the checksum after an MLU round trip. The host verifier reads the checkpoint independently and compares each reconstructed tensor byte-for-byte via SHA-256. It also verifies the ordered gate/up packing. See the [rank-level shard CSV](../artifacts/logs/phase13_2_weight_shards.csv) and [checkpoint reconstruction log](../artifacts/logs/phase13_2_checkpoint_device_verify.log).

The eight-rank checkpoint probe recorded 48 device-verified weights. The six tensors and the packed gate/up ordering reconstructed exactly at TP=1/2/4/8. A separate real-weight forward projected a single input through both layer-0 gate/up sections, each with 10,944 output channels. TP=1 used two native packs. At TP=8 each rank loaded 1,368 channels from each section and the ordered gather matched the independent TP=1 reference with maximum absolute error at or below 5.961e-8. The shard CSV also includes 16 TP=8 synthetic bias records that verify sharded Column bias and replicated Row bias by content.

<a id="phase13-2-numerical-acceptance"></a>

## Phase 13.2 — Numerical and hardware acceptance

The independent TP=1 reference is an unsharded FP32 ATen MLU GEMM with TF32 disabled. Both input precisions use the same deterministic index-based fixture. Reference BF16 operands are converted to FP32 before GEMM. The comparison covers FP32 outputs, including the native BF16 path's FP32 accumulation, FP32 CNCL reduction, and final bias/residual additions.

The comparison predicate uses `atol=2e-5` and `rtol=2e-5` against the FP32 reference. The absolute term handles small values near zero. The relative term allows FP32 rounding differences between native tiles, the unsharded reference and rank-order summation over up to 7,168 products. BF16 narrowing after the sum is outside this phase. The CSV records maximum absolute/relative error, mean absolute error, maximum FP32 ULP distance, bitwise mismatch count, tolerance mismatch count and the first flattened bitwise mismatch.

Each Column run checks the local result against its checkpoint-ordered output slice and checks the explicit gather against the full reference. Each Row run checks the partial against an independent local ATen GEMM, then compares the reduced result with the global reference for both replicated and pre-sharded input. The stability loop checks outputs on every repetition against the first validated results. Dimension rejection checks cover invalid shard sizes, malformed activation widths and malformed residual shape.

[Accuracy CSV](../artifacts/logs/phase13_2_accuracy.csv) · [Hardware log](../artifacts/logs/phase13_2_hardware.log)

### Results on 8 × MLU590-M9

| Gate | Result | Evidence |
|---|---|---|
| Vendor API compatibility | PASS | Frozen Vendor source audit, CNCL device group and MLUCommunicator on every multi-rank run |
| Column Parallel | PASS | Per-rank GEMM and ordered gather, including actual layer-0 gate/up |
| Row Parallel | PASS | Independent partial GEMM, replicated and pre-sharded inputs, post-reduction bias and residual |
| Weight sharding | PASS | 48 actual MLU checkpoint shard hashes and exact reconstruction at TP=1/2/4/8 |
| BF16 numerical parity | PASS | 318 comparisons; maximum absolute error 1.526e-5 |
| FP32 numerical parity | PASS | 300 comparisons; maximum absolute error 1.907e-5; FP32-sensitive probe passed on all eight ranks |
| TP=2 | PASS | Both modes, four shapes, both dtypes, both ranks |
| TP=4 | PASS | Both modes, four shapes, both dtypes, all four ranks |
| TP=8 | PASS | Both modes, four shapes, both dtypes, all eight ranks, CNCL MLU-Link |
| 100-run stability | PASS | 100 repetitions across 64 configurations; 24,000 per-rank timing rows |
| Performance | PASS | Global and local timings, communication, load, throughput, speedup, efficiency and peak tracked allocation |
| Resource cleanup | PASS | Rank cleanup recorded throughout; final CNMON reported no running processes |

All 618 comparisons had zero tolerance mismatches. The three host sharding unit tests passed. The malformed activation, residual and weight shape checks also passed on-device. The TP=8 FP32-sensitive probe distinguished FP32 results from BF16 and TF32 truncation and checked bias ownership by numerical output and per-rank SHA-256.

The sweep launcher stopped once after Column TP=1 emitted its cleanup record when the launcher file was edited concurrently. The 800 complete TP=1 Column samples were retained and the remaining launches resumed separately. The continuation used the same hardware and pinned image. An independent review checked the shard mappings, FP32 precision probe, wide tiling and post-reduction ordering.

[Architecture audit](#phase13-2-tensor-parallel-linear) · [Weight sharding](#phase13-2-weight-sharding) · [Performance report](#phase13-2-performance-report)

<a id="phase13-2-performance-report"></a>

## Phase 13.2 — Tensor Parallel Linear performance

The benchmark runs one Linear across a Vendor TP group. The reported throughput is the logical global `2 × M × K × N` operation count divided by distributed latency.

- **Local GEMM** is a synchronized wall-clock measurement of the per-rank GEMM path, including local padding or chunking when applicable.
- **Communication** independently measures the explicit Column gather or Row AllReduce, with device synchronization. The duration includes launch, synchronization and rank skew. Its ratio to end-to-end latency is a comparison of separate measurements and does not partition the end-to-end time.
- **Weight load** measures transfer of the already sliced CPU shard to its rank's MLU plus local bias transfer and native packing, where applicable. The timer starts after checkpoint slicing.
- **End-to-end Column** reports both rank-local output and the independently measured GEMM + explicit ordered gather. The global metric uses the latter. **End-to-end Row** includes rank-local GEMM, FP32 AllReduce, bias and residual addition.
- **Peak per card** is the PyTorch MLU allocator's peak plus the bytes of the live native BF16 packed buffer. Driver, CNCL and unrelated device allocations are outside this measurement.
- Aggregate times average the slowest rank for each repetition. Speedup divides matching TP=1 latency by measured TP latency. Efficiency divides that speedup by TP. The detailed CSV retains results for each matrix size, dtype, mode and TP setting.

[Per-rank column samples](../artifacts/logs/phase13_2_column_parallel.csv) · [Per-rank row samples](../artifacts/logs/phase13_2_row_parallel.csv) · [Aggregates](../artifacts/logs/phase13_2_performance.csv)

### Measured global latency

All 64 configurations completed 100 repetitions per rank. Latencies are in ms. The TP=8 efficiency and throughput use the same global latency definition.

| Mode | Case | Dtype | TP=1 | TP=2 | TP=4 | TP=8 | TP=8 speedup | Efficiency | TP=8 TFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Column | Small | bfloat16 | 0.477 | 0.814 | 0.928 | 0.996 | 0.48× | 6.0% | 0.034 |
| Column | Small | float32 | 0.178 | 0.632 | 0.719 | 0.666 | 0.27× | 3.3% | 0.050 |
| Column | Medium | bfloat16 | 13.382 | 7.416 | 4.250 | 2.692 | 4.97× | 62.1% | 0.798 |
| Column | Medium | float32 | 0.190 | 0.690 | 0.680 | 0.701 | 0.27× | 3.4% | 3.065 |
| Column | Large | bfloat16 | 401.016 | 199.098 | 101.324 | 51.507 | 7.79× | 97.3% | 1.334 |
| Column | Large | float32 | 1.043 | 1.392 | 0.761 | 0.748 | 1.39× | 17.4% | 91.865 |
| Column | Projection | bfloat16 | 85.491 | 43.153 | 22.122 | 11.725 | 7.29× | 91.1% | 1.282 |
| Column | Projection | float32 | 0.307 | 0.605 | 0.642 | 0.814 | 0.38× | 4.7% | 18.463 |
| Row | Small | bfloat16 | 0.435 | 0.548 | 0.703 | 0.670 | 0.65× | 8.1% | 0.050 |
| Row | Small | float32 | 0.135 | 0.294 | 0.336 | 0.367 | 0.37× | 4.6% | 0.091 |
| Row | Medium | bfloat16 | 13.025 | 7.533 | 4.587 | 3.170 | 4.11× | 51.4% | 0.677 |
| Row | Medium | float32 | 0.145 | 0.421 | 0.484 | 0.421 | 0.34× | 4.3% | 5.100 |
| Row | Large | bfloat16 | 397.630 | 209.065 | 110.759 | 61.985 | 6.41× | 80.2% | 1.109 |
| Row | Large | float32 | 1.042 | 2.412 | 1.665 | 1.945 | 0.54× | 6.7% | 35.336 |
| Row | Projection | bfloat16 | 87.618 | 43.436 | 23.581 | 12.868 | 6.81× | 85.1% | 1.168 |
| Row | Projection | float32 | 0.315 | 0.434 | 0.795 | 0.606 | 0.52× | 6.5% | 24.789 |

BF16 benefited most from sharding the large local GEMM. TP=8 accelerated Large by 7.79× for Column and 6.41× for Row, and Projection by 7.29× and 6.81×. Small incurred more overhead than the shorter local computation saved. The FP32 ATen path already had low single-card latency. Only the Large Column FP32 case improved at TP=8 in this run.
