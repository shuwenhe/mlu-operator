# Phase 13.2 — Tensor Parallel Linear

## Compatibility audit and implementation plan

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

## Reproduction

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
