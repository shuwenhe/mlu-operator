# Native DeepSeek-V2-Lite model binding

**BLOCKED.** The vendor control ran the real 27-layer checkpoint and returned four tokens. The repository's materialized cache specification and worker-compatible reshape function are implemented and component-tested. The model-registry entry, worker allocation binding and all-layer native decoder invocation remain pending.

The new projection C ABI reuses the original device RHS packer and checkpoint BF16 GEMM candidate. It accepts the original BF16 weight in [N,K] order on MLU, transposes it to contiguous [K,N] for the device packer, and passes the resulting vendor-independent packed handle to the frozen FP32-accumulating projection. The adapter synchronizes the current queue and exposes explicit close. The model binding must retain these handles according to layer lifetime, use the separate Phase 11.5 router and chunk LM head columns as in the original runner. `q_lora_rank=null` requires only the direct Q projection.

The vendor worker calls a backend-specific reshape hook when present. To use the tested native cache format, the native model needs a model registry entry, one cache spec per layer and a backend exposing that reshape hook. Each layer must bind projected Q/K/V and scheduler metadata to the Phase 12.3 adapter, then use native attention output projection and the original residual, dense or MoE paths. [Cache mapping](PHASE12_4_MLA_CACHE_LAYOUT_MAPPING.md)

## Live projection control

`LLM.apply_model` installed an explicit diagnostic replacement for Layer 0's attention output projection after the vendor worker loaded the checkpoint. The exact accepted BF16 projection and Phase 10 cast ran on a nondefault CNRT queue for a batch of four prompts with lengths 1, 7, 17 and 65. The five observed calls processed 89, 1, 4, 4 and 4 rows. The call with 89 rows processed the combined prefill. Later calls included incremental decode. The 16 greedy IDs matched the unmodified vendor run. The test restored the vendor projection and released its native packed handle. The control used vendor attention, KV pages and other layers, and does not satisfy the native model registration or all-layer gates.

## Reproduction

Use the image, cache environment and device mounts from the [vendor API audit](VLLM_MLU_COMPATIBILITY_AUDIT.md). From the repository root inside the image:

```bash
py=/torch/venv3/pytorch_infer/bin/python3
bash integration/vllm_native_mlu/phase12_4/build_projection.sh /vout/phase12_4-live
"$py" -m integration.vllm_native_mlu.phase12_4.vendor_cache_trace
"$py" -m integration.vllm_native_mlu.phase12_4.cache_spec_probe \
  /vout/phase12/libpaged_kv.so /vout/phase12_4-accuracy.csv
"$py" -m integration.vllm_native_mlu.phase12_4.projection_probe \
  /vout/phase12_4-live/libprojection.so /checkpoint /phase11/boundaries/normalized-1.bin \
  /vout/phase12_4-projection.csv
"$py" -m integration.vllm_native_mlu.phase12_4.live_projection \
  /vout/phase12_4-live/libprojection.so
```
