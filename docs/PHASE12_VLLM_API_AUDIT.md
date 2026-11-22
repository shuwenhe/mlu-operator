# Phase 12 vendor vLLM API audit

The pinned SDK image is `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`. It contains vLLM 0.25.1, distribution 0.25.1+empty, vendor extension 0.25.1+mlu0.16.0.pt212, Python 3.12.3 and torch_mlu 1.34.1+torch2.12.1. CNCC 5.7.6, CNRT 7.7.3, CNNL 2.2.18 and driver 6.5.52 were verified on the target. The vendor source snapshot has no Git metadata. The image digest and installed-source hashes in the Phase 11.5 vLLM report identify the inspected build. [Version and source evidence](VLLM_MLU_COMPATIBILITY_AUDIT.md)

| Interface | Installed implementation |
|---|---|
| Platform | The vendor platform plugin selects `MLUPlatform`, its attention backend, custom KV specs and an `MLUWorker` when the worker option is automatic. |
| Worker | `vllm_mlu.v1.worker.gpu_worker.MLUWorker` and `MLUModelRunner` provide model loading and scheduled `execute_model`. |
| Model configuration | The vendor general plugin registers DeepSeek models. `ModelRegistry.register_model` is available for an explicitly selected adapter. The original checkpoint architecture string is `DeepseekV2ForCausalLM`. |
| Operator dispatch | `CustomOp.register_oot` is present. The vendor also patches native and OOT norm dispatch. Prior isolated probes use distinct CustomOp names to avoid collisions. |
| Cache | `KVCacheSpec` and `MLAAttentionSpec` describe engine allocations. `TMOFlashMLAMetadataBuilder` forwards `block_table_tensor` and `slot_mapping`, then splits prefill and decode. The vendor block-table patch uses int32 slot indices. |
| Device | Import torch_mlu before vLLM in direct probes to register the MLU dispatch key. Use the active torch_mlu stream's CNRT queue handle. |

The vendor MLA cache stores the vendor layout. The project adapter needs its own cache specification sized for materialized per-head 192-word K and 128-word V. The scheduler must retain allocation, ownership and reclamation. A device adapter can translate slot indices on writes and block-table indices on reads into the existing head-major native attention layout. Once the cache specification is bound to a project model, verify all 27 layers with scheduler-provided metadata before testing token parity, batching, serving or speed.

The standalone runner and the vendor engine have no verified execution link to the separate NeurX checkout. [Runtime architecture audit](INFERENCE_RUNTIME_ARCHITECTURE_AUDIT.md)
