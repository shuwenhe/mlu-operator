# Phase 12.3 installed Vendor vLLM APIs

The audited SDK has vLLM 0.25.1 and vllm_mlu 0.25.1+mlu0.16.0.pt212. Installed core source is at `/torch/venv3/pytorch_infer/lib/python3.12/site-packages/vllm`. The vendor snapshot is at `/workspace/vllm-v0.25.1+mlu0.16.0.pt212/vllm_mlu`. [Version and source evidence](PHASE12_VLLM_API_AUDIT.md)

| Interface | Installed call site |
|---|---|
| Model registration | Core `model_executor/models/registry.py:1011`, `ModelRegistry.register_model`. The vendor general plugin registers vendor DeepSeek implementations for checkpoint architecture resolution. |
| Configuration and model loading | The original checkpoint declares `DeepseekV2ForCausalLM`. Vendor `v1/worker/gpu_model_runner.py:3336` implements `load_model(self, load_dummy_weights=False)`. |
| Worker and device | Vendor `platforms/mlu.py:202` sets `MLUWorker`; torch_mlu bridges device tensors and CNRT streams. |
| Scheduler and runner | Core `v1/core/sched/scheduler.py:396` implements `schedule(self, throttle_prefills=False) -> SchedulerOutput`. Vendor `v1/worker/gpu_model_runner.py:2096` executes scheduled work. |
| Allocation and completion | Core `v1/core/kv_cache_manager.py:248` implements `allocate_slots` and line 466 implements `free(self, request)`. |
| Cache spec and block tables | Core `v1/kv_cache_interface.py` declares `KVCacheSpec` and `MLAAttentionSpec`. Core `v1/worker/block_table.py:141` computes slot mappings. The vendor block-table patch uses int32 slots. |
| Attention backend | Vendor `v1/attention/backends/mla/tmo_flashmla.py:645` implements `TMOFlashMLAImpl.forward(layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, kwargs={})`. |
| Attention metadata | The same vendor backend builds metadata from `block_table_tensor` and `slot_mapping` around lines 262–263. Its forward path selects distinct prefill/decode blocks and sequence lengths. |
| Model attention call | Vendor `model_executor/models/deepseek_v2.py:1303` constructs `MLAAttention` and lines 1863–1875 associate forward metadata and cache with the layer. |
| CustomOp | Core `model_executor/custom_op.py:332` exposes `register_oot`. The project uses distinct CustomOps for isolated native probes. |

The vendor MLA decode path can store compressed keys with no separate value tensor. The accepted native Phase 10 path needs materialized head-major K and V. A project backend requires its own cache specification, model call site and scheduler-aware allocation binding. Passing the vendor cache to the native scatter/gather launcher would be a layout error.

The control engine selected `vllm_mlu.v1.core.sched.scheduler.SchedulerWithProfiler`. The first trace hooked the core scheduler and observed cache frees and tokens only. The follow-up hooks the vendor override directly. The control keeps worker execution on the vendor path and changes no installed source.

The vendor scheduler override is at `v1/core/sched/scheduler.py:266` with signature `schedule(self, throttle_prefills: bool = False) -> SchedulerOutput`. `ModelRegistry.register_model(model_arch: str, model_cls: type[nn.Module] | str) -> None` accepts a lazy module/class path. The vendor runner accepts `execute_model(self, scheduler_output, intermediate_tensors=None)`. The cache allocator accepts the request and number of new tokens plus optional computed blocks, lookahead and reservation fields. It returns `KVCacheBlocks | None`. The vendor MLA cache shape function takes `(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto")` and returns `(1, num_blocks, num_kv_heads, block_size, head_size)`. These signatures and the cache/scheduler digests were read from the installed image.
