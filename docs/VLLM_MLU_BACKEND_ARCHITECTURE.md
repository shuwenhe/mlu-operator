# Vendor vLLM APIs and native CustomOp probes

- [Inference runtime architecture audit](#inference-runtime-architecture-audit)
- [vLLM MLU compatibility audit](#vllm-mlu-compatibility-audit)
- [Native MLU backend for vLLM](#vllm-mlu-backend-architecture)
- [vLLM native MLU execution plan](#vllm-mlu-integration-plan)
- [vLLM native MLU numerical acceptance](#vllm-mlu-numerical-acceptance)
- [Phase 12 vendor vLLM API audit](#phase12-vllm-api-audit)
- [Phase 12.3 installed Vendor vLLM APIs](#phase12-3-vendor-api-audit)

<a id="inference-runtime-architecture-audit"></a>

## Inference runtime architecture audit

The accepted Phase 11 execution path is a **standalone Native Runner**. [model_run.cpp](../runtime/native/phase11/model_run.cpp) owns the 27-layer loop, checkpoint shard maps, packed-weight leases, per-layer Phase 10 cache state, native LM head and argmax. The command-line process creates its own native worker and CNRT queue. [build.sh](../runtime/native/phase11/build.sh) links repository-native objects and CNRT. The model process has no vLLM/NeurX runtime binding. The Python checkpoint and tokenizer programs are separate tools.

The separate NeurX checkout under `/app/shuwen/.shuwen/neurx` contains experimental native MLU/GEMM source. The standalone link and execution evidence establish repository-native/CNRT execution only. A NeurX execution claim would require a linked or loaded component and a runtime trace proving that it handled the request.

The pinned SDK's vendor vLLM platform used its own engine, MLU worker and DeepSeek model to complete an offline `LLM.generate` request. The repository's vLLM CustomOp probe exercised RMSNorm, router projection and the materialized QK–Softmax–AV chain independently of the vendor model. Full integration requires a registered model calling the repository-native backend and consuming vLLM's block allocation and scheduling metadata.

[Standalone hardware evidence](../artifacts/logs/phase11_hardware.log) · [vLLM hardware evidence](../artifacts/logs/vllm_mlu_hardware.log)

<a id="vllm-mlu-compatibility-audit"></a>

## vLLM MLU compatibility audit

The pinned Cambricon SDK image `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756` exposes PCI `0000:4f:00.0` as MLU device 0. The host Python lacks the required packages. The image contains Python 3.12.3, PyTorch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, and Transformers 5.11.0. The vLLM module reports 0.25.1, the distribution is 0.25.1+empty, and the vendor distribution is vllm_mlu 0.25.1+mlu0.16.0.pt212. Runtime versions on card 0 were driver 6.5.52, CNRT 7.7.3 and CNNL 2.2.18. CNCC 5.7.6 compiled the shared bridge for `mtp_592`. The vendor source snapshot under `/workspace/vllm-v0.25.1+mlu0.16.0.pt212` has no Git metadata. The image digest and interface file hashes pin the inspected implementation.

| Surface | Evidence and adaptation |
|---|---|
| Platform and PyTorch device | The vendor `vllm.platform_plugins` entry point resolves to `vllm_mlu.platforms.mlu.MLUPlatform`. It enumerated one MLU590-M9 and allocated an MLU tensor. Import `torch_mlu` before `vllm` in standalone probes to register the dispatch key. |
| Worker and runner | The vendor platform sets `vllm_mlu.v1.worker.gpu_worker.MLUWorker`. Retain its device initialization, allocation accounting, queue handling and engine process lifecycle. Select a project model explicitly through the model registry in each process. |
| Operator hooks | Installed `CustomOp` supports registration and OOT dispatch under an active `VllmConfig`. The vendor also applies its own RMSNorm hijack with a fused-residual signature. The project bridge registers a distinct op so selection is explicit. |
| Attention and KV | `MLUPlatform.get_attn_backend_cls` selects the vendor backend. The MLA metadata builder provides `block_table_tensor` and `slot_mapping`. It splits prefills and decodes. A project cache adapter must consume this metadata and use a matching cache specification. |
| MoE | The vendor DeepSeek model defines its own MoE path. A project model adapter must call the existing native routing and expert kernels explicitly. |
| CNRT queue | torch_mlu exposes the current stream's `mlu_stream` handle. A real nondefault handle was passed to the native launcher. The default stream reported sentinel value 2 and was outside the probe. |
| Standalone runner | Builds with CNCC, g++, CNRT and accepted packed-weight objects. It owns a worker, request buffers, weight registry and queue. It has no vLLM worker lifecycle or paged allocator binding. |

An offline vLLM `LLM.generate` call loaded the four original checkpoint shards and generated four tokens per request for two prompts on the vendor MLU backend. The read-only sandbox requires writable Triton, HF, inductor and vLLM caches. The initial import order caused an unknown MLU dispatch key and the first retry tried to write `/.triton`. Setting the cache paths fixed initialization. The successful vendor run logged a nonfatal usage-statistics write failure under `/.config`.

[Installed-source and build evidence](../artifacts/logs/vllm_mlu_build.log) · [Card and model execution](../artifacts/logs/vllm_mlu_hardware.log) · [Upstream plugin contracts](https://docs.vllm.ai/en/stable/design/plugin_system/) · [Upstream CustomOp contracts](https://docs.vllm.ai/en/stable/design/custom_op/)

### Reproduction in the pinned image

Mount the repository read-only at `/project`, the original model read-only at `/checkpoint`, the Phase 11 build at `/phase11`, the router captures at `/p115`, the Phase 9.1.2 fixtures at `/out`, the Phase 9.3 fixtures at `/av-out`, and an empty writable output directory at `/vout`. Expose the card 0 CNRT device nodes only. From `/project`:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCHINDUCTOR_CACHE_DIR=/tmp/phase-inductor TRITON_CACHE_DIR=/tmp/triton
export HF_HOME=/tmp/hf VLLM_CACHE_ROOT=/tmp/vllm LOGNAME=phase_audit
py=/torch/venv3/pytorch_infer/bin/python3
bash integration/vllm_native_mlu/build.sh /vout/native
"$py" -m integration.vllm_native_mlu.probe /vout/native/libnative_mlu.so \
    /vout/accuracy.csv /checkpoint /phase11/boundaries /p115
"$py" -m integration.vllm_native_mlu.attention_probe /vout/native/libnative_mlu.so \
    /vout/attention.csv /out /av-out
"$py" -m integration.vllm_native_mlu.vendor_baseline
```

<a id="vllm-mlu-backend-architecture"></a>

## Native MLU backend for vLLM

### Implementation exercised on card 0

The repository-owned package `integration/vllm_native_mlu` adds BF16 RMSNorm and router CustomOps and a BF16/FP16 materialized attention CustomOp on the installed Cambricon platform. Its build generates the exact-rsqrt source from the existing Phase 8.2 generator, compiles it for `mtp_592`, and links the existing C launcher into `libnative_mlu.so`. The Python adapter validates the tensor contract, allocates the output on MLU using PyTorch, enqueues onto the current MLU stream through the CNRT launcher, and synchronizes to keep input storage alive until reads finish. It supports contiguous BF16 inputs with final dimension 512 or 2048 and a same-device contiguous BF16 weight. Unsupported inputs and failed launches raise explicit errors. The hardware probes invoke vLLM CustomOp dispatch on explicitly selected MLU streams. The materialized attention op accepts contiguous head-major query, key and value buffers, position IDs, logical cache length and scale. It passes QK scores to Softmax and probabilities to AV on the same queue with device-resident handoffs. This interface accepts committed materialized K/V. It still needs the paged adapter and projection and cache updates in the model path.

### Model integration design

vLLM should continue to own request scheduling, batching, token budgets, cache block allocation and release, sampling, and serving. The vendor platform supplies the MLU device and worker. A registered project model must bind the native operators, define the checkpoint loading path and execute the scheduled token rows. The model must validate the backend and operator set before accepting requests.

| Operator group | Interface needed |
|---|---|
| Embedding, projection GEMM, LM head | Torch MLU tensor buffers, pinned weight packing layout and lifetime. The runner's packed weight registry currently belongs to its own worker and allocator. |
| RMSNorm | BF16 unfused bridge verified. The fused-residual model calling convention needs a device path with the original rounding order. |
| MLA, YaRN, QK, Softmax, AV | Project backend needs a queue-aware workspace and a per-request view of the scheduler's token positions and causal mask. Keep the validated Phase 10 layout and Phase 9.4 arithmetic. |
| KV cache | vLLM owns blocks. Phase 10 owns contiguous, materialized K/V and appends by cursor. The adapter needs a cache spec with physical layout and block size, device scatter from slot mappings, and gather by block table into contiguous scratch for the accepted kernels. Scratch use must be ordered on the queue. Validate remapping and reuse before enabling prefix sharing or preemption. |
| SwiGLU and MoE | The router projection CustomOp is verified on captured inputs. Bind the existing BF16 pointwise and expert kernels. Preserve the original FP32 router and softmax, unsorted top-six, routed reduction and shared expert addition. Group rows by expert when the selected IDs and arithmetic are preserved. Measure the synchronous device-to-host ID transfer in the standalone scheduling design. |
| Token selection | Return complete vocabulary logits from native LM head to vLLM sampling. Preserve greedy token-ID comparison against the original checkpoint. |

The vLLM model and KV adapters and the NeurX-to-vLLM linkage are pending. The standalone runner, vendor vLLM stack and repository CustomOp probe are separately reproducible paths.

[Compatibility audit](#vllm-mlu-compatibility-audit) · [Runtime architecture audit](#inference-runtime-architecture-audit)

<a id="vllm-mlu-integration-plan"></a>

## vLLM native MLU execution plan

1. Pin the vendor SDK and verify the MLU platform, worker and extension interfaces on card 0. Run the original checkpoint using the vendor stack as an environment control.
2. Build the repository's accepted kernels as callable native libraries, starting with the unfused BF16 norm. Check exact tensor shape, dtype, device, storage lifetime and CNRT queue semantics. Add native embedding, GEMM, LM head, residual and SwiGLU entry points.
3. Close FP32 router numerical parity against pinned PyTorch. Match both GEMV and SGEMM and rerun the standalone diagnostic, isolated MoE matrix, full vocabulary logs and both 32-token generations before changing the production router.
4. Bind the native operators to an explicitly registered 27-layer vLLM model. Load the original four shards and original tokenizer. Check the untied LM head, all 26 Top-6 MoE layers and Layer 0 dense path. Reject requests if a required native operator has no binding.
5. Register an MLA attention backend and compatible cache spec. Use vLLM slot mappings and block tables to scatter and gather native K/V on device. Test cache reuse, sequence termination, and mixed-length prefills and decodes with batch sizes 1, 2 and 4 before enabling optional cache-sharing or preemption features.
6. Compare checkpoint PyTorch, the standalone runner and the registered vLLM model at all layer boundaries, cache contents, logits and token IDs using the existing tolerance. Capture the first mismatch as well as bound violations.
7. Serve an OpenAI-compatible request through the native model. Verify native device dispatch for the complete request, streaming, concurrent requests and cancellation. Benchmark load time, memory, TTFT, inter-token latency, throughput, cache use and host-device transfers for cold and warm cases.

Step 1 and the RMSNorm, router and materialized attention portions of step 2 have hardware evidence. The router-only standalone build matches the checkpoint’s original failing cases and is available for the native model adapter. Full native serving and performance gates require the model and cache integration.

[Gate results](#vllm-mlu-numerical-acceptance) · [Router investigation](DEEPSEEK_V2_LITE_E2E_ACCEPTANCE.md#phase11-5-router-numerical-acceptance)

<a id="vllm-mlu-numerical-acceptance"></a>

## vLLM native MLU numerical acceptance

| Gate | Result | Hardware evidence or dependency |
|---|---|---|
| A. Platform | **PASS** | The installed vendor platform and worker discovered and used MLU590-M9 card 0. |
| B. Native operators | **BLOCKED** | BF16 RMSNorm and router passed, as did the BF16/FP16 materialized QK, Softmax and AV chain. The remaining bindings and worker integration are pending. |
| C. DeepSeek-V2-Lite model | **BLOCKED** | Vendor vLLM loaded the original checkpoint and generated tokens. The repository-native model registration and execution path are pending. |
| D. Scheduler and KV | **BLOCKED** | The native materialized MLA cache needs the block-table and slot-mapping adapter. |
| E. Numerical parity | **BLOCKED** | The native vLLM model path is required for layer-by-layer, cache and vocabulary comparison. The standalone router and complete checkpoint traces are tracked separately. |
| F. Continuous batching | **NOT RUN** | The native model and paged cache paths are required. |
| G. OpenAI-compatible serving | **NOT RUN** | Offline vendor generation is documented under the compatibility gate. |
| H. Native performance | **NOT RUN** | The full native model and scheduler path is required. |

A vLLM CustomOp invoked the Phase 8.2 BF16 kernel using a real CNRT queue on card 0. Four cases covered 27,136 values, including original checkpoint final-RMSNorm weights and two Phase 11 embedding-to-norm fixtures. The independent CPU calculation agreed with both stored checkpoint references. Maximum absolute error, relative error, ULP distance, bit mismatches, tolerance mismatches and unexpected nonfinite outputs were zero across all four cases. The threshold was `1e-5 + 1e-5 * abs(reference)`.

The router CustomOp matched all 1,216 FP32 logits bit-for-bit across the three captured Layer 2 and Layer 26 cases. The BF16 and FP16 materialized attention CustomOp passed all 32 checkpoint cases across lengths 1, 7, 32 and 128, prefill and decode, nonzero decode offsets, and two request slots on separate queues. QK scores, Softmax probabilities and low-precision AV outputs matched the fixtures bit-for-bit. Raw FP32 AV met tolerance at all 1,409,024 elements and produced zero unexpected nonfinite values. Comparisons used the Phase 9.4 reference contract.

The vendor run emitted `[8913, 13, 185, 549]` and `[317, 254, 4345, 280]` for two prompts. The native backend acceptance gate requires a complete vLLM request whose model operations execute the repository-native kernels.

[Operator accuracy CSV](../artifacts/logs/vllm_mlu_accuracy.csv) · [Hardware log](../artifacts/logs/vllm_mlu_hardware.log) · [Router acceptance](DEEPSEEK_V2_LITE_E2E_ACCEPTANCE.md#phase11-5-router-numerical-acceptance)

<a id="phase12-vllm-api-audit"></a>

## Phase 12 vendor vLLM API audit

The pinned SDK image is `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`. It contains vLLM 0.25.1, distribution 0.25.1+empty, vendor extension 0.25.1+mlu0.16.0.pt212, Python 3.12.3 and torch_mlu 1.34.1+torch2.12.1. CNCC 5.7.6, CNRT 7.7.3, CNNL 2.2.18 and driver 6.5.52 were verified on the target. The vendor source snapshot has no Git metadata. The image digest and installed-source hashes in the Phase 11.5 vLLM report identify the inspected build. [Version and source evidence](#vllm-mlu-compatibility-audit)

| Interface | Installed implementation |
|---|---|
| Platform | The vendor platform plugin selects `MLUPlatform`, its attention backend, custom KV specs and an `MLUWorker` when the worker option is automatic. |
| Worker | `vllm_mlu.v1.worker.gpu_worker.MLUWorker` and `MLUModelRunner` provide model loading and scheduled `execute_model`. |
| Model configuration | The vendor general plugin registers DeepSeek models. `ModelRegistry.register_model` is available for an explicitly selected adapter. The original checkpoint architecture string is `DeepseekV2ForCausalLM`. |
| Operator dispatch | `CustomOp.register_oot` is present. The vendor also patches native and OOT norm dispatch. Prior isolated probes use distinct CustomOp names to avoid collisions. |
| Cache | `KVCacheSpec` and `MLAAttentionSpec` describe engine allocations. `TMOFlashMLAMetadataBuilder` forwards `block_table_tensor` and `slot_mapping`, then splits prefill and decode. The vendor block-table patch uses int32 slot indices. |
| Device | Import torch_mlu before vLLM in direct probes to register the MLU dispatch key. Use the active torch_mlu stream's CNRT queue handle. |

The vendor MLA cache stores the vendor layout. The project adapter needs its own cache specification sized for materialized per-head 192-word K and 128-word V. The scheduler must retain allocation, ownership and reclamation. A device adapter can translate slot indices on writes and block-table indices on reads into the existing head-major native attention layout. Once the cache specification is bound to a project model, verify all 27 layers with scheduler-provided metadata before testing token parity, batching, serving or speed.

The standalone runner and the vendor engine have no verified execution link to the separate NeurX checkout. [Runtime architecture audit](#inference-runtime-architecture-audit)

<a id="phase12-3-vendor-api-audit"></a>

## Phase 12.3 installed Vendor vLLM APIs

The audited SDK has vLLM 0.25.1 and vllm_mlu 0.25.1+mlu0.16.0.pt212. Installed core source is at `/torch/venv3/pytorch_infer/lib/python3.12/site-packages/vllm`. The vendor snapshot is at `/workspace/vllm-v0.25.1+mlu0.16.0.pt212/vllm_mlu`. [Version and source evidence](#phase12-vllm-api-audit)

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
