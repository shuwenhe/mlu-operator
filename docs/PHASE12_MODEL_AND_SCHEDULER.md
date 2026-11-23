# Phase 12 model, scheduler and cache binding

- [Phase 12 DeepSeek-V2-Lite model binding](#phase12-deepseek-native-model-binding)
- [Phase 12 paged KV cache](#phase12-paged-kv-cache-acceptance)
- [Native DeepSeek-V2-Lite model binding](#phase12-3-native-model-binding)
- [Scheduler metadata](#phase12-3-scheduler-metadata)
- [DeepSeek-V2-Lite checkpoint architecture audit](#phase12-4-deepseek-architecture-audit)
- [Vendor MLA cache layout mapping](#phase12-4-mla-cache-layout-mapping)
- [Native DeepSeek-V2-Lite model binding](#phase12-4-native-model-binding)
- [DeepSeek-V2-Lite per-layer cache specification](#phase12-4-per-layer-cache-spec)
- [Phase 12.5.1 model registration](#phase12-5-1-model-registration)
- [Phase 12.5.1 native forward contract](#phase12-5-1-native-forward-contract)
- [Phase 12.5.1 scheduler metadata contract](#phase12-5-1-scheduler-metadata)
- [Phase 12.5.1 worker cache binding](#phase12-5-1-worker-cache-binding)

<a id="phase12-deepseek-native-model-binding"></a>

## Phase 12 DeepSeek-V2-Lite model binding

**BLOCKED.** Vendor vLLM loaded the checkpoint on card 0 in the previously verified vendor run. The project exposes native RMSNorm, router projection and a materialized QK–Softmax–AV chain as isolated vLLM CustomOps. The scheduled vendor model currently selects its own implementation. The separate NeurX checkout has no verified model, cache or operator binding to the vendor worker.

| Boundary | Evidence and outstanding binding |
|---|---|
| Embedding, GEMM, LM head | Standalone accepted implementations. The vLLM model needs checkpoint loading and packed-weight residency wired to framework-owned device storage. |
| RMSNorm | Isolated BF16 probe passed. The model must preserve the original residual rounding and explicitly call the accepted norm. |
| MLA, RoPE/YaRN, QK, Softmax, AV | The materialized chain passed in isolation. The model needs projection, position handling, paged append, and attention metadata wired to each layer. |
| Dense SwiGLU, routed and shared experts, aggregation | Standalone accepted implementations. The scheduled model must call the native expert and pointwise paths. |
| Router and Top-6 | The BF16-to-FP32 corrected projection passed the isolated vLLM probe. Preserve its FP32 order and the original six-expert selection and combine. |
| Sampling | The registered model must return the native LM head's vocabulary logits to vLLM sampling. |

Use a distinct model-registry entry, with explicit native operator selection and a matching KV cache specification. An actual worker request, with per-operator dispatch and CNRT execution recorded, is required to accept any model-path binding. [Vendor API audit](VLLM_MLU_BACKEND_ARCHITECTURE.md#phase12-vllm-api-audit)

<a id="phase12-paged-kv-cache-acceptance"></a>

## Phase 12 paged KV cache

The native adapter consumes scheduler-format int32 physical slot mappings on writes and physical block IDs on reads. Its page buffers are token-major `[physical slots, 16, width]`, with 192 key and 128 value words per head. Gather produces the head-major logical prefix used by the accepted native attention chain. The BANG C kernel copies low-precision bits without arithmetic, and the wrapper launches it on the current MLU queue. It synchronizes before returning so buffers remain live through the transfer.

The scheduler is responsible for unique live write slots, valid block ownership, committed lengths and the ordering of reclamation. Padding writes with negative slots are skipped. Out-of-range indices are guarded in the kernel. The current adapter supports logical lengths through 131, the existing standalone position-table limit.

**Device probe PASS.** BF16 and FP16 passed bitwise readback for noncontiguous physical blocks, a one-token append, two independent logical prefixes, ignored padding and explicit block reassignment on MLU590-M9 card 0.

**Scheduler integration BLOCKED.** The standalone hardware harness supplies metadata and simulates reassignment. The vendor model has yet to allocate this cache format and connect its actual block tables and slot mappings at each layer. Preemption, recomputation and framework-driven reclamation remain untested.

[Device evidence and reproduction commands](../artifacts/logs/phase12_kv_cache_hardware.log)

<a id="phase12-3-native-model-binding"></a>

## Native DeepSeek-V2-Lite model binding

**BLOCKED.** The installed vendor registry and model loader are mapped in the API audit. The existing vendor DeepSeek model invokes a vendor backend with a compressed MLA cache layout. The accepted native router, QK, Softmax, AV and Phase 10 MLA sources remain separate from this model path.

The new `ScheduledAttention` adapter accepts projected, rotary-applied Q/K/V on device. A native model class must supply projections and positions, bind an engine-owned materialized cache specification for each of the 27 layers, and call the adapter from its attention forward method. It must also bind Layer 0 dense FFN, the 26 MoE layers using the accepted FP32 router, and native embedding, residuals, normalization and LM head. This run did not register a substitute model or override the vendor model class. [Installed APIs](VLLM_MLU_BACKEND_ARCHITECTURE.md#phase12-3-vendor-api-audit)

The vendor model's `forward_mla_attn` at `model_executor/models/deepseek_v2.py:1396` takes positions, hidden states, attention metadata and cache tensors. It slices decode tokens at the front using `num_decode_tokens`, runs separate prefill and decode functions, and restores that order when joining outputs. This identifies the boundary for native projections and cache dispatch. A future binding must preserve the same scheduler row ordering and return projected attention outputs to the decoder residual path.

<a id="phase12-3-scheduler-metadata"></a>

## Scheduler metadata

The new adapter consumes vendor-style `query_start_loc`, `seq_lens`, `block_table_tensor` and `slot_mapping`, aligned with the worker's active request IDs and scheduled positions. It validates all active spans before writing any cache data. It rejects duplicate request IDs, repeated or cross-request page ownership, out-of-range physical pages and slots that disagree with the block table and position. Its current contract requires exclusive pages and contiguous scheduled positions at the end of each committed sequence. The validator reads small metadata arrays on the host. Projected activations and KV cache remain device-resident.

The vendor control instruments `SchedulerWithProfiler.schedule` and `KVCacheManager.free` in the test process and records request IDs, token counts and physical blocks. The native adapter probe consumes deterministic fixture metadata in a separate process. Binding actual worker metadata, including its request ordering, to the native model is **BLOCKED**. [Vendor hardware log](../artifacts/logs/phase12_3_scheduler_hardware.log)

The final vendor control recorded a standalone five-token prompt and a batch containing five- and 65-token prompts. All three requests generated four tokens and logged a cache free. The 65-token request received blocks 2–6. Block 1 was reclaimed from the first request and reassigned to the next short request. The trace records three decode scheduling steps for each request. A retry that imported the vendor scheduler before the engine raised a vendor `check_stop` signature error. The final control attaches the trace after vendor engine initialization.

Run the vendor control in the pinned SDK with the writable cache environment from the compatibility audit:

```bash
/torch/venv3/pytorch_infer/bin/python3 -m integration.vllm_native_mlu.phase12_3.vendor_trace
```

<a id="phase12-4-deepseek-architecture-audit"></a>

## DeepSeek-V2-Lite checkpoint architecture audit

**PASS.** The header audit checked all 5,291 original BF16 tensor names, shapes, shard locations and byte extents across four safetensors shards. The checkpoint has 27 layers, hidden width 2048, an untied 102,400-token embedding and LM head, no attention bias, and RMSNorm epsilon 1e-6.

| Weight under the model or relative decoder prefix | Shape |
|---|---|
| `model.embed_tokens.weight`, `lm_head.weight` | [102400, 2048] each |
| `model.norm.weight`, `input_layernorm.weight`, `post_attention_layernorm.weight` | [2048] each |
| `self_attn.q_proj.weight` | [3072, 2048], direct query projection |
| `self_attn.kv_a_proj_with_mqa.weight` | [576, 2048], 512 latent plus 64 rotary |
| `self_attn.kv_a_layernorm.weight` | [512] |
| `self_attn.kv_b_proj.weight` | [4096, 512], expands 16 heads of 128 NoPE key and 128 value components |
| `self_attn.o_proj.weight` | [2048, 2048] |
| Layer 0 `mlp.{gate,up,down}_proj.weight` | [10944, 2048], [10944, 2048], [2048, 10944] |
| Layers 1–26 `mlp.gate.weight` | [64, 2048], greedy Top-6 with softmax |
| Layers 1–26 `mlp.experts.<expert>.{gate,up,down}_proj.weight` | [1408, 2048], [1408, 2048], [2048, 1408], for expert IDs 0–63 |
| Layers 1–26 `mlp.shared_experts.{gate,up,down}_proj.weight` | [2816, 2048], [2816, 2048], [2048, 2816] |

Each layer has 16 attention heads with a 192-wide query/key (128 NoPE plus 64 RoPE) and 128-wide value. The checkpoint's `q_lora_rank` is null and it has no query latent or expansion weight. YaRN uses factor 40, original context 4096, base 10000 and mscale values 0.707.

The vendor loader resolves checkpoint architecture `DeepseekV2ForCausalLM` to its MLU DeepSeek model. The installed plugin API is `ModelRegistry.register_model(model_arch, model_cls)`. The vendor model uses distinct prefill and decode attention call sites with common scheduler metadata. [Installed APIs](VLLM_MLU_BACKEND_ARCHITECTURE.md#phase12-3-vendor-api-audit)

<a id="phase12-4-mla-cache-layout-mapping"></a>

## Vendor MLA cache layout mapping

**PASS for the layout audit.** An actual 65-token vendor request traced prefill and decode attention for each of the 27 layers on MLU590-M9 card 0. Both paths used BF16 cache shape `[1,28014,1,16,576]` with element strides `[258177024,9216,9216,576,1]` and an empty scale tensor. The logical axes are one cache plane, physical block, compressed KV head, offset within a block and feature. Each token stores 512 normalized latent components and 64 positionally encoded key components. One block holds 16 tokens and occupies 18,432 bytes per layer. For physical block `b` and in-block offset `j`, the feature row starts at `2 * (b * 9216 + j * 576)` bytes.

Vendor prefill passed materialized token-major Q and K `[65,16,192]` and V `[65,16,128]` while retaining the compressed persistent cache. Vendor decode passed Q `[1,16,576]`, K `[1,1,576]` and a 512-wide latent V input. The decode MLA cache write uses scheduler slots with the vendor paged-cache reshape path. Its read uses the block table and committed lengths. During tracing the request used blocks 1–5, with the remaining block-table entries padded.

The frozen native chain consumes materialized, head-major Q/K `[16,tokens,192]` and V `[16,tokens,128]`. The native page adapter uses token-major K and V planes before gathering into that layout.

| Option | Storage and execution tradeoff |
|---|---|
| A. Consume vendor compressed bytes directly | Retains 1,152 bytes per token per layer. Requires separate compressed-attention arithmetic, projected queries, latent reduction and output transformation. The frozen materialized kernels cannot read these bytes directly. |
| B. Expand on device | Retains compact persistent storage and can feed frozen kernels after KV-B projection and layout assembly. Gather and expansion of historical keys and values on every decode add compute and scratch. Correctness depends on preserving each rounding boundary. A persistent duplicate would increase memory and is outside this design. |
| C. Scheduler-owned materialized pages | Uses 10,240 bytes per token per layer, 8.89× the vendor logical cache size. It preserves the accepted native arithmetic and needs only per-request gather scratch. This is the tested component contract. The vendor model must select its spec and provide engine storage and metadata before deployment. |

A/B latency and the end-to-end cost of C were not measured. The cache layout cannot be converted with a view because the latent needs the original learned expansion. [Vendor hardware trace](../artifacts/logs/phase12_4_attention_hardware.log)

<a id="phase12-4-native-model-binding"></a>

## Native DeepSeek-V2-Lite model binding

**BLOCKED.** The vendor control ran the real 27-layer checkpoint and returned four tokens. The repository's materialized cache specification and worker-compatible reshape function are implemented and component-tested. The model-registry entry, worker allocation binding and all-layer native decoder invocation remain pending.

The new projection C ABI reuses the original device RHS packer and checkpoint BF16 GEMM candidate. It accepts the original BF16 weight in [N,K] order on MLU, transposes it to contiguous [K,N] for the device packer, and passes the resulting vendor-independent packed handle to the frozen FP32-accumulating projection. The adapter synchronizes the current queue and exposes explicit close. The model binding must retain these handles according to layer lifetime, use the separate Phase 11.5 router and chunk LM head columns as in the original runner. `q_lora_rank=null` requires only the direct Q projection.

The vendor worker calls a backend-specific reshape hook when present. To use the tested native cache format, the native model needs a model registry entry, one cache spec per layer and a backend exposing that reshape hook. Each layer must bind projected Q/K/V and scheduler metadata to the Phase 12.3 adapter, then use native attention output projection and the original residual, dense or MoE paths. [Cache mapping](#phase12-4-mla-cache-layout-mapping)

### Live projection control

`LLM.apply_model` installed an explicit diagnostic replacement for Layer 0's attention output projection after the vendor worker loaded the checkpoint. The exact accepted BF16 projection and Phase 10 cast ran on a nondefault CNRT queue for a batch of four prompts with lengths 1, 7, 17 and 65. The five observed calls processed 89, 1, 4, 4 and 4 rows. The call with 89 rows processed the combined prefill. Later calls included incremental decode. The 16 greedy IDs matched the unmodified vendor run. The test restored the vendor projection and released its native packed handle. The control used vendor attention, KV pages and other layers, and does not satisfy the native model registration or all-layer gates.

### Reproduction

Use the image, cache environment and device mounts from the [vendor API audit](VLLM_MLU_BACKEND_ARCHITECTURE.md#vllm-mlu-compatibility-audit). From the repository root inside the image:

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

<a id="phase12-4-per-layer-cache-spec"></a>

## DeepSeek-V2-Lite per-layer cache specification

The checkpoint audit and vendor runtime trace verify identical attention dimensions and compressed cache layout on layers 0–26. Layer 0 has a dense FFN, while layers 1–26 have MoE FFNs. That difference does not alter the attention/cache contract.

| Decoder layers | Type | Native specification |
|---|---|---|
| 0 | Dense | BF16 FullAttentionSpec, 16 heads, K width 192, V width 128 |
| 1–26 inclusive | MoE | Same independent FullAttentionSpec for each of the 26 layers |

The implementation creates a separate spec entry for each layer and accepts a raw byte allocation in the vendor worker's `reshape_kv_cache(raw, spec, num_blocks, kernel_block_size, cache_dtype)` format. All K slots precede all V slots in a single allocation. Their tensor views have element strides `[3072,192,1]` and `[2048,128,1]`. With block size (B) and (N) blocks, both views have (N B) token slots and the page byte size is (10240 B). The physical token slot is `block_table[position // B] * B + position % B`. Cache writes use the scheduler's int32 slot mapping. Reads use block IDs and valid logical length. In a real binding the worker owns device allocation and the scheduler owns page IDs and lifetime. The views own no duplicate backing allocation.

**Component PASS on card 0.** The probe merged the 27 installed FullAttentionSpec entries and used their computed 163,840-byte page size at block size 16. For each layer, it verified zero-copy view pointers and ran the original native paged write/read across noncontiguous blocks 2 and 0. Scheduler allocation and the native backend reshape callback remain unbound to the actual model. [Component log](../artifacts/logs/phase12_4_model_binding_hardware.log)

<a id="phase12-5-1-model-registration"></a>

## Phase 12.5.1 model registration

### Scope

- Registered a distinct architecture name: `NeurXDeepSeekV2ForCausalLM`.
- Kept the original Vendor architecture (`DeepseekV2ForCausalLM`) untouched.
- Bound model registry entry to `integration.vllm_native_mlu.phase12_5.model:NeurXDeepSeekV2ForCausalLM`.

### Code changes

- `integration/vllm_native_mlu/phase12_5/run.py`
  - Updated `ARCHITECTURE` to `NeurXDeepSeekV2ForCausalLM`.
  - Updated `ModelRegistry.register_model` target class path.
- `integration/vllm_native_mlu/phase12_5/model.py`
  - Added canonical class `NeurXDeepSeekV2ForCausalLM`.
  - Kept compatibility alias `NativeDeepseekV2Lite = NeurXDeepSeekV2ForCausalLM`.
  - Added structured runtime instrumentation fields:
    - `architecture_name`
    - `model_instance_id`
    - `static_context_layer_count`
    - `native_cache_layer_count`

### Contract preservation

- Preserves 27-layer layout and Layer 0 dense + Layers 1–26 MoE path.
- Preserves `q_lora_rank=null` expectation and BF16 single-card constraints.
- Reuses accepted native projection C ABI and Phase 11.5 router path.
- Preserves chunked LM head execution.

### Validation

- Python syntax validation passed for updated modules via `python3 -m py_compile`.
- Vendor-container runtime verification completed in Phase 12.5.2 with:
  - Pinned Python `/torch/venv3/pytorch_infer/bin/python3`
  - Architecture resolution `NeurXDeepSeekV2ForCausalLM`
  - Successful generation requests through registered native architecture
  - Evidence: `artifacts/logs/phase12_5_2_environment.log`, `artifacts/logs/phase12_5_2_registration_hardware.log`

<a id="phase12-5-1-native-forward-contract"></a>

## Phase 12.5.1 native forward contract

### Implemented forward contract

- Native forward receives:
  - input token IDs
  - positions
  - forward-context attention metadata
  - worker-owned K/V cache views
  - active CNRT stream handle
- Native forward explicitly rejects unsupported `intermediate_tensors` and `inputs_embeds`.

### Instrumentation additions

- `integration/vllm_native_mlu/phase12_5/model.py` now emits structured trace events (when `NATIVE_DEEPSEEK_TRACE=1`) including:
  - `architecture_name`
  - `model_instance_id`
  - `layer_id`
  - `layer_class`
  - `cache_spec`
  - `worker_allocation_id`
  - `cache_base_address`
  - `cache_allocation_bytes`
  - `k_view_address`
  - `v_view_address`
  - `reshape_hook_invoked`
  - `zero_copy_verified`
  - `scheduler_metadata_received`
  - `block_table`
  - `slot_mapping`
  - `request_id`
  - `cnrt_queue_id`
  - `native_cache_layer_count`
  - `fallback_count`

### Phase boundary

Phase 12.5.2 executed real Vendor requests through this path and verified:

- Native model instantiation and forward invocation.
- 27-layer cache discovery and reshape-hook usage.
- Scheduler metadata arrival in native forward.
- Incremental decode scheduling after prefill.

Full numerical fixture-regression acceptance remains tracked separately from runtime integration.

<a id="phase12-5-1-scheduler-metadata"></a>

## Phase 12.5.1 scheduler metadata contract

### Scope

The native path consumes scheduler-produced metadata from the active forward context and validates it before cache writes.

### Metadata path

- `integration/vllm_native_mlu/phase12_5/model.py`
  - Reads `get_forward_context().attn_metadata`.
  - Uses per-layer metadata entries keyed by registered static layer names.
  - Passes scheduler metadata into request-span validation and cache read/write flow.
- `integration/vllm_native_mlu/phase12_5/cache.py`
  - `NativeMetadataBuilder.build` now logs metadata build events and enforces exclusive-page assumption.
- `integration/vllm_native_mlu/phase12_3/scheduled_attention.py`
  - Existing validation remains the source of truth for:
    - query spans
    - sequence lengths
    - block tables
    - slot mappings
    - block ownership and bounds

### Preserved ownership boundaries

- No scheduler-state reconstruction in native model.
- No synthetic block tables.
- No fixture metadata substitution for production path.
- Request isolation checks remain explicit in span validation.

### Validation

- Static validation path is exercised by syntax check.
- Runtime scheduler metadata dispatch was verified in Phase 12.5.2 on real Vendor requests.
- Evidence includes:
  - Vendor scheduler events with prefill/decode scheduling and finished-request transitions.
  - Native per-layer metadata logs containing block tables and slot mappings.
  - `artifacts/logs/phase12_5_2_scheduler_metadata.log`

<a id="phase12-5-1-worker-cache-binding"></a>

## Phase 12.5.1 worker cache binding

### Scope

This phase binds Vendor worker-owned allocations to native materialized K/V views through the backend reshape hook, without replacing scheduler ownership or allocation lifetime.

### Code changes

- `integration/vllm_native_mlu/phase12_5/cache.py`
  - `NativeBackend.reshape_kv_cache` now enforces:
    - BF16 alignment (`raw.data_ptr() % 2 == 0`).
    - Spec page-size compatibility against `FullAttentionSpec`.
    - Shared underlying storage for raw/K/V views (no duplicate allocation).
    - Expected K/V storage offsets for accepted layout.
  - Added runtime instrumentation:
    - `reshape_hook_invoked`
    - `cache_allocation_bytes`
    - `cache_base_address`
    - `k_view_address`
    - `v_view_address`
    - `zero_copy_verified`
    - `cache_spec`
- `integration/vllm_native_mlu/phase12_5/model.py`
  - Emits per-layer runtime binding diagnostics while executing requests:
    - Layer class/index and cache spec.
    - Worker allocation IDs and K/V addresses.
    - Scheduler block table and slot mapping slices.

### Ownership guarantees

- No replacement cache storage allocation in reshape hook.
- No ownership transfer from worker allocation to native model.
- K/V views remain zero-copy aliases over worker-provided allocation.

### Layer coverage

- Native model instantiates 27 `NativeCacheLayer` entries.
- Each layer advertises cache spec:
  - BF16, 16 heads, K width 192, V width 128.

### Validation

- Python syntax validation passed for updated modules.
- Live worker allocation and reshape-hook execution were verified in Phase 12.5.2.
- Evidence includes:
  - 27-layer static-context discovery.
  - Worker-owned allocations reshaped for all layers.
  - Zero-copy verification (`zero_copy_verified=true`) for worker K/V views.
  - `artifacts/logs/phase12_5_2_worker_cache_hardware.log`
