# Native MLU backend for vLLM

## Implementation exercised on card 0

The repository-owned package `integration/vllm_native_mlu` adds BF16 RMSNorm and router CustomOps and a BF16/FP16 materialized attention CustomOp on the installed Cambricon platform. Its build generates the exact-rsqrt source from the existing Phase 8.2 generator, compiles it for `mtp_592`, and links the existing C launcher into `libnative_mlu.so`. The Python adapter validates the tensor contract, allocates the output on MLU using PyTorch, enqueues onto the current MLU stream through the CNRT launcher, and synchronizes to keep input storage alive until reads finish. It supports contiguous BF16 inputs with final dimension 512 or 2048 and a same-device contiguous BF16 weight. Unsupported inputs and failed launches raise explicit errors. The hardware probes invoke vLLM CustomOp dispatch on explicitly selected MLU streams. The materialized attention op accepts contiguous head-major query, key and value buffers, position IDs, logical cache length and scale. It passes QK scores to Softmax and probabilities to AV on the same queue with device-resident handoffs. This interface accepts committed materialized K/V. It still needs the paged adapter and projection and cache updates in the model path.

## Model integration design

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

[Compatibility audit](VLLM_MLU_COMPATIBILITY_AUDIT.md) · [Runtime architecture audit](INFERENCE_RUNTIME_ARCHITECTURE_AUDIT.md)
