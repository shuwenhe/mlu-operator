# DeepSeek MLU native runtime design

## Ownership and scheduling

Use one explicit worker context per assigned physical MLU and record its PCI identity and CNRT ordinal. Each context owns model metadata, a map of immutable packed weights, persistent layer buffers and queue handles. The registry key is the actual checkpoint revision, tensor identity, shard/offset, layout, dtype and device. Invalidate it when loading a different checkpoint. No process-global cache is needed.

During preparation, transpose checkpoint row-major `[N,K]` to logical `[K,N]` and pack once. Host pack can publish after its synchronous upload. Device pack remains asynchronous, so publish after its queue completes or after explicit event dependency. Record the logical and padded packed sizes. Queue-owning execution code selects the weight's device and retains both request scratch and weights until all dependent work completes. Concurrent read-only queue use is allowed after publication. Never free a referenced handle. Require successful drain or explicit runtime recovery before reclaiming buffers on error. Propagate raw CNRT/CNNL status with device, op, tensor, layer and request context.

Preallocate typed workspace per queue or request slot. Reuse descriptors and scratch only after the owning completion boundary. Keep FP32 GEMM output staging separate from BF16 activation buffers. Reserve cache blocks independently from short-lived activation scratch. Track persistent bytes, cache bytes, peak live workspace and allocator reserve separately.

## Dispatch and increment

1. Start with one device and explicit opt-in native projections and RMSNorm. Verify shape, stride, dtype and ownership before dispatch. Preserve CNNL as an independent selectable path. Do not silently substitute it for a failed native call.
2. The implemented KV subgraph passes persistent weights, cast and normalization on one queue. To extend it, carry the shared 64-wide rotary key into the validated standalone rotation interface, then publish cache entries after writes are ordered. Embed and resolve fixed input token IDs, validate layer intermediates, and test one prefill followed by token-at-a-time decode. Integrate MLA only after low-precision parity and cache append/rollback checks. Keep FP32 router, optional collectives and unsupported operators on explicit approved backends.
3. Layer plans own transient views, normalized latent layout, backend choice and output semantics. Integrate the remaining output projection, residuals, expert dispatch, gated MLP, final norm, logits and sampling before measuring end-to-end output. A vLLM adapter would bind these operations to existing device-resident tensors and the current queue in a separate development checkout. The production deployment is unchanged.

## Acceptance protocol

Compare identical BF16 checkpoint weights, token IDs, batch sizes, prompt and generation lengths, sampling policy, runtime versions and card placement. Collect setup/load time separately. Report TTFT to first token and TPOT after the first token, plus prefill tokens/s, decode tokens/s, aggregate tokens/s, latency distributions and peak device allocation. Establish single-card token/logit parity before multi-card comparison. Independent throughput and model sharding require separate measurements. None of these inference-level metrics was collected in Phase 5. [Integration status](DEEPSEEK_V2_LITE_MLU_INTEGRATION.md)
