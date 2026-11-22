# Phase 12 DeepSeek-V2-Lite model binding

**BLOCKED.** Vendor vLLM loaded the checkpoint on card 0 in the previously verified vendor run. The project exposes native RMSNorm, router projection and a materialized QK–Softmax–AV chain as isolated vLLM CustomOps. The scheduled vendor model currently selects its own implementation. The separate NeurX checkout has no verified model, cache or operator binding to the vendor worker.

| Boundary | Evidence and outstanding binding |
|---|---|
| Embedding, GEMM, LM head | Standalone accepted implementations. The vLLM model needs checkpoint loading and packed-weight residency wired to framework-owned device storage. |
| RMSNorm | Isolated BF16 probe passed. The model must preserve the original residual rounding and explicitly call the accepted norm. |
| MLA, RoPE/YaRN, QK, Softmax, AV | The materialized chain passed in isolation. The model needs projection, position handling, paged append, and attention metadata wired to each layer. |
| Dense SwiGLU, routed and shared experts, aggregation | Standalone accepted implementations. The scheduled model must call the native expert and pointwise paths. |
| Router and Top-6 | The BF16-to-FP32 corrected projection passed the isolated vLLM probe. Preserve its FP32 order and the original six-expert selection and combine. |
| Sampling | The registered model must return the native LM head's vocabulary logits to vLLM sampling. |

Use a distinct model-registry entry, with explicit native operator selection and a matching KV cache specification. An actual worker request, with per-operator dispatch and CNRT execution recorded, is required to accept any model-path binding. [Vendor API audit](PHASE12_VLLM_API_AUDIT.md)
