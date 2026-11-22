# Final DeepSeek inference architecture

`make deepseek` selects `NeurXDeepSeekV2ForCausalLMStageC3` and `fused,batched`. The inference path uses native MLU operators and the vendor CNCL device group on eight MLU590-M9 devices. The Stage D launcher adapts the command line to the frozen synchronous vendor engine. All model code used by the worker remains in the frozen Stage C trees.

| Boundary | Accepted Stage C.3 configuration |
|---|---|
| Engine and scheduler | Vendor multiprocess workers, BF16, TP8, pipeline and data parallelism of one, eager execution, 128-token context, 16-token cache pages |
| Query and expanded KV | Two contiguous attention heads per rank, with head-local Q, KV-B and MLA attention |
| Latent KV | Replicated 512-dimensional latent and shared 64-dimensional rotary key, with full latent RMSNorm |
| KV cache | Per-rank BF16 key and value views of vendor-allocated storage, with two heads and exclusive pages |
| MLA output | Gather BF16 attention channels, execute local FP32 output stripes, gather stripes, then round to BF16 and add the residual |
| MoE | Replicated top-six routing, eight whole routed experts owned per rank, route digest agreement, packed fused gate/up BF16 GEMM, per-expert prefill row batching, sparse BF16 reduction and original-order combine |
| Lifecycle | Per-rank device, model and decoder checks before generation, finite-logit enforcement in the native worker, and verified status from each worker at shutdown |

The host launcher checks the frozen image and eight idle cards. The in-container gate requires eight MLU devices and confirms the exact model and decoder classes, CNCL backend, rank/device mapping, output-axis layers and accepted expert flags on every worker. The checkpoint and repository are mounted read-only. Evidence is written to a fresh host run directory. The launcher exits before generation when any worker falls outside this configuration.

Stage C.2 keeps output-axis MLA with the original expert decoder. Stage C.1 retains ordered input-axis output projection. Their sources and evidence remain independently verifiable.

[Stage C.1 tensor partition audit](../artifacts/logs/phase15/stage_c1/stage_c1_architecture_audit.md) · [Stage C.2 communication changes](../artifacts/logs/phase15/stage_c2/final_report.md) · [Stage C.3 MoE audit](../artifacts/logs/phase15/stage_c3/architecture_audit.md)
