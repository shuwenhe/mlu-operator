# vLLM native MLU numerical acceptance

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

[Operator accuracy CSV](../artifacts/logs/vllm_mlu_accuracy.csv) · [Hardware log](../artifacts/logs/vllm_mlu_hardware.log) · [Router acceptance](PHASE11_5_ROUTER_NUMERICAL_ACCEPTANCE.md)
