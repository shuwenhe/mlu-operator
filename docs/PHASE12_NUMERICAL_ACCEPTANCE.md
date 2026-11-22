# Phase 12 numerical acceptance

| Gate | Result | Evidence or dependency |
|---|---|---|
| A. Vendor vLLM baseline | **PASS** | Phase 11.5 vendor engine loaded the checkpoint and generated tokens on card 0. |
| B. Native RMSNorm and router model bindings | **BLOCKED** | Isolated CustomOps passed. A scheduled model invocation is pending. |
| C. Native attention model bindings | **BLOCKED** | The isolated materialized chain passed. Project MLA and cache dispatch from the scheduled model are pending. |
| D. Native MoE model bindings | **BLOCKED** | Standalone acceptance passed. Native expert calls from the vendor worker are pending. |
| E. Full-model binding | **BLOCKED** | A registered native decoder model and compatible cache specification are pending. |
| F. Native paged KV | **BLOCKED** | Engine ownership and dispatch of the adapter remain pending. |
| G. Full-model numerical parity | **BLOCKED** | CPU PyTorch and standalone references exist. The third execution path is pending. |
| H. Continuous batching | **NOT RUN** | Requires scheduled native model requests. |
| I. OpenAI-compatible serving | **NOT RUN** | Requires the native model and cache integration. |
| J. Performance benchmark | **NOT RUN** | The native vLLM path is required for the three-way benchmark. |

The Phase 11.5 standalone router build remains the regression baseline. The operator CSV carries forward the previously measured checkpoint CustomOp comparisons. The tolerance remains `1e-5 + 1e-5 * abs(reference)`. Full-model comparison will require traced scheduled prefill and decode on the native vLLM model.

[Operator accuracy](../artifacts/logs/phase12_operator_accuracy.csv) · [Router baseline](PHASE11_5_ROUTER_NUMERICAL_ACCEPTANCE.md)
