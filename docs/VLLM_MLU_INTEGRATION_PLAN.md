# vLLM native MLU execution plan

1. Pin the vendor SDK and verify the MLU platform, worker and extension interfaces on card 0. Run the original checkpoint using the vendor stack as an environment control.
2. Build the repository's accepted kernels as callable native libraries, starting with the unfused BF16 norm. Check exact tensor shape, dtype, device, storage lifetime and CNRT queue semantics. Add native embedding, GEMM, LM head, residual and SwiGLU entry points.
3. Close FP32 router numerical parity against pinned PyTorch. Match both GEMV and SGEMM and rerun the standalone diagnostic, isolated MoE matrix, full vocabulary logs and both 32-token generations before changing the production router.
4. Bind the native operators to an explicitly registered 27-layer vLLM model. Load the original four shards and original tokenizer. Check the untied LM head, all 26 Top-6 MoE layers and Layer 0 dense path. Reject requests if a required native operator has no binding.
5. Register an MLA attention backend and compatible cache spec. Use vLLM slot mappings and block tables to scatter and gather native K/V on device. Test cache reuse, sequence termination, and mixed-length prefills and decodes with batch sizes 1, 2 and 4 before enabling optional cache-sharing or preemption features.
6. Compare checkpoint PyTorch, the standalone runner and the registered vLLM model at all layer boundaries, cache contents, logits and token IDs using the existing tolerance. Capture the first mismatch as well as bound violations.
7. Serve an OpenAI-compatible request through the native model. Verify native device dispatch for the complete request, streaming, concurrent requests and cancellation. Benchmark load time, memory, TTFT, inter-token latency, throughput, cache use and host-device transfers for cold and warm cases.

Step 1 and the RMSNorm, router and materialized attention portions of step 2 have hardware evidence. The router-only standalone build matches the checkpoint’s original failing cases and is available for the native model adapter. Full native serving and performance gates require the model and cache integration.

[Gate results](VLLM_MLU_NUMERICAL_ACCEPTANCE.md) · [Router investigation](PHASE11_5_ROUTER_NUMERICAL_ACCEPTANCE.md)
