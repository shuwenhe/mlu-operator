# Inference runtime architecture audit

The accepted Phase 11 execution path is a **standalone Native Runner**. [model_run.cpp](../runtime/native/phase11/model_run.cpp) owns the 27-layer loop, checkpoint shard maps, packed-weight leases, per-layer Phase 10 cache state, native LM head and argmax. The command-line process creates its own native worker and CNRT queue. [build.sh](../runtime/native/phase11/build.sh) links repository-native objects and CNRT. The model process has no vLLM/NeurX runtime binding. The Python checkpoint and tokenizer programs are separate tools.

The separate NeurX checkout under `/app/shuwen/.shuwen/neurx` contains experimental native MLU/GEMM source. The standalone link and execution evidence establish repository-native/CNRT execution only. A NeurX execution claim would require a linked or loaded component and a runtime trace proving that it handled the request.

The pinned SDK's vendor vLLM platform used its own engine, MLU worker and DeepSeek model to complete an offline `LLM.generate` request. The repository's vLLM CustomOp probe exercised RMSNorm, router projection and the materialized QK–Softmax–AV chain independently of the vendor model. Full integration requires a registered model calling the repository-native backend and consuming vLLM's block allocation and scheduling metadata.

[Standalone hardware evidence](../artifacts/logs/phase11_hardware.log) · [vLLM hardware evidence](../artifacts/logs/vllm_mlu_hardware.log)
