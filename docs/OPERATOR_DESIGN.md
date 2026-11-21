# Operator design

P0 uses CNNL from the verified Ubuntu 24.04 package interface. Keep the host and workload configuration unchanged. First compile CPU references and native objects. After an exclusive device reservation and explicit approval, link in a compatible SDK environment and validate CNRT before executing operators.

GEMM accepts packed row-major FP16/BF16 A[M,K] and B[K,N] and returns FP32 D[M,N]. An independent FP32 reference checks rectangular and square outputs. RMSNorm accepts packed FP16/BF16 rows, FP32 gamma and epsilon, and returns FP32 rows. CNNL documents the storage combinations, and its internal reduction precision must be confirmed separately. Both wrappers return the first vendor failure and perform cleanup.

Future vLLM-MLU integration belongs in an authorized development checkout. Use device-resident tensor buffers, the current stream and cached descriptors/workspace, and register a custom op with an explicit dtype and layout contract. Keep host staging in the standalone tool. Compare against the existing compiled implementation under identical dtype, shape and synchronization. Existing serving uses TMO and CNNL, including MatMulEx, batched GEMM and grouped GEMM. Use CNNL rather than adding a custom BANG kernel until a measured gap justifies one.

P1 should compare existing YaRN RoPE, MLA prefill/decode and paged cache paths against the DeepSeek reference. P2 should address softmax top-k, expert gather/combine and grouped GEMM after dense GEMM and MLA pass.
