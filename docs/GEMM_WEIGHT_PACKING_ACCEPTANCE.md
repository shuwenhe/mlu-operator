# Packed-weight hardware acceptance

**PASS for tested FP16/BF16 fixtures.** The final build used CNCC 5.7.6, CNRT 7.7.3, CNNL 2.2.18 and host driver 6.5.52. It ran on MLU 0 (PCI `0000:4f:00.0`), with MLU 1 (`0000:53:00.0`) visible for wrong-device rejection. Only MLU 0 executed GEMM. The device timing run used MLU 2 (`0000:6f:00.0`).

For both dtypes, the harness compared every packed word, including all padding, in both preparation paths against the existing host layout reference. That is 32768 words per path for 3×193×65 and 98304 for 1×129×257. The prior hardware one-hot maps independently establish the fixed tile mapping. The check covered 524,288 word comparisons with zero differences.

The same handles served 1-, 10-, and 100-call groups with distinct deterministic A matrices (LCG seeds 7+i, RHS seed 43). The test used two live weights and two independent queues, including both queues reading the same immutable weight. Each result was compared against the independent serial FP32 oracle, with the unchanged tolerance `1e-5 + 1e-5*abs(reference)`. Both dtypes had zero maximum absolute, maximum relative and mean absolute error, mismatches, nonfinite outputs and output canary errors. Canaries detect adjacent writes for these fixtures and do not establish complete memory safety.

The benchmark separately verified its final result against the same oracle in each timed trial, including the DeepSeek-shaped projection. No end-to-end DeepSeek inference was run.

[Accuracy CSV](../artifacts/logs/gemm_prepacked_accuracy.csv) · [Build and correctness transcript](../artifacts/logs/gemm_prepacked_build.log) · [Original tile hardware analysis](GEMM_MATRIX_INSTRUCTION_ANALYSIS.md)
