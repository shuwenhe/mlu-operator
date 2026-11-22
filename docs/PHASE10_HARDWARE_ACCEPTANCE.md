# Phase 10 hardware acceptance

**PASS for the supported DeepSeek-V2-Lite Layer 0 native materialized MLA and KV cache interface.** Hardware was MLU590-M9 card 0, PCI `0000:4f:00.0`, with 98,304 MiB, CNCC 5.7.6, CNRT 7.7.3 and target `mtp_592`. The independent reference was the original eager checkpoint under PyTorch 2.12.1 CPU AVX2. Comparisons retained `1e-5 + 1e-5 * abs(reference)`.

| Acceptance check | Result |
| --- | --- |
| Native projection, latent RMSNorm, YaRN RoPE, attention, and output projection | PASS |
| Cache write/read, capacity, reset/reuse, guards, reclamation | PASS |
| BF16/FP16 prefill, three decode steps and full-sequence continuation parity | PASS |
| Single request, two independent requests sharing one or two queues, and mixed lengths | PASS, 28 configurations |
| Numerical results | PASS, 2,304 MLA stage comparisons and 480 cache stage comparisons with zero mismatches or unexpected nonfinites |
| Frozen QK, Softmax, AV and Phase 8.2 | PASS, byte-for-byte source verification |
| Phase 9.4 checkpoint replay, dispatch, and R1–R5 regressions | PASS |
| Legacy R6 reference-MLA parity | FAIL as recorded, with the same exit status and byte-for-byte output. Non-regression PASS. |
| Performance measurements | COMPLETE, separately reported |

The implementation is in `runtime/native/phase10/mla.{h,cpp}` and `layout.{h,mlu}`. The build, checkpoint capture, device tests, and matrix runner are in the same directory. The original attention sources and dispatch were preserved. The source snapshot, pre-change Git status, and pre-change tracked patch are under `artifacts/build/phase10`.

For reproduction, use the same Phase 8.2 SDK container with only card 0 exposed. Run `capture.py CHECKPOINT RUNTIME_FULL_FIXTURES OUTPUT` with `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, then rerun it with `--full-sequence-only` to capture the continuation controls. Run `build.sh OUTPUT ACCEPTED_PACKED_OBJECT_DIRECTORY` and `run_matrix.sh OUTPUT/mla_test FIXTURES RUN_LOG_DIR 0000:4f:00.0`. Set `MLA_PROFILE=1` for the performance pass and run the capture script with `--benchmark` for the CPU baseline. The fixture and source verification report records the exact hashes.

The supported matrix uses the checkpoint's 16 heads, a maximum tested chunk of 128 and maximum tested capacity of 131. Groups of two use independent request allocations. Other head counts, fused batch kernels, absorbed MLA, full-model generation and context lengths beyond the tested matrix are outside this acceptance.

[Design audit](PHASE10_DESIGN_AND_AUDIT.md) · [MLA](PHASE10_MLA_ACCEPTANCE.md) · [KV cache](PHASE10_KV_CACHE_ACCEPTANCE.md) · [Prefill and decode](PHASE10_PREFILL_DECODE_ACCEPTANCE.md) · [Performance](PHASE10_PERFORMANCE_REPORT.md) · [Regression log](../artifacts/logs/phase10_regression.log) · [Source hashes](../artifacts/logs/phase10_source_hashes.txt)
