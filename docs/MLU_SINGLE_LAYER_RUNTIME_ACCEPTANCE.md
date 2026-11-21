# DeepSeek-V2-Lite single-layer runtime acceptance

**Execution PASS, numerical acceptance FAIL.** MLU 0 (`0000:4f:00.0`) completed both full layer-zero BF16 repetitions. The test uses real local checkpoint embedding rows and layer-zero tensors. Each batch-size-one request prefilled two fixed tokens and decoded the third token. Request A uses 100000, 42, 100. Request B uses 777, 512, 16. The two queues submit projection work before either is synchronized. Each request has independent device storage for expanded FP32 pre-rotary KV data and separate host rotary history. This is a bounded three-token materialized test cache with an explicit cursor, not a paged production cache or compressed MLA cache.

The layer executes native input RMSNorm, native Q and connected KV projections, native RoPE, explicitly selected CPU reference MLA, native output projection and post-attention RMSNorm, native gate/up/down projections, and explicit host BF16 SwiGLU and residual operations. Host attention and elementwise code are selected by the test contract. They are not hidden runtime fallback. The reference uses independently accumulated FP32 GEMM with BF16 model boundaries and the same explicit CPU attention formula, rather than a full PyTorch layer differential. Rotary values at positions 0–2 follow checkpoint YaRN parameters. Previously accepted standalone PyTorch RoPE parity and the failed FP16 MLA fixture were preserved.

| Stage | Compared elements across both repetitions | Outside unchanged tolerance | Max absolute error |
| --- | ---: | ---: | ---: |
| Q | 36,864 | 0 | 1.62124634e-5 |
| Projected KV cache | 81,920 | 0 | 7.15255737e-7 |
| CPU reference MLA using native intermediates | 24,576 | 0 | 1.90734863e-6 |
| Output projection FP32 | 24,576 | 0 | 7.15255737e-7 |
| Attention residual BF16 | 24,576 | 4 | 0.000244140625 |
| Post-attention RMSNorm with matched inputs | 24,576 | 0 | 0 |
| Post-attention RMSNorm fed by independent chains | 24,576 | 4 | 0.000244140625 |
| Gate | 131,328 | 1,754 | 2.7670525e-5 |
| Up | 131,328 | 2,412 | 2.77534127e-5 |
| Down | 24,576 | 1,948 | 5.33424318e-5 |
| Final layer output | 24,576 | 404 | 0.00048828125 |

The threshold remains `1e-5 + 1e-5*abs(reference)`. The final failures repeat at the same points across both repetitions. The first failed gate occurs after casting O projection to BF16 and adding the residual. Matched-input RMSNorm agrees exactly, isolating the earlier rounding boundary. Do not ship this path as a numerically accepted full layer. Keep the failing fixture and investigate projection accumulation and BF16 midpoint behavior before enabling production dispatch.

The initial reference-MLA test also exposed a position-buffer sizing error. CNRT rejected the upload into a two-byte-per-element allocation. Fixing u32 to four bytes per element resolved it. The first full-layer run stopped at an intermediate parity failure. The final run collected every stage and returned exit 1 after explicit teardown. It recorded 163,053,568 packed-weight bytes, 98,304 cache bytes, 414,732 peak separately managed bytes, and **zero remaining tracked bytes**. Host CNMON showed zero MiB used on all eight cards after the test.

[Full accuracy CSV](../artifacts/logs/runtime_full_layer_accuracy.csv) · [Final execution log](../artifacts/logs/runtime_final_hardware.log) · [Initial failed parity data](../artifacts/logs/runtime_full_layer_first_failure.csv) · [Original failed MLA fixture](../artifacts/logs/phase5_mla_accuracy.csv) · [Cleanup](../artifacts/logs/runtime_cleanup.log)

## Layer 0 P0 checkpoint rerun

**FAIL.** The original historical oracle still reports 404 final mismatches over two repetitions. An original-source PyTorch trace of the same BF16 fixture finds 4 out-of-tolerance final values in decode and an exact BF16 attention residual. The first original-fixture tolerance failures occur in FFN projections. The expanded matrix also exposes earlier failures for converted FP16 and long BF16 chunks. See [root cause](LAYER0_RESIDUAL_ROOT_CAUSE.md) and [precision analysis](LAYER0_RESIDUAL_PRECISION_ANALYSIS.md).

| Check | Result |
| --- | --- |
| Trace-enabled R6 compile and link | PASS |
| Native MLU execution of both dtypes and all four lengths | Executed; numerical FAIL |
| Original BF16 residual bit replay and independent MLU microprobe | PASS |
| Two interleaved slots and per-slot KV cache validation | PASS |
| Repeated execution and managed allocation cleanup | PASS |
| R3/R4 injected allocation rollback, queue sync fault and recovery | PASS |
| Original and expanded numerical acceptance | FAIL |
| Post-run physical MLU memory | PASS, all eight cards showed 0 MiB used |

Device traces ran on MLU590-M9 `0000:4f:00.0`; the microprobe ran on `0000:53:00.0`. Target `mtp_592`, driver 6.5.52, Ubuntu 24.04 SDK image `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`, CNCC 5.7.6, CNRT 7.7.3 and CNNL 2.2.18. R6 explicitly selects native projections and reference MLA. There is no implicit backend retry. Every R6 run reported `remaining=0`, R3/R4 exited zero and CNMON verified all eight cards. GLM53 remained exited and both Team containers remained running.

The [hardware log](../artifacts/logs/layer0_residual_hardware.log), [CPU trace log](../artifacts/logs/layer0_pytorch_trace.log) and [microprobe log](../artifacts/logs/layer0_residual_probe.log) contain the expanded compile and execution commands. From the workspace root, the reusable acceptance command is:

```bash
python3 runtime/native/p0/check_acceptance.py artifacts/logs/layer0_residual_accuracy.csv
```

It exited 1 with 284 of 576 original-source stage comparisons failing. Numerical tolerances remain unchanged. The accepted kernels and historical failure artifacts were preserved.
