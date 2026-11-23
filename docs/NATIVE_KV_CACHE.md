# Native KV cache and prefill/decode

- [Phase 10 KV cache acceptance](#phase10-kv-cache-acceptance)
- [Phase 10 prefill and continuous decode acceptance](#phase10-prefill-decode-acceptance)
- [Phase 10 hardware acceptance](#phase10-hardware-acceptance)
- [Phase 10 MLA performance measurements](#phase10-performance-report)

<a id="phase10-kv-cache-acceptance"></a>

## Phase 10 KV cache acceptance

**PASS on MLU590-M9 card 0.** Each request owns physical head-major K `[16,capacity,192]` and V `[16,capacity,128]` buffers in its dtype. A device kernel writes only the appended token interval. The frozen QK and AV kernels read that same device memory using the physical head stride and proposed valid length. The request advances its committed length only after its queue completes. The 512-wide compressed latent is a projection intermediate under the eager checkpoint contract. Persistent cache storage uses expanded K/V.

All checked committed prefixes matched the original PyTorch eager cache bit-for-bit. Per dtype this covered 17,491,968 K values and 11,661,312 V values. Full-prefix comparisons after each append verified that earlier tokens remained intact. The device cache guards and unused suffix sentinels remained intact through prefill and capacity filling. The harness also checked positions, rejected a non-contiguous initial append and a capacity-overflow append, and rejected reset or append while work was pending.

Reset and reuse were tested with the other request's input tokens. Stale suffix storage remained outside the new committed length and the resulting probabilities and outputs matched checkpoint. Two independent cache allocations were exercised on one queue and on two queues. A mixed group used a 10-token capacity and a 131-token capacity at the same time. Device memory returned to zero managed bytes after cache, scratch and shared-table release. Packed weights were also closed.

The implementation uses logical reset. Erasing cache storage for secure reuse and concurrent host-thread access are outside the request API. Queue submission and cleanup are owned by the worker thread.

[Cache accuracy](../artifacts/logs/phase10_kv_cache_accuracy.csv) · [Decode and lifecycle log](../artifacts/logs/phase10_decode_hardware.log)

<a id="phase10-prefill-decode-acceptance"></a>

## Phase 10 prefill and continuous decode acceptance

**PASS on card 0 for the supported materialized MLA matrix.** BF16 and FP16 each ran prefill lengths 1, 7, 32 and 128, then three single-token decode steps, then reset and reuse. The following configuration families were tested for both dtypes.

| Lengths | Requests | Queues | Runs |
| --- | ---: | ---: | ---: |
| 1, 7, 32, 128 | 1 | 1 | 8 |
| 1, 7, 32, 128 | 2 | 1 | 8 |
| 1, 7, 32, 128 | 2 | 2 | 8 |
| Mixed 7 and 128 | 2 | 1 or 2 | 4 |

There were 48 prefill, 144 decode, and 48 reset/reuse request steps. Every decode output matched both the independently captured incremental checkpoint and the same token in an original full-sequence PyTorch run. Prefill compares against same-shape checkpoint prefill. A cross-shape CPU-only diagnostic found 19 out-of-tolerance values when comparing FP16 request 1's seven-token prefill against the corresponding prefix of PyTorch's ten-token full run. The subsequent three continuation tokens match. This is a difference between reference execution shapes and is separate from native same-shape prefill acceptance.

All comparisons retained `1e-5 + 1e-5 * abs(reference)`. The causal masks, latent path, expanded K/V, Softmax, merged AV and final projection were checked per step. There were zero unexpected nonfinites, out-of-tolerance values, or cache guard failures. Logical batch sizes 1 and 2 are represented by independently owned requests. Other head counts, a fused multi-request batch kernel, prefill chunks above 128, and cache capacities above 131 are outside the measured matrix.

[Prefill hardware log](../artifacts/logs/phase10_prefill_hardware.log) · [Decode hardware log](../artifacts/logs/phase10_decode_hardware.log)

<a id="phase10-hardware-acceptance"></a>

## Phase 10 hardware acceptance

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

[Design audit](NATIVE_ATTENTION.md#phase10-design-and-audit) · [MLA](NATIVE_ATTENTION.md#phase10-mla-acceptance) · [KV cache](#phase10-kv-cache-acceptance) · [Prefill and decode](#phase10-prefill-decode-acceptance) · [Performance](#phase10-performance-report) · [Regression log](../artifacts/logs/phase10_regression.log) · [Source hashes](../artifacts/logs/phase10_source_hashes.txt)

<a id="phase10-performance-report"></a>

## Phase 10 MLA performance measurements

Hardware measurements ran after numerical and cache acceptance on MLU590-M9 card 0. All 28 configurations were profiled, with one validated warm-up pass and three measured passes per configuration. The native scopes begin with normalized hidden state already on the device and end with the rounded attention O projection. CNRT notifiers measure device intervals. Synchronized host wall time includes cache transaction submission and queue completion. Weight loading, fixture transfer, validation downloads and the separate cache-copy experiment are outside the end-to-end attention timing.

### Attention prefill and decode

Medians for length-128 prefill and the three following one-token continuations. Per-request device intervals can overlap on two queues. Wall latency and query-token throughput cover the active request group.

| Dtype | Requests / queues | Prefill wall ms | Prefill query tokens/s | Decode wall ms | Decode query tokens/s |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 | 1 / 1 | 49.529 | 2,584 | 0.867 | 1,154 |
| BF16 | 2 / 1 | 99.367 | 2,576 | 1.712 | 1,168 |
| BF16 | 2 / 2 | 70.808 | 3,615 | 0.958 | 2,088 |
| FP16 | 1 / 1 | 49.505 | 2,586 | 0.867 | 1,154 |
| FP16 | 2 / 1 | 99.388 | 2,576 | 1.713 | 1,168 |
| FP16 | 2 / 2 | 70.807 | 3,615 | 0.963 | 2,077 |

For FP16 on one queue, the projection, latent norm and RoPE interval was 15.589 ms in prefill and 269.52 µs in decode. The append/layout kernel was 175.00/4.08 µs, QK → Softmax → AV with direct cache reads was 27.243 ms/461.40 µs, and head merge plus O projection/cast was 6.465 ms/106.04 µs. Complete device intervals were 49.477 ms and 840.96 µs. The separate device-to-device prefix gather used 32 CNRT copies and measured 184.76/183.52 µs. This gather measures a copy-based cache read. Production attention reads the cache directly inside QK and AV.

The same normalized-input eager PyTorch baseline used the original Layer 0 attention and two CPU threads. At length 128, median BF16 prefill/decode was 528.783/4.354 ms and FP16 was 462.439/3.917 ms. These are CPU-versus-MLU reference measurements with different runtime implementations. They do not establish a gain over another MLU implementation.

### Memory and boundaries

At 131 committed-token capacity, the guarded cache allocated 1,341,696 bytes per request. One request used 15,843,840 managed bytes including its shared tables. Two used 31,653,120 managed bytes. Four packed weights used another 27,787,264 bytes per dtype. These steady-state byte counts exclude driver memory and the diagnostic prefix-gather scratch.

Prefill wall time is the time to the first attention output for an already resident normalized chunk. Decode wall time is the attention contribution to inter-token latency. Model-level TTFT, generation TPOT, sustained serving throughput, and capacity beyond 131 tokens were **not run** in Phase 10.

[Native performance samples](../artifacts/logs/phase10_performance.csv) · [PyTorch CPU samples](../artifacts/build/phase10/cpu-baseline.csv) · [Profiling run](../artifacts/build/phase10/performance.log)
