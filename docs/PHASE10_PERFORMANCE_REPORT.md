# Phase 10 MLA performance measurements

Hardware measurements ran after numerical and cache acceptance on MLU590-M9 card 0. All 28 configurations were profiled, with one validated warm-up pass and three measured passes per configuration. The native scopes begin with normalized hidden state already on the device and end with the rounded attention O projection. CNRT notifiers measure device intervals. Synchronized host wall time includes cache transaction submission and queue completion. Weight loading, fixture transfer, validation downloads and the separate cache-copy experiment are outside the end-to-end attention timing.

## Attention prefill and decode

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

## Memory and boundaries

At 131 committed-token capacity, the guarded cache allocated 1,341,696 bytes per request. One request used 15,843,840 managed bytes including its shared tables. Two used 31,653,120 managed bytes. Four packed weights used another 27,787,264 bytes per dtype. These steady-state byte counts exclude driver memory and the diagnostic prefix-gather scratch.

Prefill wall time is the time to the first attention output for an already resident normalized chunk. Decode wall time is the attention contribution to inter-token latency. Model-level TTFT, generation TPOT, sustained serving throughput, and capacity beyond 131 tokens were **not run** in Phase 10.

[Native performance samples](../artifacts/logs/phase10_performance.csv) · [PyTorch CPU samples](../artifacts/build/phase10/cpu-baseline.csv) · [Profiling run](../artifacts/build/phase10/performance.log)
