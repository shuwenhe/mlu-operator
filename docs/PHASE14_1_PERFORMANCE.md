# Phase 14.1 — Performance acceptance

Formal performance acceptance requires the numerical and lifecycle gates, warm-up, at least 100 measured repetitions, equivalent TP=1 and TP=2 workload conditions, and slowest-rank aggregation. The real engine traces used for numerical and lifecycle validation enable operator logging and raw file capture. Wall times from those traces describe those particular acceptance runs. They do not serve as a production performance baseline.

The formal comparison should capture TTFT, TPOT, prefill/decode and tokens/s through the real LLM API, then correlate GEMM, CNCL, ordered O and KV read/write timings with framework allocator and native-buffer peak memory.

The launchers and exporter are available in the Phase 14.1 integration directory. The TP=1 path exposes request timings; the TP=2 path also records worker stage timings and separates framework allocator peak from native packed weights. The benchmark must finish before reporting a comparison. See the machine-readable gate report for completion and sample counts.

Status: **NOT RUN**. Both numerical gates passed. The Vendor parent shutdown stalled after fault handling, so the ordered acceptance sequence stopped before formal timing. The CSV contains zero measured samples.
