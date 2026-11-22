# Native MLU performance roadmap

## Archived measurements and scope

| Measurement | Hardware result | Scope |
|---|---:|---|
| TP=8 ordered O chain at length 17 | p50 3.728 ms, p95 6.933 ms | Isolated projection chain with 8.50 MiB state scratch per rank |
| TP=8 output-axis O chain at length 17 | p50 2.084 ms, p95 2.834 ms | Isolated projection chain with two gathers |
| TP=1 reliability window | 3,609 s, 1,380/1,380 successful requests, 2.132 generated tokens/s | Historical TP=1 service loop. Separate KV lifecycle and performance gates remained blocked. |
| Production TP=2/4/8 latency, occupancy, utilization and bandwidth | NOT RUN | Real-engine performance acceptance remains open. |

[O performance evidence](PHASE13_3_3_PERFORMANCE_REPORT.md) · [TP=1 reliability data](../artifacts/logs/phase12_6_reliability_1h_summary.json)

## Measurement sequence

1. After the real TP=2 numerical gate, record end-to-end TTFT, decode inter-token latency, request p50/p95 and aggregate tokens/s. Replay identical token inputs at TP=1/2/4/8, with cold initialization reported separately.
2. Correlate Vendor worker timestamps and device profiling for GEMM, RMSNorm, RoPE, QK, Softmax, AV, O, cache scatter/gather and CNCL. Record actual input dimensions, dtype, workload, device capacity, bytes transferred, throughput, p50/p95 and slowest rank. Use MLU activity/occupancy and utilization counters when exposed by the frozen profiler. Mark unavailable counters explicitly as NOT RUN.
3. Profile expert-token histograms, per-rank wait and host synchronization. Compare grouped/device dispatch, O strategies and page/scratch improvements one at a time against frozen numerical references.
4. Attribute device allocation to packed weights, framework allocator, KV, scratch and runtime communication. Track encoded versus physical payload separately. Compare reuse, allocator peaks and fragmentation through concurrent and mixed workloads.
5. Report speedup and efficiency only between matching workloads and numerical contracts. Hardware evidence from replay and isolated kernels can guide investigation. Engine measurements establish production performance.
