# Native matrix GEMM performance

Measured on MLU 2, `mtp_592`, driver 6.5.52, CNCC 5.7.6, CNRT 7.7.3 and CNNL 2.2.18. Each backend uses identical logical inputs, shape, dtype, queue and timing protocol. Five warmups precede 30 individual device-event samples. P95 is nearest rank. Transfer-inclusive wall time measures H2D for A and each backend's B representation, one synchronized launch and D2H for C. Allocations, algorithm setup and the one-time host packing are outside both timing intervals. The CSV reports host packing separately and includes both latency distributions and effective TFLOPS, calculated as 2MNK / kernel seconds / 1e12.

| M×N×K | FP16 kernel p50, BANG / CNNL (µs) | BF16 kernel p50, BANG / CNNL (µs) |
| --- | ---: | ---: |
| 127×193×65 | 3.60 / 2.76 | 3.56 / 2.76 |
| 32×64×128 | 2.86 / 2.32 | 2.92 / 2.32 |
| 128×256×512 | 7.16 / 5.24 | 7.16 / 5.24 |
| 256×128×512 | 7.28 / 4.02 | 7.28 / 4.04 |
| 1024×1024×1024 | 168.60 / 14.74 | 168.42 / 15.22 |

At 1024³, native reached 12.74–12.75 TFLOPS and CNNL reached 141.10–145.69 TFLOPS. The irregular case reached 0.885–0.895 native TFLOPS versus 1.155 for CNNL. The old scalar BANG kernel took about 1.99 ms at the irregular shape in a separate run with different fixtures and timing aggregation. Its approximate improvement is 550×. The new same-input comparison is the table above. Host RHS packing ranged about 39–7,081 µs across the native cases, which favors reusing packed weights. The benchmark records warm-up counts. Individual warm-up latencies and hardware utilization counters were not collected. The remaining large-shape gap merits profiling A row copies, repeated WRAM loads and tile occupancy. No equivalent end-to-end win is claimed.

## Independent eight-card scaling

BF16 resident 128×256×512, five warmups and 100 timed launches per card. One host thread and independent queue per card. The common wall interval ends with the last synchronized queue. Initial copies, readback and cleanup are outside the timed interval. Each card uses a distinct constant RHS and validates its own output. Mean per-card throughput is aggregate divided by card count. Individual card timing and inter-card communication were not measured.

| Cards | Aggregate TFLOPS | Mean per card | Speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.602 | 4.602 | 1.000× | 100.0% |
| 2 | 6.362 | 3.181 | 1.383× | 69.1% |
| 4 | 12.563 | 3.141 | 2.730× | 68.3% |
| 8 | 24.333 | 3.042 | 5.288× | 66.1% |

This is independent job throughput. No distributed GEMM or CNCL was introduced.

[Matched timing CSV](../artifacts/logs/gemm_native_matrix_performance.csv) · [Scaling CSV](../artifacts/logs/gemm_native_matrix_scaling.csv) · [Scaling command log](../artifacts/logs/gemm_matrix_scaling.log)
