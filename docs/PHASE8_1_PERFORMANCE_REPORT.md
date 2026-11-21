# Phase 8.1 performance report

**BLOCKED by Layer 0 numerical acceptance.** Phase E requires the full numerical matrix to pass before dedicated benchmarking. No cross-backend kernel, prefill, decode, throughput, packing, transfer-inclusive or peak physical memory benchmark was run. The projection CSV leaves per-kernel latency blank, and the performance CSV marks those metrics BLOCKED.

The already completed hardware fixtures record whole-process wall time. They include fixture loading, device work, two request slots and repetitions, CPU reference MLA and historical oracle work, transfers, and diagnostic trace output. For the candidate, M=1 took 3.190 s BF16 and 3.762 s FP16. M=128 took 38.255 s BF16 and 57.700 s FP16. These times do not establish kernel or model performance.

The runtime recorded 163,053,568 bytes of packed weights for each candidate case, 65,536–4,227,072 bytes of KV cache, and 247,048–21,542,916 bytes of peak managed memory. Managed peak includes the KV cache and excludes the packed registry. All managed allocations were released. CNMON is a post-run snapshot, and peak physical memory was not sampled.

[Performance CSV](../artifacts/logs/phase8_1_performance.csv) · [Hardware log](../artifacts/logs/phase8_1_hardware.log)
