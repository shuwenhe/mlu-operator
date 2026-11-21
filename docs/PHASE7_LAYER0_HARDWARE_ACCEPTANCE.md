# Phase 7 Layer 0 hardware acceptance

**FAIL.** The authorized eight-case FP16/BF16 sweep ran on MLU590-M9 PCI `0000:4f:00.0`, with two interleaved requests, individual KV cache buffers, two repetitions and one decoded token per request. The native and original eager checkpoint comparisons retain the existing tolerance. Full Layer 0 closure remains BLOCKED by the projection parity failures.

Host CNMON/driver 6.5.52 reported eight MLU590-M9 devices. The pinned Ubuntu 24.04.3 SDK image used CNCC 5.7.6 targeting `mtp_592`, CNRT 7.7.3 and CNNL 2.2.18. The CPU reference was PyTorch 2.12.1+cpu. Native was selected explicitly; MLA and trailing pointwise arithmetic used the existing explicit reference path.

| Dtype | Prefill length | Final prefill values outside tolerance | Final decode values outside tolerance |
| --- | ---: | ---: | ---: |
| BF16 | 1 | 0 / 4096 | 0 / 4096 |
| BF16 | 7 | 9 / 28672 | 2 / 4096 |
| BF16 | 32 | 616 / 131072 | 0 / 4096 |
| BF16 | 128 | 3723 / 524288 | 376 / 4096 |
| FP16 | 1 | 375 / 4096 | 361 / 4096 |
| FP16 | 7 | 5251 / 28672 | 666 / 4096 |
| FP16 | 32 | 27109 / 131072 | 1209 / 4096 |
| FP16 | 128 | 119360 / 524288 | 783 / 4096 |

The CSV compares unique chunks; the runtime repeats each case and rewrites its trace with the repeated output. No nonfinite values were observed. The original historical fixture again failed with 404 / 24,576 final values. The separately preserved original-source comparison for that fixture failed with 4 / 12,288 final values. Across the full original-source stage gate, 284 / 576 comparisons failed.

| Gate | Status | Evidence |
| --- | --- | --- |
| Trace build and execution | PASS | All eight cases completed and produced both slots' traces |
| FP32 projection agreement with matched FP64 | PASS | 224 comparisons at unchanged tolerance |
| Low-precision PyTorch agreement | FAIL | Matching operands still produce rounded disagreements |
| Cache ownership, repeated execution and packed weight reuse | PASS | The two runtime slots have disjoint KV buffers and reuse persistent registry entries |
| Allocation rollback and injected queue sync recovery | PASS | Fresh hardware R3/R4 exited 0 |
| Cleanup | PASS | Every run reported zero remaining tracked bytes; before and final CNMON snapshots available; final physical usage was 0 MiB on all eight cards |
| Corrected Layer 0, isolated performance and experimental compiler evidence | BLOCKED | No validated parity correction |

The traced fixture process took 3.253–58.447 s, including CPU references, device calls, transfers, and trace serialization. Persistent packed weights used 163,053,568 bytes. Tracked KV memory was 65,536–4,227,072 bytes and peak managed memory was 247,048–21,542,916 bytes. Peak managed memory includes cache and excludes packed registry entries. The timing CSV labels unavailable measurements for individual loading, packing, initialization, prefill, decode, kernel and transfer scopes, peak physical use and workspace-only peak. No accepted end-to-end latency is reported.

- [Projection accuracy](../artifacts/logs/phase7_projection_accuracy.csv)
- [Layer 0 accuracy](../artifacts/logs/phase7_layer0_accuracy.csv)
- [Performance and memory CSV](../artifacts/logs/phase7_performance.csv)
- [Hardware commands and execution log](../artifacts/logs/phase7_hardware.log)
- [CNMON before](../artifacts/logs/phase7_cnmon_before.log) · [CNMON after](../artifacts/logs/phase7_cnmon_final.log)
