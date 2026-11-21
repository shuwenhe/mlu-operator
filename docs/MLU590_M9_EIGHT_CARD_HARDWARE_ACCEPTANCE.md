# MLU590-M9 eight-card hardware acceptance

Executed on 2026-11-21 on `tza-0a09-ai01-em9`, after the user confirmed formal authorization for all eight cards. The dedicated disposable contexts granted only the requested compute and IPCM nodes and shared control. Existing Team containers retained their mappings and were left untouched. GLM53 remained stopped.

| Phase | Status | Actual evidence |
| --- | --- | --- |
| 1. Inventory and SDK | PASS | Host CNMON returned 0, eight MLU590-M9 reported Good with driver Running. Host PCI addresses in card order were `4f:00.0, 53:00.0, 6f:00.0, 73:00.0, af:00.0, b3:00.0, cf:00.0, d3:00.0`. Direct CNCC executable reported 5.7.6, MLVM 1.4, Clang 11.1.0. The Ubuntu 24.04 image reported installed CNNL 2.2.18 and CNRT 7.7.3. The in-container CNMON filter produced no card lines; CNDEV health evidence is from the host. |
| 2. Independent CNRT smoke | PASS | Runtime version 7.7.3 enumerated ordinals 0–7, each MLU590-M9 with 98,304 MiB. All device selections, properties, 4 KiB allocations, byte-exact H2D/D2H round trips, queue synchronization, frees and queue destroys passed. |
| 3. Multi-card | PASS | Eight simultaneously live buffers and independent queues held distinct patterns. Reverse-order readback matched all patterns and cleanup succeeded. The first negative-control check expected a different invalid-device enum. CNRT returned 100006 for ordinal 8, which the rerun accepted as an invalid-argument rejection. |
| 4. RMSNorm | PASS for fixtures | MLU 0 ran FP16 and BF16 at all five widths. All 17,540 pre-gamma values and final values matched expected bits, normalized FP32 passed, guards passed, and CNNL passed its separately evaluated FP32 contract. |
| 5. GEMM | PASS for fixtures | MLU 2 ran standalone BANG C and CNNL independently. BANG C passed 12 cases and CNNL passed 14, all with zero numerical error on their respective fixtures. |
| 6. Performance | PASS for recorded scopes | Single-card RMSNorm and GEMM timings recorded. Independent BF16 RMSNorm jobs measured 1/2/4/8-card throughput. GEMM scaling and CNCL collectives were outside this measurement. |

[Host inventory](../artifacts/logs/eight_card_inventory.log) · [CNRT smoke](../artifacts/logs/cnrt_smoke_test.log) · [Multi-card probe](../artifacts/logs/cnrt_multi_card.log) · [Initial negative-control log](../artifacts/logs/cnrt_multi_card_first_attempt.log)

## Scaling and cleanup

The [scaling harness](../operators/rmsnorm/native/scaling.cpp) ran one thread and queue per selected CNRT ordinal with resident BF16 256×2048 input. Each card used distinct gamma, five warmups and 100 timed launches. Elapsed time starts at common release and ends at the last queue completion. Allocation, initial transfers, readback and cleanup are excluded. Every readback matched its own weight pattern exactly. This measures aggregate throughput for independent resident jobs.

| Cards | Aggregate rows/s | Speedup |
| ---: | ---: | ---: |
| 1 | 621,780 | 1.000× |
| 2 | 1,242,277 | 1.998× |
| 4 | 2,484,242 | 3.995× |
| 8 | 4,948,901 | 7.959× |

Post-test CNMON reported 0 MiB used on all eight cards. Both Team containers remained running and GLM53 remained exited. Existing operator sources, fixtures and CPU evidence were preserved. Added only independent multi-card and scaling harnesses, reports and hardware evidence. No numerical tolerances changed. Device correctness claims above apply to the fixtures run; no end-to-end model inference was attempted.

[RMSNorm accuracy and timings](MLU590_M9_RMSNORM_HARDWARE_ACCURACY.md) · [GEMM accuracy and performance](MLU590_M9_GEMM_HARDWARE_ACCURACY_PERFORMANCE.md) · [Scaling CSV](../artifacts/logs/rmsnorm_mlu8_scaling.csv) · [Post-test device inventory](../artifacts/logs/eight_card_after_tests.log)
