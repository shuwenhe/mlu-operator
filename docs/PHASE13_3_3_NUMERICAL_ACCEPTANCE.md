# Phase 13.3.3 — Numerical acceptance on MLU590-M9

The ordered row compatibility run advanced from TP=1 through TP=2, TP=4 and TP=8 only after each configuration passed. Each exercises layer 0 with the original BF16 weights, frozen prompt bytes, lengths 1/7/17/65, prefill and the three saved decode steps. The O input is read from the saved Phase 13.3.2 AV boundary, and the historical O target is read from Phase 12.5.4. The criterion remains `abs(error) <= 1e-5 + 1e-5*abs(reference)`.

| Gate | Status | Evidence |
|---|---|---|
| Root cause and midpoint | PASS | Source fold comparison, IEEE midpoint and cancellation tests, old TP=8 snapshot, real old TP=2 failure |
| TP=1 compatibility | PASS | FP32 O, historical BF16 O and residual compare bit-exact |
| TP=2 BF16 O and residual | PASS | Both ranks bit-exact for all 16 fixtures |
| TP=4 BF16 O and residual | PASS | Four ranks bit-exact for all 16 fixtures |
| TP=8 BF16 O and residual | PASS | Eight ranks bit-exact for all 16 fixtures |
| Frozen numerical tolerance | PASS | Zero O/residual violations and zero NaN/Inf |
| 100-run stability | PASS | Bitwise local FP32 partial, ordered FP32, BF16 O and residual checks for each rank |
| Resource cleanup | PASS | All accepted launchers exit 0; rank cleanup and final CNMON recorded |
| Frozen evidence integrity | PASS | 9,599 distinct files verified, including all four checkpoint shards |
| Independent review | PASS | Frozen recurrence/fold, rank/stream ordering, stability comparison and alternative isolation |
| Projection compatibility | PASS | Isolated hardware experiment; production integration is outside this result |

There are 960 rank/boundary comparisons, 720 on the compatible path and 240 on the legacy FP32 control. All 720 ordered comparisons have zero bitwise differences. The CSV contains errors and SHA-256 for each result. The ordered `tensors/` capture contains the old local FP32 partial and CNCL SUM as controls plus ordered FP32, O BF16 and residual BF16. The original case at length 7, decode step 2 and index 1134 returns FP32 `0x3d5e8002` and BF16 `0x3d5f` at TP=8. The legacy control matches the midpoint `0x3d5e8000`.

The Phase 13.3.2 host mapping tests and the repository C GEMM reference test passed. The frozen Phase 13.3.2 hardware probe passed again at TP=1. Its TP=2 negative control reproduced 18 tolerance-mismatch occurrences on each rank and exited nonzero after rank cleanup. Legacy TP=4 and TP=8 reruns were NOT RUN under the stop-on-failure rule. All four ordered configurations passed.

[Accuracy CSV](../artifacts/logs/phase13_3_3/accuracy.csv) · [Host bit patterns](../artifacts/logs/phase13_3_3/host_rounding.log) · [Hardware log](../artifacts/logs/phase13_3_3/hardware.log) · [Source hashes](../artifacts/logs/phase13_3_3/source_hashes.txt)

The full-K output-axis alternative also matched the frozen FP32 and BF16 boundaries bit-for-bit at TP=1/2/4/8. The separate deferred-fold strategy passed TP=1 and failed at TP=2, with three O and three residual tolerance violations per rank. It still folds exactly once. This isolates the missing sequential rounding in independently initialized lane accumulators. Deferred TP=4/8 were NOT RUN after that failure.

[Output-axis accuracy](../artifacts/logs/phase13_3_3/output_axis/accuracy.csv) · [Deferred-fold negative control](../artifacts/logs/phase13_3_3/deferred/accuracy.csv)
