# Phase 13.3.2 — Numerical acceptance

Layer-0 replay covers frozen prompt lengths 1, 7, 17 and 65, prefill plus the three stored incremental decode steps. BF16 checkpoint weights and activations are used for the native kernels. The existing predicate is `abs(error) <= 1e-5 + 1e-5 * abs(reference)` with finite-value checks. Supplemental FP32 diagnostics use the same predicate, validated on independent unsharded MLU ATen references before the multi-rank sweep.

The CSV records maximum absolute/relative and mean absolute error, maximum ULP distance, bit mismatches, tolerance mismatches, first tolerance mismatch, NaN/Inf, fixture and output SHA-256 and per-rank device identity. The report checks output hash equality on replicated and gathered boundaries. Direct TP=1 and multi-rank O comparisons distinguish FP32 partials and CNCL sum, the single BF16 cast, and the addition of residual.

[Accuracy CSV](../artifacts/logs/phase13_3_2/accuracy.csv) · [Per-rank checkpoint hashes](../artifacts/logs/phase13_3_2/weight_shards.csv)

## Hardware result

| Gate | Result | Evidence |
|---|---|---|
| Five parameter mappings and reconstruction | PASS | 1,080 frozen inventory rank mappings cross-checked; all layer-0 device shard hashes verified, including 40 TP=8 records |
| TP=8 hardware participation | PASS | Eight device-bound ranks, CNCL and MLUCommunicator, physical MLU-Link |
| A — Q | PASS | Local projection, assembled head order and raw BF16 rotary slice |
| B — replicated KV-A | PASS | Latent and shared rotary components match on all ranks |
| C — KV-A RMSNorm | PASS | Frozen native BF16 operator replay and replicated output hashes |
| D — KV-B | PASS | Local K-noPE/V and assembled head order |
| E — O partial and FP32 CNCL reduction | PASS | Local independent FP32 diagnostic and exact repeated SUM before BF16 cast |
| E — O BF16 output and residual | FAIL | Original tolerance exceeded after BF16 conversion for TP=2/4/8 |
| TP=1 numerical | PASS | Historical O boundary matches exactly for all lengths and steps |
| TP=2/4/8 numerical | FAIL | Out-of-tolerance BF16 O and residual records retained |
| Full distributed RoPE / attention / cache | NOT RUN | Outside projection milestone |

All failing boundaries occur at or after BF16 conversion. Per-rank mismatch counts below include the same O mismatch once against the supplemental TP=1 output and once against the frozen historical O output.

| TP | O BF16 vs frozen | Residual BF16 | Largest FP32 O absolute error | Largest BF16 O absolute error |
|---|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 |
| 2 | 7 | 4 | 2.38419e-7 | 2.44141e-4 |
| 4 | 7 | 5 | 1.78814e-7 | 2.44141e-4 |
| 8 | 3 | 2 | 2.38419e-7 | 2.44141e-4 |

At TP=8, prompt length 7, decode step 2 and flattened O index 1134, FP32 was 0.0543212890625 versus 0.0543212965130806 in the unsharded reference. The TP value is exactly halfway between adjacent BF16 values. Round-to-nearest-even yields 0.05419921875, while the reference rounds to 0.054443359375. The frozen native GEMM accumulates FP32 into lanes over the full K axis before folding. The row-sharded path folds each local K range before CNCL SUM, so its FP32 accumulation order differs. The measurement preserves the existing tolerance, checkpoint bytes and single BF16 cast.

Independent review confirmed axes, head order, fixture offsets and the cast/residual order. Review also identified that stream stage time must be separated from direct launch timing, and that stability must cover latent and normalized outputs. The final timing path includes these changes.

| Additional gate | Result | Evidence |
|---|---|---|
| Repeated execution | PASS | 100 bitwise-stable repeats per rank in each TP configuration, across all projection and cast boundaries |
| Successful numerical launcher exit | FAIL | TP=2/4/8 returned nonzero after logging numerical mismatches and completing rank cleanup |
| Resource reclamation | PASS | Every rank emitted cleanup; final CNMON showed no running processes |
| Frozen evidence | PASS | Original Phase 13.3.1 SHA-256 manifest fully verified before work; frozen local files rechecked afterward |
| Phase 13.3.2 acceptance | FAIL | Frozen BF16 O/residual gate is unmet |

[Hardware and cleanup evidence](../artifacts/logs/phase13_3_2/hardware.log) · [Source manifest](../artifacts/logs/phase13_3_2/source_hashes.txt)
