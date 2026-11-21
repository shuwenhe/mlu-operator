# Phase 8.2 Layer 0 hardware acceptance

| Gate | Result |
| --- | --- |
| Experimental BANG C RMSNorm | PASS on the captured norm cases and the revised FP16 length-128 failing chunk |
| Frozen native projection compatibility | PASS, 0 differing output bits across 252 matched-input comparisons |
| Original checkpoint PyTorch Layer 0 | **PASS**, 0 / 576 stage comparisons outside tolerance and 0 / 1,421,312 final values outside tolerance |
| Historical C++ oracle, original BF16 fixture | **FAIL**, 400 / 24,576 final values outside tolerance across both repetitions |
| Native attention kernel acceptance | OPEN, the isolated fixture uses explicit checkpoint CPU reference attention |
| Dedicated performance acceptance | NOT RUN |

## Actual device coverage

The revised isolated executable ran on authorized MLU590-M9 cards at PCI `0000:4f:00.0` and `0000:53:00.0`. Target `mtp_592`, driver 6.5.52, CNCC 5.7.6, CNRT 7.7.3 and CNNL 2.2.18. Checkpoint reference arithmetic used the pinned PyTorch 2.12.1 CPU build. BF16 and FP16 used prefill lengths 1, 7, 32 and 128 plus the original BF16 two-token fixture. Each run executed two interleaved requests, token-at-a-time decode, and two repetitions. The CSV covers unique chunks. Observed nonfinite values: 0.

| Checkpoint stage | Chunk comparisons | Values outside tolerance |
| --- | ---: | ---: |
| input_norm | 36 | 0 |
| post_norm | 36 | 0 |
| KV latent RMSNorm | 36 | 0 |
| attention | 36 | 0 |

The revised fixture uses device RMSNorm with the experimentally validated cascade sum and explicitly rounded square root and reciprocal, frozen AVX2-compatible GEMM, and frozen YaRN RoPE. Attention uses explicit checkpoint ATen CPU arithmetic. The hybrid control ran ATen scale computation with the remaining norm operations on MLU. The initial full device-only variant left one out-of-tolerance norm value and 61 final failures in one FP16 case, and that evidence is preserved in the results directory. No engine automatically substitutes a backend.

The unchanged gate is `1e-5 + 1e-5 * abs(reference)`. The historical C++ comparator remains linked and can return 1 even when the independent checkpoint gate passes. Projection replay uses exact current operands for each projection. Accepted-source hashes and frozen files were checked. No production implementation or fixture was replaced.

## Lifecycle

The tests retained two disjoint request caches, reused the registry weights and reported zero remaining tracked bytes after each run. A fresh R3/R4 hardware run passed allocation rollback, injected synchronization fault recovery and cleanup. Post-run CNMON recorded 0 MiB used on every card.

- [Checkpoint Layer 0 accuracy](../artifacts/logs/phase8_2_layer0_accuracy.csv)
- [Matched projection regression](../artifacts/logs/phase8_2_projection_regression.csv)
- [Hardware FP16 128 execution](../artifacts/logs/phase8_2_exact_rsqrt_hardware.log)
- [Hardware remaining cases and lifecycle](../artifacts/logs/phase8_2_exact_rsqrt_remaining.log)
- [CPU replay commands](../artifacts/logs/phase8_2_exact_rsqrt_replay.log)
- [Post-run CNMON inventory](../artifacts/logs/phase8_2_cnmon_after.log)
