# Checkpoint-backed Q and KV projection acceptance

**Hardware PASS on host MLU 2 (`0000:6f:00.0`).** The fixture exporter reads full BF16 layer-zero Q, KV-A, KV-B and latent gamma tensors directly from safetensors. Activations are deterministic synthetic values. FP16 cases convert the source BF16 weights explicitly. One packed device weight per dtype and projection is reused across M=1, 2, 4, 8, 32, 64 and 128. CNNL uses the same logical inputs and resident unpacked weights. Outputs are compared to independent increasing-K FP32 accumulation with the existing `1e-5 + 1e-5*abs(ref)` threshold.

All 84 comparison rows passed with no nonfinite output. Maximum absolute error across all runs was 1.78813934e-7. Each timing uses five warmups and 30 samples. Kernel timings use queue notifiers. Transfer-inclusive timing adds A H2D, synchronization and FP32 C D2H, with both RHS representations resident. Weight H2D and pack-ready wall time are reported separately in the command log. That wall interval includes packing allocation and synchronization. The first BF16 Q preparation includes a cold path. It is a single observation, with no warmup redistribution.

| BF16 projection | M=1 BANG / CNNL median kernel µs | M=128 BANG / CNNL median kernel µs |
| --- | ---: | ---: |
| Q | 26.90 / 8.74 | 184.78 / 13.94 |
| KV-A | 17.62 / 7.40 | 42.18 / 8.44 |
| KV-B | 6.32 / 5.58 | 45.56 / 12.48 |

The connected KV path uses real KV-A, norm gamma and KV-B weights. Native preparation uses host packing once. The device casts and compacts 512 latent columns from a 576-stride FP32 intermediate, applies the accepted native RMSNorm, and submits KV-B on the same queue. Both dtypes passed M=1 and M=32 with zero mismatches. Worst absolute error was 7.72625208e-6 in FP16 M=32. This is one connected projection subgraph without attention.

[Projection accuracy/performance CSV](../artifacts/logs/phase5_projection_accuracy_performance.csv) · [Preparation and commands](../artifacts/logs/phase5_projection_run.log) · [Connected KV accuracy](../artifacts/logs/phase5_kv_chain_accuracy.csv) · [Connected KV build and run](../artifacts/logs/phase5_kv_chain_run.log)
