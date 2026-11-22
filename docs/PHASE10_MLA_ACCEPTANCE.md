# Phase 10 native MLA acceptance

**PASS for the materialized DeepSeek-V2-Lite Layer 0 attention interface.** The input is the checkpoint's normalized hidden state `[S,2048]`. All model computation from that boundary through low-precision O projection output executes on MLU590-M9 card 0. The original checkpoint uses 16 heads. The reusable `MlaRequest` API groups independent requests on one or two queues. No host computation is used between stages.

The new `layout.mlu` kernels form the rotary input and absolute positions, cast and arrange Q, append expanded K/V, merge the AV heads, and cast the O projection. `mla.cpp` submits the accepted Phase 8 projection and latent RMSNorm arithmetic, YaRN RoPE, and the frozen Phase 9.4 QK, Softmax and AV launchers. The absorbed MLA formulation and fused batched GEMM are outside this interface.

| Boundary | Shape | Values checked per dtype | Bit differences |
| --- | --- | ---: | ---: |
| Q projection, after checkpoint cast | `[S,3072]` | 7,041,024 | 0 |
| KV-A projection, after checkpoint cast | `[S,576]` | 1,320,192 | 0 |
| Normalized latent | `[S,512]` | 1,173,504 | 0 |
| KV-B projection, after checkpoint cast | `[S,4096]` | 9,388,032 | 0 |
| Materialized Q, including rotated channels | `[16,S,192]` | 7,041,024 | 0 |
| Masked QK and Softmax probabilities | `[16,S,L]`, each | 3,900,576 | 0 |
| Merged AV and final O projection | `[S,2048]`, each | 4,694,016 | 0 |
| Decode O projection versus full-sequence PyTorch | `[1,2048]` | 147,456 | 0 |

Per-case maximum absolute, mean absolute, maximum relative and RMS errors were zero at the checkpoint boundaries above, with ULP histograms entirely at zero. The CSV distinguishes the checkpoint cast of FP32 GEMM staging from outputs physically stored in BF16 or FP16. The expected negative infinities from causal masks matched exactly. No unexpected NaN or Inf occurred.

The reference executes the original eager attention source with real Layer 0 weights and independent BF16 and FP16 captures. The test compares Q and expanded K after RoPE, including all rotated and nonrotary channels, and checks the generated positions explicitly. Input normalization preceding the attention interface was computed for fixture generation and is outside this acceptance.

[Per-case MLA accuracy](../artifacts/logs/phase10_mla_accuracy.csv) · [Design and layouts](PHASE10_DESIGN_AND_AUDIT.md)
