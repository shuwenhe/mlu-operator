# Phase 13.3.1 — DeepSeek-V2-Lite MLA architecture audit

## Verified architecture

The local checkpoint config and safetensors headers establish hidden width 2,048, 27 layers, 16 attention heads and BF16 weights. Each Q/K head is 192 channels (128 non-rotary + 64 rotary), each V head is 128 channels. The query LoRA rank is null. The KV latent rank is 512. Attention bias is disabled and the configured dropout is zero. The model config advertises 163,840 positions. The audited NeurX model accepts configured max length at most 128, the metadata/attention interfaces cap history at 131 and queries at 128, and the frozen RoPE tables contain 131 rows. Those are distinct limits.

Checkpoint `DeepseekV2Attention.__init__/forward` sets the per-head reshape and applies attention. `DeepseekV2DecoderLayer.forward` applies pre-attention norm and the residual outside attention. The installed Vendor class `MLUDeepseekV2MLAAttention` independently selects local heads, column Q/KV-B, replicated KV-A, and row output. Its compressed-cache kernels are a separate implementation surface. [Vendor source locations](../artifacts/logs/phase13_3_1/vendor_audit.txt)

## Execution order and native boundaries

Shapes below omit the request batch axis. M is the scheduled query count, L its committed history length. H=16 in the frozen path, H=2 in the proposed TP=8 path. Projection GEMMs use BF16 inputs/weights and return FP32 in NeurX; the checkpoint eager Linear returns activation dtype.

| Operation | Frozen shape and dtype | Current implementation and native entry | Phase 13.2 and planned placement |
|---|---|---|---|
| Input / pre-norm | BF16 [M,2048] to same; gamma [2048] | Decoder input norm, `NativeRMSNorm` | Replicated; no Linear |
| Query projection | [M,2048] × W [3072,2048] to FP32 [M,3072] | `_project`, `PackedProjection` | Column, BF16 GEMM, local [M,384] |
| Query LoRA A/B and Query RMSNorm | Absent | Eager branch skipped because query rank is null; no weights | Absent on every rank |
| KV latent projection | W [576,2048], FP32 [M,576] to 512 latent + 64 shared rotary K | `PackedProjection`; `phase5_cast_kv` narrows latent | Replicate GEMM and raw rotary term |
| KV latent norm | BF16 [M,512] to same; gamma [512] | `NativeRMSNorm`, epsilon 1e-6 | Replicated across full width |
| KV expansion | W [4096,512], FP32 [M,4096] = [M,16,256] | `PackedProjection` | Column, local [M,512]; each head owns K-nope then V |
| Q/K separation and RoPE | Q [M,16,192] splits 128+64; shared raw K [M,64]; BF16 rotary staging [17,M,64] | `phase10_prepare_bf16`, `worker_rope_bf16`, `phase10_append_bf16` | Local Q stripes and one shared K stripe; existing 17-stripe ABI needs safe staging |
| Cache scatter/gather | BF16 K [16,M,192], V [16,M,128] to slot-major storage and back as [16,L,D] | `PagedKV.write/read` | Local two-head physical cache; global logical metadata |
| Scores and mask | BF16 Q/K; FP32 raw QK [16,M,L], BF16 scaled, FP32 causal-masked | `NativeMaterializedAttention`, `phase9_launch_qk_bf16` and frozen scale | Local heads, full L per head; no Linear |
| Softmax | FP32 mask to FP32 probabilities and BF16 cast [16,M,L] | `phase9_softmax_bf16`, normalize across L | Local, no rank reduction |
| Value aggregation | BF16 P/V to FP32 accumulator then BF16 [16,M,128] | `phase9_av_bf16`; `phase10_merge` converts to token-major [M,2048] | Local attention, merge to [M,256] |
| Output projection | W [2048,2048], BF16 GEMM inputs, FP32 [M,2048] then BF16 | `PackedProjection`, `phase10_cast_bf16` | Row, local weight [2048,256], FP32 CNCL SUM then one BF16 cast |
| Residual | BF16 [M,2048] + [M,2048] | `phase11_residual` through `_add` | Replicated, after reduced output is rounded; no fused Row residual |

[Native model](../integration/vllm_native_mlu/phase12_5/model.py) · [Attention ABI](../integration/vllm_native_mlu/attention.py) · [Native layout](../runtime/native/phase10/layout.mlu)

## RoPE / YaRN compatibility

Theta=10,000, rotary width=64, factor=40, original context=4,096, beta_fast=32, beta_slow=1, mscale=mscale_all_dim=0.707. The checkpoint mixes interpolated and extrapolated inverse frequencies with the YaRN ramp. Its table amplitude ratio is 1. Score scale is `192**(-0.5) * (1 + 0.1 * 0.707 * log(40))**2`, about 0.1147213868.

Checkpoint `apply_rotary_pos_emb` deinterleaves adjacent pairs into halves before `rotate_half`. Native `worker_rope_bf16` reads even/odd lanes and uses the same signs, narrows each product to BF16, then sums and narrows. The native cursor is L−M and is validated against the scheduler's zero-based absolute positions. Prefill starts at 0; decode continues at the committed cursor even if its physical page changes. Shared K is rotated once per token per rank, then copied into the K tail of each local head. Frozen Phase 12.5.4 key-tail and decode acceptance are the existing device evidence. The audit rebuilds the BF16 tables and compares scale at its actual `ctypes.c_float` ABI; the two decimal spellings differ by one FP64 ULP. No enlarged positional contract is inferred. [RoPE replay](../artifacts/logs/phase13_3_1/rope_verified.txt) · [Frozen regression](PHASE12_5_4_ROOT_CAUSE_REPORT.md)
