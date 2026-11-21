# DeepSeek-V2-Lite MLU integration, phase 5

## Checkpoint contract

The local checkpoint config and model source define 27 layers with hidden width 2048, 16 heads, BF16 storage and untied 102,400-row embedding and logits tables. The local index and safetensors headers supply the shapes below. Linear weights are contiguous output-by-input `[N,K]), element strides `[K,1]`. Standalone GEMM uses contiguous A[M,K], B[K,N] and output C[M,N], with element strides `[K,1]`, `[N,1]`, and `[N,1]`. The host harness transposes checkpoint weights to B before device packing; production must preserve these logical layouts or introduce a tested transposed packer. FP32 results need an explicit activation-dtype cast at model boundaries.

| Model path | Checkpoint weight / activation contract |
| --- | --- |
| Q projection | `[3072,2048]`; output [B,S,16,192]. No Q LoRA rank. Split per head into nonrotary 128 and rotary 64. |
| KV-A and latent norm | `[576,2048]`; contiguous output row stride 576; latent columns 0–511 and shared rotary key columns 512–575. RMSNorm gamma `[512]`. Compact the strided latent slice before native RMSNorm. |
| KV-B and output projection | KV-B `[4096,512]`; output view [B,S,16,256], split key-noPE 128 and V 128. Output projection `[2048,2048]` follows attention concatenation of all V heads. |
| Dense FFN, layer 0 | Gate/up `[10944,2048]`, down `[2048,10944]`; SiLU(gate) × up. |
| MoE, layers 1–26 | Routed gate/up `[1408,2048]`, down `[2048,1408]`. Shared width 2816, two shared experts. 64 routed experts and six selected per token. Router `[64,2048]` is used as FP32 with softmax/greedy TopK, without top-k renormalization. Keep routing precision independent of low-precision GEMM. |
| Embedding, norms and LM head | Embedding and LM head `[102400,2048]`. Hidden RMSNorm vectors `[2048]`; epsilon 1e-6. Model RMSNorm rounds normalized values before gamma. |

MLA views, cache strides, masks and YaRN scale are in the [MLA design](MLU_ATTENTION_MLA_DESIGN.md).

## Execution evidence and phase-6 status

- Checkpoint-backed layer-zero Q, KV-A and KV-B projections passed on MLU 2 for FP16 and BF16, including all requested token counts. This compares BANG GEMM and CNNL as independent data paths.
- A connected native subgraph on MLU 2 kept persistent KV-A and KV-B weights and intermediates on device. It ran KV-A → low-precision cast and compact → native DeepSeek-order RMSNorm → KV-B for M=1 and 32. It copied back only the final tensor for validation, and all four combinations passed.
- Native YaRN rotation passed on MLU 4 against the exact checkpoint PyTorch CPU function. Standalone MLA on MLU 5 passed FP32 materialized and absorbed formulations, with one FP16 model-order discrepancy retained as a failed parity gate. No inference service was started.
- **End-to-end acceptance is incomplete.** The connected subgraph does not execute all decoder layers, attention/cache updates, MoE dispatch, logits and sampling as a native graph. Production routing and cache ownership have not been integrated. TTFT, TPOT, tokens/s, prefill/decode throughput, peak memory, multi-card inference, and a matched vLLM comparison remain unmeasured. No end-to-end result is inferred from operator timings.

[Projection acceptance](MLU_QKV_PROJECTION_ACCEPTANCE.md) · [RoPE acceptance](MLU_ROPE_ACCEPTANCE.md) · [Runtime design](MLU_RUNTIME_DESIGN.md)
