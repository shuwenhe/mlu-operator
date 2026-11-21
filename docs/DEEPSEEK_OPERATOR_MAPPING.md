# DeepSeek-V2-Lite Layer 0 operator mapping

These dimensions come from the local checkpoint config and safetensors headers. Layer 0 uses direct Q projection and a dense FFN. The selected tensors are stored as BF16. The adapter can run the original BF16 weights or explicitly convert them and inputs to FP16.

| Adapter tensor | Checkpoint suffix under `model.layers.0` | Checkpoint shape | Device operation |
| --- | --- | --- | --- |
| Q | `self_attn.q_proj.weight` | [3072, 2048] | [M,2048] × [2048,3072] → [M,3072]; 16 heads × (128 non-rotary + 64 rotary) |
| KV-A | `self_attn.kv_a_proj_with_mqa.weight` | [576, 2048] | [M,2048] × [2048,576] → [M,576]; 512 compressed + 64 shared rotary |
| KV-B | `self_attn.kv_b_proj.weight` | [4096, 512] | [M,512] × [512,4096] → [M,4096]; 16 heads × (128 non-rotary key + 128 value) |
| Output | `self_attn.o_proj.weight` | [2048, 2048] | [M,2048] × [2048,2048] → [M,2048] |
| Gate | `mlp.gate_proj.weight` | [10944, 2048] | [M,2048] × [2048,10944] → [M,10944] |
| Up | `mlp.up_proj.weight` | [10944, 2048] | [M,2048] × [2048,10944] → [M,10944] |
| InputNorm | `input_layernorm.weight` | [2048] | [M,2048] → [M,2048] |
| KvNorm | `self_attn.kv_a_layernorm.weight` | [512] | [M,512] → [M,512] |
| PostNorm | `post_attention_layernorm.weight` | [2048] | [M,2048] → [M,2048] |

All strides are in elements. A dense projection input uses strides `[K,1]` and its output `[N,1]`. The checkpoint stores `[N,K]` with strides `[K,1]`. Host preparation transposes once to `[K,N]` with strides `[N,1]`. Native stores a registry-owned packed representation and CNNL reads a separately uploaded logical buffer. Both projection routes stage FP32 outputs. A caller implementing the checkpoint decoder must narrow the projected values at the checkpoint's defined boundaries.

The native connected subgraph reads low-precision [M,2048], runs KV-A to FP32 [M,576], casts columns 0 through 511 to a compact low-precision [M,512], runs native RMSNorm with the checkpoint KV gamma, and runs KV-B to FP32 [M,4096]. The wide KV-A result still holds the shared rotary columns 512 through 575. Native RMSNorm uses matching low-precision input, gamma and output with epsilon `1e-6`. The CNNL RMSNorm route uses FP32 gamma and output and retains its own precision semantics.

Native YaRN expects low-precision input [17×S,64] with head-major rows and the interleaved rotary lane layout, two contiguous [P,64] low-precision tables, U32 positions [1,S], and low-precision output [17×S,64] with half-split rotary lanes. The 17 streams are 16 query heads and one shared key stream. Existing Dispatch requires `1 <= S <= 1024` and in-range position indices. The caller builds the tables and stages the streams. Local config sets base 10000, factor 40, beta 32/1, original context 4096 and equal `mscale`/`mscale_all_dim` of 0.707.

CPU comparisons run independent dot products and RMSNorm/RoPE arithmetic on the host. CNNL is explicitly selectable for individual projections and RMSNorm. Connected KV and YaRN require Native.
