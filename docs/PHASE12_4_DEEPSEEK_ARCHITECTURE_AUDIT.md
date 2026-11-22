# DeepSeek-V2-Lite checkpoint architecture audit

**PASS.** The header audit checked all 5,291 original BF16 tensor names, shapes, shard locations and byte extents across four safetensors shards. The checkpoint has 27 layers, hidden width 2048, an untied 102,400-token embedding and LM head, no attention bias, and RMSNorm epsilon 1e-6.

| Weight under the model or relative decoder prefix | Shape |
|---|---|
| `model.embed_tokens.weight`, `lm_head.weight` | [102400, 2048] each |
| `model.norm.weight`, `input_layernorm.weight`, `post_attention_layernorm.weight` | [2048] each |
| `self_attn.q_proj.weight` | [3072, 2048], direct query projection |
| `self_attn.kv_a_proj_with_mqa.weight` | [576, 2048], 512 latent plus 64 rotary |
| `self_attn.kv_a_layernorm.weight` | [512] |
| `self_attn.kv_b_proj.weight` | [4096, 512], expands 16 heads of 128 NoPE key and 128 value components |
| `self_attn.o_proj.weight` | [2048, 2048] |
| Layer 0 `mlp.{gate,up,down}_proj.weight` | [10944, 2048], [10944, 2048], [2048, 10944] |
| Layers 1–26 `mlp.gate.weight` | [64, 2048], greedy Top-6 with softmax |
| Layers 1–26 `mlp.experts.<expert>.{gate,up,down}_proj.weight` | [1408, 2048], [1408, 2048], [2048, 1408], for expert IDs 0–63 |
| Layers 1–26 `mlp.shared_experts.{gate,up,down}_proj.weight` | [2816, 2048], [2816, 2048], [2048, 2816] |

Each layer has 16 attention heads with a 192-wide query/key (128 NoPE plus 64 RoPE) and 128-wide value. The checkpoint's `q_lora_rank` is null and it has no query latent or expansion weight. YaRN uses factor 40, original context 4096, base 10000 and mscale values 0.707.

The vendor loader resolves checkpoint architecture `DeepseekV2ForCausalLM` to its MLU DeepSeek model. The installed plugin API is `ModelRegistry.register_model(model_arch, model_cls)`. The vendor model uses distinct prefill and decode attention call sites with common scheduler metadata. [Installed APIs](PHASE12_3_VENDOR_API_AUDIT.md)
