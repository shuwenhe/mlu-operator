# DeepSeek-V2-Lite operator integration and acceptance

## Checkpoint and operator mapping

All four local safetensors headers agree with the index and declare 5,291 BF16 tensors. The scanner checked shard lengths and per-tensor byte counts. Full payload checksums were outside this audit. The tokenizer is BPE with 100,000 base entries, BOS 100000 and EOS 100001. The model has 102,400 embedding and output rows. Config selects `DeepseekV2ForCausalLM`, 27 layers, hidden width 2048, BF16 weights, no weight quantization and cache enabled. Source for the mapping is the checkpoint's `config.json`, `tokenizer.json`, index, tensor headers and `modeling_deepseek.py`.

Use T for flattened batch × tokens and E for tokens assigned to an expert. Linear weight tensors are checkpoint row-major [N,K]. The standalone packed GEMM contract takes A[T,K], B[K,N] and returns FP32 D[T,N]. Test fixtures transpose the selected weight rows before calling either wrapper.

| Model invocation | Input → output | Mapping |
| --- | --- | --- |
| `DeepseekV2Model.embed_tokens` | token IDs → [T,2048] | BF16 embedding gather, outside these operators. |
| `DeepseekV2Attention.q_proj` | [T,2048] → [T,3072] | Direct Q projection, no Q LoRA. |
| `kv_a_proj_with_mqa`, then `kv_a_layernorm` | [T,2048] → [T,576], split latent [T,512] and rotary key [T,64]; normalize latent only | GEMM then width-512 RMSNorm. |
| `kv_b_proj` and `o_proj` | [T,512] → [T,4096]; [T,2048] → [T,2048] | GEMM. KV-B supplies 16 heads × (128 nonrotary key + 128 value). |
| MLA attention and cache | Q/K head width 192 = 128 nonrotary + 64 rotary. V head width 128. | YaRN RoPE on the 64-wide rotary components, causal masking, FP32 softmax, QK/AV batched operations and KV cache update/read. Reference eager attention expands K/V; Flash, paged cache and compressed-cache MLA are optional execution choices. These paths are outside the standalone GEMM/RMSNorm integration. |
| `DeepseekV2DecoderLayer` input and post-attention norms; `DeepseekV2Model.norm` | [T,2048] → [T,2048] | Width-2048 RMSNorm, epsilon 1e-6 and learned gamma. Residual addition remains external. |
| Dense layer zero `DeepseekV2MLP` | gate/up [T,2048] → [T,10944]; down [T,10944] → [T,2048] | Three GEMMs, with SiLU(gate) × up between them. |
| Routed and shared MoE, layers 1–26 | routed gate/up [E,2048] → [E,1408], down reverses; shared projections use width 2816 | 64 routed experts, top 6 per token, 2 shared experts. Grouped GEMM and dispatch fusion are optional. |
| `MoEGate.forward` | [T,2048] → [T,64] → six IDs and probabilities | FP32 linear inputs/weights and FP32 softmax, greedy TopK, no top-k renormalization. Retain existing route. FP16/BF16 GEMM is unsuitable for this precision contract. |
| `DeepseekV2ForCausalLM.lm_head` | [T,2048] → [T,102400] | Untied GEMM. Test uses a representative output slice. |

Checkpoint fidelity requires BF16 activation/output semantics. FP16 fixtures are portability tests and explicitly re-encode selected checkpoint weights. The model's RMSNorm widens activations for variance and reciprocal square root, then casts normalized activations to input dtype **before** multiplying by gamma. The current CNNL wrapper returns FP32 post-scale values. A final cast alone cannot reproduce the reference's intermediate rounding. Keep existing model RMSNorm in serving until the output/rounding adapter or equivalent device implementation passes layer-level comparisons. GEMM FP32 outputs likewise require an explicit output-dtype adapter at boundaries.

## Operator test suite

[Fixture generator](../integration/deepseek/fixtures.py) reads real BF16 checkpoint slices directly, without importing model code or loading the full model. It generates 26 files with synthetic activations and either real sliced projection weights, real gamma, or synthetic weights. It preserves full K for Q, KV-A, KV-B, expert-up and logits projections. It covers width-512 and width-2048 norms, all-zero rows, width 1 and 65, and irregular GEMM M=3/N=129/K=65 in both dtypes.

[Runner](../integration/deepseek/runner.cpp) uses independent increasing-K FP32 GEMM and scalar FP32 RMSNorm references. Device comparison reports max and mean absolute error, max relative error with a 1e-6 denominator floor, NaN/Inf counts and mismatches against `1e-4 + 1e-4*abs(reference)`. Each device GEMM path launches three times per fixture and compares the last output. The RMSNorm comparison validates the current wrapper's FP32 contract. It does not validate the model's pre-gamma cast. Fixture tests cover independent call sites. They do not chain them into an end-to-end graph.

```sh
python3 integration/deepseek/fixtures.py /app/shuwen/model/DeepSeek-V2-Lite artifacts/build/deepseek-fixtures
integration/deepseek/build.sh artifacts/build/deepseek
export LD_LIBRARY_PATH="/usr/local/neuware/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
artifacts/build/deepseek/deepseek_operators --reference-only artifacts/build/deepseek-fixtures/*.fixture
```

Build in a compatible SDK userspace. The verified build used Ubuntu 24.04 with CNCC 5.7.6, CNRT 7.7.3 and CNNL 2.2.18, targeting `mtp_592`, from local image ID `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`. It used no network or device mappings. Default PIE linkage initially failed on CNCC host stubs. The harness links successfully with `-no-pie`. The reference-only log is `artifacts/logs/deepseek_reference.log`.

After explicit exclusive allocation and driver/SDK compatibility confirmation, run the CNRT transfer probe in a context exposing just the assigned card. Replace `--reference-only` with `--authorized-cnnl` to test CNNL GEMM and RMSNorm, or `--authorized-bang` to test custom BANG GEMM and CNNL RMSNorm. Save each CSV and stderr separately. The flag records operator intent and does not allocate hardware.

## Model integration design

The deployed vLLM-MLU distribution is `0.25.1+mlu0.16.0.pt212` with vLLM 0.25.1. Its package entry points register `vllm_mlu:register_mlu_platform` and `vllm_mlu:register_mlu_hijack`. In the installed editable checkout, `vllm_mlu/model_executor/models/__init__.py` maps `DeepseekV2ForCausalLM` to the MLU class. `model_executor/layers/linear.py` replaces the unquantized linear apply with `_mlu_ops.matmul` and TMO. `model_executor/layers/layernorm.py` replaces the out-of-tree RMSNorm path with `_mlu_ops.fused_rms_norm`. MLA selection and its cache remain model/backend responsibilities.

In a separate development checkout, register opt-in device-resident operations and wire only verified BF16 linear and norm call sites. Do not use the standalone wrappers directly inside inference because their host copies, allocation and private queues violate the existing tensor and stream lifecycle. Persist transposed weights, bind to the caller's current queue, reuse descriptors and workspace, preserve tensor metadata, and propagate vendor errors. Keep norm substitution gated on the cast-before-gamma contract and leave router, RoPE, MLA, cache, collectives and sampling unchanged. Add counters or trace symbols to prove the custom route is selected. Compare intermediate activations and logits for fixed token IDs, then compare prefill and cached decode token IDs against the deployed baseline on the same checkpoint, device and scheduler configuration.

## Acceptance and performance

| Deliverable | Evidence |
| --- | --- |
| GEMM correctness | 16 fixture FP32 references computed successfully. CNNL and custom GEMM linked into the runner. Device error metrics pending. |
| RMSNorm correctness | 10 fixture FP32 references computed successfully. CNNL wrapper linked. Device error metrics and model-rounding parity pending. |
| Actual MLU logs | BLOCKED. No authorized exclusive card was established and the verification container had no Cambricon nodes. No production process was modified. |
| End-to-end inference | BLOCKED. The integration adapter is a design, and the runner validates independent operator calls. No logits or generated tokens were measured. |
| Performance comparison | BLOCKED. The correctness runner can record CNNL GEMM kernel time. For matched timing, use the existing native and CNNL benchmark harnesses with the same shapes, dtypes, residency and warmups, and report kernel and end-to-end separately. RMSNorm needs matched device-event instrumentation before comparison. No measurements were collected. |
