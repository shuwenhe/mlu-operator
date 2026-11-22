# DeepSeek-V2-Lite end-to-end native MLU acceptance

**Native BF16 generation executed successfully. Overall acceptance remains PARTIAL because the full forward pass failed the existing numerical tolerance.** The original local checkpoint ran through all 27 decoder layers on MLU590-M9 card 0 at PCI `0000:4f:00.0`. Two native runs generated 32 greedy tokens each from a 17-token prompt. All 32 IDs in each run matched the original checkpoint under independent CPU PyTorch.

| Area | Decision |
| --- | --- |
| A. Native operators | PASS on the recorded isolated matrix for embedding, GEMM, RMSNorm, YaRN, MLA and KV cache, QK, Softmax, AV, dense SwiGLU, MoE routing/experts, vocabulary projection and argmax. FP32 router projection parity remains open on integrated inputs. |
| B. Full loading and forward | PASS for actual BF16 checkpoint residency and execution of all 27 layers, on one-token and multi-token prefill and repeated decode. FAIL for full tensor numerical parity. |
| C. Prompt-to-token generation | PASS for native execution and all 32 greedy IDs on the tested prompt. Full numerical acceptance and general model support remain open. |

## Verified checkpoint and execution path

The index and four safetensors shards contain 5,291 BF16 tensors and 31,412,968,448 payload bytes. The audit matched every tensor name and shape. The tokenizer contains 100,002 entries within the model's 102,400-row vocabulary. The checkpoint has 16 attention heads, hidden width 2,048, an untied embedding and LM head, one dense layer with width 10,944, and 26 MoE layers. Each MoE layer routes six of 64 width-1,408 experts and also evaluates the width-2,816 shared branch.

Layer 0's full dense block matched bitwise across 160 stage comparisons. The isolated BF16 suite exercised all 26 real MoE layers using shared representative one- and seven-row inputs. The integrated runner packs expert weights lazily and caches the packed weights. It uses a device-resident MLA cache per layer. The host handles tokenizer encoding, queue scheduling from integer expert IDs, and token emission. No model computation fell back to CPU, PyTorch or CNNL. The independent reference uses the original checkpoint layer classes and separate cache state.

## Forward-pass parity

The acceptance threshold was `1e-5 + 1e-5 * abs(reference)`. The comparison used 330 tensors spanning embedding, per-layer attention output, committed K/V, decoder output and final vocabulary logits. The one-token and multi-token traces used the same inputs. The first decode token also matched, so its trace used identical token IDs across implementations.

| Case | First differing layer | First-layer out-of-tolerance values | Out-of-tolerance logits / 102,400 | Logit max abs |
| --- | --- | ---: | ---: | ---: |
| One-token prefill | 26 | 3 | 1,109 | 0.0625 |
| 17-token prefill | 2 | 1 | 89,636 | 0.375 |
| First incremental decode | 3, following the prefill divergence | See trace CSV | 91,286 | 0.375 |

Embedding matched bit-for-bit. No compared tensor had unexpected nonfinite values. Matched-input replays identified the FP32 MoE router projection as the first differing operation in the failing Layer 2 and Layer 26 cases. Expert IDs, dispatched expert outputs and the shared branch agreed. The router score differences crossed BF16 rounding boundaries during weighted aggregation. Feeding exact checkpoint logits in an isolated diagnostic restored bitwise equality through device Softmax, top-six and expert reduction. This diagnostic is outside the native generation path.

## Source protection and performance scope

Frozen QK, Softmax, AV and Phase 10 MLA hashes verified. The attention and cache regressions passed. Legacy R6 reproduced its known parity failure byte-for-byte. Runtime scope is BF16, batch one, checkpoint-fixed head dimensions and an existing 131-position rotary table. The measured generation reached 48 committed tokens. Other prompts, longer context, full-model FP16, batch serving and early-EOS termination were not accepted by these tests.

[Phase 11 implementation, numerical diagnosis and reproduction](PHASE11_DEEPSEEK_V2_LITE_INTEGRATION.md) · [Hardware log](../artifacts/logs/phase11_hardware.log) · [Regressions](../artifacts/logs/phase11_regression.log) · [Full-model trace accuracy](../artifacts/logs/phase11_model_accuracy.csv)

## Measured latency and memory

The raw full-checkpoint upload took 14.161 s with 29.30 GiB in accounted MLU allocations at peak, including the checkpoint and temporary LM-head tile. Peak host process RSS was 137.4 MiB for that test. The separate native generation runner initialized in 5.792 s in the repeated run. Its peak accounted device allocations were 23.19 GiB, including cached packed weights, and peak host RSS was 175.7 MiB. These device figures measure application allocations.

With tracing disabled, the repeat took 54.826 s from submitted prompt to first token, including lazy expert loading. Median inter-token latency was 511.977 ms, and measured decode throughput was 1.593 tokens/s across the following 31 steps. Including first-token latency, the 32-token generation averaged 0.431 tokens/s, excluding initial model setup. These are cold-cache request measurements and do not establish steady-state serving performance.
