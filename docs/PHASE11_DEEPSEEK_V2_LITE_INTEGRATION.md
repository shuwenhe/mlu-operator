# Phase 11 DeepSeek-V2-Lite integration

## Architecture and implementation order

1. Audit the safetensors index and every tensor's dtype and dimensions against the 27-layer configuration. Load the original tokenizer and exercise prompt encoding locally.
2. Connect normalized input, Phase 10 attention and its cache, the two rounded residuals, second RMSNorm and dense SwiGLU to form Layer 0. Verify each boundary on the physical device before using this block in a longer stack.
3. Layers 1–26 need checkpoint-specific FP32 routing logits, FP32 Softmax, unsorted top-six selection, expert scheduling, SwiGLU experts, FP32 weighted accumulation and a rounded sum with the shared expert output. The checkpoint has 64 routed experts of width 1,408 and a shared projection of width 2,816.
4. Integrate independent per-layer cache state, BF16 embedding lookup, final RMSNorm, the untied 102,400-token LM head, greedy argmax and EOS handling. Verify each layer, then prompt-to-token output and latency.

Frozen Phase 9.4 attention and Phase 10 MLA remain the numerical baseline.

## Implemented and tested

The BF16 runner, `runtime/native/phase11/model_run.cpp`, loads the real embedding and final norm, registers the checkpoint's attention projections per layer, runs the dense Layer 0 and all 26 MoE layers, keeps one native materialized KV cache per layer, and computes all 102,400 last-token logits and device argmax. Experts are packed lazily from the original shards and cached. The host reads integer expert IDs to schedule native kernels, and reads the chosen token for output and EOS handling. Model computation remains on device. The tokenizer runs on the host.

- **Dense Layer 0 PASS.** BF16 prefill lengths 1, 7, 32 and 128 each cover two fixture requests and a decode step. All 16,225,792 values across 160 stage comparisons matched checkpoint PyTorch bit-for-bit.
- **Isolated MoE suite PASS.** Every layer from 1 through 26 used actual checkpoint router and expert weights on one- and seven-row Layer 0 captures. Expert IDs and BF16 outputs matched. FP32 router scores and probabilities met tolerance in these cases.
- **Model boundary operators PASS.** Real embedding, final RMSNorm, LM head and argmax matched bit-for-bit for the one- and five-token isolated cases.
- **Prompt generation execution PASS.** The native runner executed one-token and 17-token prefill and continuous decode. All 32 generated IDs matched the independent checkpoint. Each layer was traced at attention, K/V and residual output for prefill and the first decode step. Full tensor parity failed.

## Numerical blocker

The first one-token divergence is Layer 26 with three BF16 decoder output values outside tolerance. It leads to 1,109 out-of-tolerance logits out of 102,400, with maximum absolute error 0.0625. On 17-token prefill, Layer 2 first differs at flattened output index 16179, native 0.1513671875 versus checkpoint 0.15234375. Downstream layer and cache contents then differ. At the logits, 89,636 values exceeded tolerance in prefill and 91,286 in the first decode step. Maximum absolute error was 0.375 in each. All compared values were finite.

In matched-input hardware replays of Layers 2 and 26, the expert IDs and the routed and shared expert outputs all matched bit-for-bit. The native router projection's FP32 scores differed from PyTorch. The selected probabilities differed by up to 1.49e-7 and 2.03e-6 respectively. Those differences were within tolerance at the router, but crossed BF16 rounding boundaries in the weighted output. Using identical checkpoint probabilities in a diagnostic control restored bitwise equality for both MoE outputs. This control is isolated from generation. Checkpoint-compatible FP32 routing arithmetic under integrated inputs remains the numerical acceptance blocker.

## Reproduction and protection

Use CNCC 5.7.6 and CNRT 7.7.3 targeting `mtp_592` on PCI `0000:4f:00.0`. Build with `runtime/native/phase11/build.sh OUT PHASE10_OBJECTS ACCEPTED_PACKED_OBJECTS`. Run `model_fixture.py CHECKPOINT META_DIR 'prompt text'`, then `OUT/model_run PCI CHECKPOINT META_DIR PHASE10_BF16_FIXTURES META_DIR/prompt.bin 32`. The optional final argument names a pre-created trace directory. Run `decode_tokens.py CHECKPOINT RUN_LOG` to render native IDs.

The original checkpoint reference uses `reference_model.py CHECKPOINT PROMPT_BIN STEPS [TRACE_DIR]` in the SDK Python with `TORCH_DEVICE_BACKEND_AUTOLOAD=0` and a writable `TORCHINDUCTOR_CACHE_DIR`. `compare_model.py` applies `1e-5 + 1e-5 * abs(reference)` at each traced boundary.

Phase 9.4 and Phase 10 were preserved. Frozen hashes, QK smoke, Softmax and AV checkpoint replay and dispatch, R1–R5, and all 28 Phase 10 configurations passed. Legacy R6 reproduced the expected failure byte-for-byte.

[End-to-end acceptance](DEEPSEEK_V2_LITE_E2E_ACCEPTANCE.md) · [Hardware log](../artifacts/logs/phase11_hardware.log) · [Regression log](../artifacts/logs/phase11_regression.log)

A second hardware control supplied the exact PyTorch FP32 router logits to the native Softmax, top-six and combine kernels. Both selected probabilities and final MoE outputs then matched bitwise. This localizes the first differing boundary to the FP32 router projection rather than the downstream Softmax, top-k, or expert reduction on these inputs.
