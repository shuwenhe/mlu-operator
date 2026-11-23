# DeepSeek-V2-Lite standalone native inference

- [DeepSeek-V2-Lite Layer 0 operator mapping](#deepseek-operator-mapping)
- [DeepSeek-V2-Lite native inference implementation plan](#master-inference-implementation-plan)
- [Phase 11 DeepSeek-V2-Lite integration](#phase11-deepseek-v2-lite-integration)
- [DeepSeek-V2-Lite end-to-end native MLU acceptance](#deepseek-v2-lite-e2e-acceptance)
- [Phase 11.5 FP32 router numerical acceptance](#phase11-5-router-numerical-acceptance)
- [Phase 5 MLU integration](#phase-5-mlu-integration)

<a id="deepseek-operator-mapping"></a>

## DeepSeek-V2-Lite Layer 0 operator mapping

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

<a id="master-inference-implementation-plan"></a>

## DeepSeek-V2-Lite native inference implementation plan

### Repository audit

| Component | Source and build | Recorded execution and acceptance |
| --- | --- | --- |
| FP16/BF16 GEMM, matrix GEMM, packed weights | BANG C under `operators/gemm/native` and its build scripts | Historical build and hardware logs cover scalar, matrix and prepacked paths. Phase 8.2 reports zero differing projection output bits in 252 matched-input comparisons. |
| RMSNorm | BANG C under `operators/rmsnorm/native` and the Phase 8.2 experimental reduction under `runtime/native/phase8` | Frozen RMSNorm has independent hardware logs. Phase 8.2 Layer 0 uses a separate experimental device reduction and rounded rsqrt path. Its earlier failure fixtures are retained. |
| RoPE and KV projection layout | `runtime/native/rope.mlu` and `integration/deepseek/phase5` | Historical RoPE and KV-chain runs are recorded in `artifacts/logs`. |
| CNNL | `operators/{gemm,rmsnorm}/src`, plus the runtime dispatch | Separate CNNL implementations and recorded GEMM hardware runs. The CMake SDK targets build these wrappers. |
| CPU references | `common/reference`, Phase 8.2 checkpoint arithmetic, and `runtime/native/phase9/attention.cpp` | Phase 8.2 reports 0/576 stage comparisons and 0/1,421,312 final values outside tolerance with explicit ATen CPU attention. Keep the checkpoint and legacy C++ oracle comparisons separate. |
| Phase 9 | QK, Softmax and AV host helpers, draft validators and throwing dispatch placeholders | Phase 9 completion and acceptance documents describe design and planned tests. The audited build does not link a Phase 9 device kernel. |
| Runtime and checkpoint adapter | Worker, PCI registry, queue, workspace and packed-weight ownership under `runtime/native`. Checkpoint adapter and Layer 0 fixtures under `integration/deepseek`. | Historical logs cover registry, workspace, projections and Layer 0. Production cache transactions, full decoder, MoE, final norm, LM head and generation still require implementation. |

Frozen source baseline (SHA-256):
```
dc7e382fb20d8d15d316b2b299d06bfb9625eb1ff5ead5f2ad1b438fb72e9722  operators/gemm/native/matrix/matrix_gemm.mlu
6d1975f4c60ca0bb217e562f75eb2ccbbfe32a1989c57a96860af00b84654630  operators/rmsnorm/native/rmsnorm.mlu
b4d940c87d2861196e98c62d6a81aac7e4edbb11caad4a27e8265dff86fe89d6  integration/deepseek/phase5/rope.mlu
063bdceb010dd20e1489113b6298a6507b2a68861bf8fae25278baf17c3213b3  integration/deepseek/phase5/mla_check.py
2254f8be9aa0beb1491e07890d61ab4a4bab94146ac2327821e52e6dc1e9e209  runtime/native/rope.mlu
24257528c95315ef7bed23e0c4a34f2cb469db66305461e29e531d2464ed188c  operators/gemm/native/gemm_fp16.mlu
109d1a82e3cb9981ead5c9a1fd5928ea27fa732d8771dad3776ccf3265ae7ac8  operators/gemm/native/gemm_bf16.mlu
20362b55ed47eecbe3d4a4fde07179b87a06dc5752cf0a26c6d60ea4f2c3bed9  operators/gemm/native/prepacked/packed_weight.mlu
```

### Implementation sequence

1. Add a standalone native QK launcher on the caller's queue. Use head-major materialized Q/K and a separate key capacity stride. Accumulate in FP32, round the matmul result to the input dtype, multiply by the configured YaRN scale, then round again and apply the causal mask. The local checkpoint's eager forward defines these two rounding points. Preserve raw FP32 dot products for diagnostics. Start with a scalar FP32 reduction over 192 products and measure parity before changing reduction order.
2. Compile for `mtp_592`. Test both dtypes with inputs that distinguish one rounding from two, and cover multiple heads, cache stride, masking and decode offsets. Replay materialized Layer 0 Q and K against the original checkpoint PyTorch path at `1e-5 + 1e-5 * abs(reference)`. Record mismatches, relative and absolute errors, bit differences, and unexpected nonfinite values separately from expected causal negative infinities.
3. After QK passes on MLU590-M9, implement stable FP32 Softmax and native AV. Verify each low-precision boundary. Keep the eager materialized MLA layout and independently validate any absorbed variant. Retain explicit CPU reference attention for validation and wire native attention without fallback.
4. Add per-request device cache ownership, queue-ordered append, committed length, rollback, cancellation and deferred reclaim. Validate interleaved prefill and decode and test concurrent request isolation.
5. Reuse projection and norm components for Layer 0 dense FFN and then layers 1–26. Validate the checkpoint's FP32 router, top-6 of 64, and two shared experts stage by stage. Connect validated device embedding, final RMSNorm, untied LM head, greedy decoding and EOS handling, followed by sampling.
6. After parity, measure loading, packing, TTFT, TPOT, throughput, peak device memory, operator latency and H2D costs. Compare against a compatible baseline on identical hardware and generation settings.

### Current hardware gate

The checkpoint source, config and four safetensors shards are available locally. The default shell lacks the compiler and device nodes. The historical Phase 8.2 SDK image compiled both QK variants for `mtp_592` and ran them with MLU590-M9 card 0 exposed.

Build with `bash runtime/native/phase9/build_qk.sh /tmp/qk` inside the SDK image. Run `/tmp/qk/qk_device_test 0` for the synthetic cases. Run `runtime/native/phase9/replay_qk.py` with the device binary, checkpoint config, and the Phase 8.2 `*attention_audit.npz` captures using `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for the CPU reference.

Four synthetic cases passed with zero mismatches. Replay of the captured Layer 0 Q/K tensors found 195 scaled-score and 97 masked-score mismatches beyond tolerance across 1,056,832 values per stage. All causal masks matched and there were no unexpected nonfinite values. The FP16 replay uses converted BF16 captures. The raw dot diagnostic also differs from PyTorch, consistent with the historical evidence that FP32 reduction order affects low-precision rounding. Native QK acceptance is open. The next acceptance run needs a checkpoint-compatible reduction and independently captured FP16 inputs. The software and hardware results are recorded in [the QK run log](../artifacts/logs/phase9_native_qk_hardware.log).

<a id="phase11-deepseek-v2-lite-integration"></a>

## Phase 11 DeepSeek-V2-Lite integration

### Architecture and implementation order

1. Audit the safetensors index and every tensor's dtype and dimensions against the 27-layer configuration. Load the original tokenizer and exercise prompt encoding locally.
2. Connect normalized input, Phase 10 attention and its cache, the two rounded residuals, second RMSNorm and dense SwiGLU to form Layer 0. Verify each boundary on the physical device before using this block in a longer stack.
3. Layers 1–26 need checkpoint-specific FP32 routing logits, FP32 Softmax, unsorted top-six selection, expert scheduling, SwiGLU experts, FP32 weighted accumulation and a rounded sum with the shared expert output. The checkpoint has 64 routed experts of width 1,408 and a shared projection of width 2,816.
4. Integrate independent per-layer cache state, BF16 embedding lookup, final RMSNorm, the untied 102,400-token LM head, greedy argmax and EOS handling. Verify each layer, then prompt-to-token output and latency.

Frozen Phase 9.4 attention and Phase 10 MLA remain the numerical baseline.

### Implemented and tested

The BF16 runner, `runtime/native/phase11/model_run.cpp`, loads the real embedding and final norm, registers the checkpoint's attention projections per layer, runs the dense Layer 0 and all 26 MoE layers, keeps one native materialized KV cache per layer, and computes all 102,400 last-token logits and device argmax. Experts are packed lazily from the original shards and cached. The host reads integer expert IDs to schedule native kernels, and reads the chosen token for output and EOS handling. Model computation remains on device. The tokenizer runs on the host.

- **Dense Layer 0 PASS.** BF16 prefill lengths 1, 7, 32 and 128 each cover two fixture requests and a decode step. All 16,225,792 values across 160 stage comparisons matched checkpoint PyTorch bit-for-bit.
- **Isolated MoE suite PASS.** Every layer from 1 through 26 used actual checkpoint router and expert weights on one- and seven-row Layer 0 captures. Expert IDs and BF16 outputs matched. FP32 router scores and probabilities met tolerance in these cases.
- **Model boundary operators PASS.** Real embedding, final RMSNorm, LM head and argmax matched bit-for-bit for the one- and five-token isolated cases.
- **Prompt generation execution PASS.** The native runner executed one-token and 17-token prefill and continuous decode. All 32 generated IDs matched the independent checkpoint. Each layer was traced at attention, K/V and residual output for prefill and the first decode step. Full tensor parity failed.

### Numerical blocker

The first one-token divergence is Layer 26 with three BF16 decoder output values outside tolerance. It leads to 1,109 out-of-tolerance logits out of 102,400, with maximum absolute error 0.0625. On 17-token prefill, Layer 2 first differs at flattened output index 16179, native 0.1513671875 versus checkpoint 0.15234375. Downstream layer and cache contents then differ. At the logits, 89,636 values exceeded tolerance in prefill and 91,286 in the first decode step. Maximum absolute error was 0.375 in each. All compared values were finite.

In matched-input hardware replays of Layers 2 and 26, the expert IDs and the routed and shared expert outputs all matched bit-for-bit. The native router projection's FP32 scores differed from PyTorch. The selected probabilities differed by up to 1.49e-7 and 2.03e-6 respectively. Those differences were within tolerance at the router, but crossed BF16 rounding boundaries in the weighted output. Using identical checkpoint probabilities in a diagnostic control restored bitwise equality for both MoE outputs. This control is isolated from generation. Checkpoint-compatible FP32 routing arithmetic under integrated inputs remains the numerical acceptance blocker.

### Reproduction and protection

Use CNCC 5.7.6 and CNRT 7.7.3 targeting `mtp_592` on PCI `0000:4f:00.0`. Build with `runtime/native/phase11/build.sh OUT PHASE10_OBJECTS ACCEPTED_PACKED_OBJECTS`. Run `model_fixture.py CHECKPOINT META_DIR 'prompt text'`, then `OUT/model_run PCI CHECKPOINT META_DIR PHASE10_BF16_FIXTURES META_DIR/prompt.bin 32`. The optional final argument names a pre-created trace directory. Run `decode_tokens.py CHECKPOINT RUN_LOG` to render native IDs.

The original checkpoint reference uses `reference_model.py CHECKPOINT PROMPT_BIN STEPS [TRACE_DIR]` in the SDK Python with `TORCH_DEVICE_BACKEND_AUTOLOAD=0` and a writable `TORCHINDUCTOR_CACHE_DIR`. `compare_model.py` applies `1e-5 + 1e-5 * abs(reference)` at each traced boundary.

Phase 9.4 and Phase 10 were preserved. Frozen hashes, QK smoke, Softmax and AV checkpoint replay and dispatch, R1–R5, and all 28 Phase 10 configurations passed. Legacy R6 reproduced the expected failure byte-for-byte.

[End-to-end acceptance](#deepseek-v2-lite-e2e-acceptance) · [Hardware log](../artifacts/logs/phase11_hardware.log) · [Regression log](../artifacts/logs/phase11_regression.log)

A second hardware control supplied the exact PyTorch FP32 router logits to the native Softmax, top-six and combine kernels. Both selected probabilities and final MoE outputs then matched bitwise. This localizes the first differing boundary to the FP32 router projection rather than the downstream Softmax, top-k, or expert reduction on these inputs.

<a id="deepseek-v2-lite-e2e-acceptance"></a>

## DeepSeek-V2-Lite end-to-end native MLU acceptance

**Native BF16 generation executed successfully. Overall acceptance remains PARTIAL because the full forward pass failed the existing numerical tolerance.** The original local checkpoint ran through all 27 decoder layers on MLU590-M9 card 0 at PCI `0000:4f:00.0`. Two native runs generated 32 greedy tokens each from a 17-token prompt. All 32 IDs in each run matched the original checkpoint under independent CPU PyTorch.

| Area | Decision |
| --- | --- |
| A. Native operators | PASS on the recorded isolated matrix for embedding, GEMM, RMSNorm, YaRN, MLA and KV cache, QK, Softmax, AV, dense SwiGLU, MoE routing/experts, vocabulary projection and argmax. FP32 router projection parity remains open on integrated inputs. |
| B. Full loading and forward | PASS for actual BF16 checkpoint residency and execution of all 27 layers, on one-token and multi-token prefill and repeated decode. FAIL for full tensor numerical parity. |
| C. Prompt-to-token generation | PASS for native execution and all 32 greedy IDs on the tested prompt. Full numerical acceptance and general model support remain open. |

### Verified checkpoint and execution path

The index and four safetensors shards contain 5,291 BF16 tensors and 31,412,968,448 payload bytes. The audit matched every tensor name and shape. The tokenizer contains 100,002 entries within the model's 102,400-row vocabulary. The checkpoint has 16 attention heads, hidden width 2,048, an untied embedding and LM head, one dense layer with width 10,944, and 26 MoE layers. Each MoE layer routes six of 64 width-1,408 experts and also evaluates the width-2,816 shared branch.

Layer 0's full dense block matched bitwise across 160 stage comparisons. The isolated BF16 suite exercised all 26 real MoE layers using shared representative one- and seven-row inputs. The integrated runner packs expert weights lazily and caches the packed weights. It uses a device-resident MLA cache per layer. The host handles tokenizer encoding, queue scheduling from integer expert IDs, and token emission. No model computation fell back to CPU, PyTorch or CNNL. The independent reference uses the original checkpoint layer classes and separate cache state.

### Forward-pass parity

The acceptance threshold was `1e-5 + 1e-5 * abs(reference)`. The comparison used 330 tensors spanning embedding, per-layer attention output, committed K/V, decoder output and final vocabulary logits. The one-token and multi-token traces used the same inputs. The first decode token also matched, so its trace used identical token IDs across implementations.

| Case | First differing layer | First-layer out-of-tolerance values | Out-of-tolerance logits / 102,400 | Logit max abs |
| --- | --- | ---: | ---: | ---: |
| One-token prefill | 26 | 3 | 1,109 | 0.0625 |
| 17-token prefill | 2 | 1 | 89,636 | 0.375 |
| First incremental decode | 3, following the prefill divergence | See trace CSV | 91,286 | 0.375 |

Embedding matched bit-for-bit. No compared tensor had unexpected nonfinite values. Matched-input replays identified the FP32 MoE router projection as the first differing operation in the failing Layer 2 and Layer 26 cases. Expert IDs, dispatched expert outputs and the shared branch agreed. The router score differences crossed BF16 rounding boundaries during weighted aggregation. Feeding exact checkpoint logits in an isolated diagnostic restored bitwise equality through device Softmax, top-six and expert reduction. This diagnostic is outside the native generation path.

### Source protection and performance scope

Frozen QK, Softmax, AV and Phase 10 MLA hashes verified. The attention and cache regressions passed. Legacy R6 reproduced its known parity failure byte-for-byte. Runtime scope is BF16, batch one, checkpoint-fixed head dimensions and an existing 131-position rotary table. The measured generation reached 48 committed tokens. Other prompts, longer context, full-model FP16, batch serving and early-EOS termination were not accepted by these tests.

[Phase 11 implementation, numerical diagnosis and reproduction](#phase11-deepseek-v2-lite-integration) · [Hardware log](../artifacts/logs/phase11_hardware.log) · [Regressions](../artifacts/logs/phase11_regression.log) · [Full-model trace accuracy](../artifacts/logs/phase11_model_accuracy.csv)

### Measured latency and memory

The raw full-checkpoint upload took 14.161 s with 29.30 GiB in accounted MLU allocations at peak, including the checkpoint and temporary LM-head tile. Peak host process RSS was 137.4 MiB for that test. The separate native generation runner initialized in 5.792 s in the repeated run. Its peak accounted device allocations were 23.19 GiB, including cached packed weights, and peak host RSS was 175.7 MiB. These device figures measure application allocations.

With tracing disabled, the repeat took 54.826 s from submitted prompt to first token, including lazy expert loading. Median inter-token latency was 511.977 ms, and measured decode throughput was 1.593 tokens/s across the following 31 steps. Including first-token latency, the 32-token generation averaged 0.431 tokens/s, excluding initial model setup. These are cold-cache request measurements and do not establish steady-state serving performance.

<a id="phase11-5-router-numerical-acceptance"></a>

## Phase 11.5 FP32 router numerical acceptance

### Integrated baseline

The standalone model invokes `Model::run_moe` in `runtime/native/phase11/model_run.cpp`, which calls `project` and the linked `phase8_candidate_bf16` symbol from the Phase 10 `projection-bf16.o`. The isolated MoE harness calls the same symbol separately. In the model, each input row is the BF16 post-attention RMSNorm output. The logical activation matrix is `[M,2048]`, contiguous with strides `[2048,1]`. The checkpoint gate weight is BF16 `[64,2048]`, also contiguous with strides `[2048,1]`. The baseline transposes the gate weight before storing 128-by-128 packed tiles and pads the expert dimension from 64 to 128. The output buffer is contiguous FP32 `[M,64]` with strides `[64,1]`.

The compiled projection transposes each RHS tile, widens both operands to FP32, multiplies them, and accumulates 64 interleaved lanes. It folds those lanes in tiers of eight and then 4, 2 and 1. The router forwards its FP32 logits directly to the FP32 probability output of native Softmax. The original and candidate builds use CNCC `--bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off`, with the original projection additionally compiled using `-DPHASE8_BF16`. Host linking uses g++ `-O2 -ffp-contract=off`.

### Root cause and reproduction

The pinned independent PyTorch checkpoint widens BF16 activations and weights before calling FP32 linear. Profiling on the acceptance host identified oneMKL 2024.0 Update 2 generic CPU kernels with PyTorch using two threads. Different row counts use distinct addition sequences:

| Shape | Reference sequence |
|---|---|
| One row and packed SGEMM fringe rows | Accumulate products at features 0–3 in order into lane 0. Interleave the remaining products across four lanes and fold as `(lane0 + lane2) + (lane1 + lane3)`. |
| Two to seven rows | Accumulate all products in four interleaved lanes starting at feature 0 and use the same final fold. The sampled function was `mkl_blas_def_sgemm_pst`. |
| Eight to 128 rows | The sampled `mkl_blas_def_sgemm_inner` processes complete groups of four rows. Each dot sums the 32-feature tail panel first, then adds eighteen independent serial 112-feature sums in increasing feature order. Remaining rows use GEMV. The pinned two-thread row partition above 64 rows splits the range at `floor(M/2)`, giving each partition its own fringe. |

The independent device probe exported every product and each strategy's partial sums to `artifacts/build/phase11_5/layer*-m*/hardware-mode*.npz`. Products and partials matched the explicit host controls bit-for-bit. The original router reduction differed by 53/64 bits for the single row taken from the Layer 2 capture, 806/1,088 bits for Layer 2 17-token prefill and 49/64 bits for Layer 26 one-token prefill. The Layer 2 single-row case uses the first actual 17-token activation with a freshly calculated one-row PyTorch reference. The integrated one-token run also traces Layer 2. Every original router score was within tolerance. Small probability changes crossed BF16 rounding boundaries during expert aggregation. In the 17-token case the baseline first differed at row 0, expert 0, with native `0x3e326e4e` versus checkpoint `0x3e326e56`. Maximum absolute and relative errors and signed-order ULP distances for all tested strategies are in the CSV.

The isolated mixed-order CNRT probe matched all 1,216 logits bit-for-bit. Its queue-plus-synchronization timings, including writes of diagnostic products and partials, were approximately 0.16 ms for the one-row cases and 1.00 ms for 17 rows in this run. These are diagnostic timings with extra memory traffic. The 64-lane probe is an arithmetic control. Phase 11 standalone timings are in the same hardware log. FP64 followed by a single FP32 rounding still differed from the checkpoint on 794 of 1,088 logits for the 17-token case.

### Router-only build

The router device source is `runtime/native/phase11_5/router.mlu`. Build the alternate executable using `build_candidate.sh`, which patches copies of the model runner and MoE harness in the output directory. The baseline runner and its linked attention, projection, normalization and expert objects remain available for regression. The alternate call site uploads the original 64-by-2048 BF16 gate weight in checkpoint order. Expert selection, dispatch, expert weights and expert combine retain their existing implementations. The kernel performs FP32 accumulation on the device and has no CPU path.

The shape sweep uses repeated captured Layer 2 activation rows and the actual gate weight. It validates each supported row count from 1 through 128 on the physical card. Full model fixtures use the original prompt token IDs and original checkpoint, independent eager CPU PyTorch, and the existing `1e-5 + 1e-5 * abs(reference)` threshold.

### Hardware acceptance

**PASS for the standalone Phase 11.5 router build on MLU590-M9 card 0.** The frozen Phase 11 executable remains available as the independent baseline.

| Check | Result |
|---|---|
| Actual Layer 2 and Layer 26 router captures | 1,216 logits, zero bit mismatches |
| Supported router lengths 1–128 | 528,384 logits, zero bit mismatches |
| Layer 2 one-token and 17-token prefill, Layer 26 one-token prefill, first decode | Bitwise checkpoint equality at each traced boundary |
| All 26 isolated MoE layers | PASS at one and seven rows. Both matched-input diagnostics also passed, with bitwise MoE output equality. |
| Both 32-token generations | All 32 IDs per run matched the independent PyTorch checkpoint |
| Full traced tensors, including vocabulary logits | 152,806,400 values over 3,630 tensors, zero bit mismatches and zero unexpected nonfinite values. All 3,379,200 logits in 33 complete vocabulary vectors matched bit-for-bit. |
| Frozen sources and fixtures | PASS against historical SHA-256 manifests |

The build is pinned to the tested reference's oneMKL dispatch and two-thread settings. Revalidate its arithmetic contract when changing the reference CPU, library or thread count. The separate vLLM probes establish component execution on the vendor MLU platform. The native vLLM model and scheduler integration remain pending.

### Reproduction

Use the pinned SDK image and the mounts listed in the vLLM compatibility audit, plus Phase 10 at `/phase10`. Mount the Phase 11.5 output at `/p115` with its captured operands. Export the cache settings from that audit and run from the repository root.

```bash
py=/torch/venv3/pytorch_infer/bin/python3
"$py" runtime/native/phase11/reference_model.py /checkpoint /phase11/model/prompt.bin 32 /p115/reference-full > /p115/reference-full.log
bash runtime/native/phase11_5/run_candidate.sh /p115/final /checkpoint /phase10 /phase11 \
    artifacts/build/mlu8-bringup.28JR1P/prepacked-final-build /p115 "$py" 0000:4f:00.0
```

[Accuracy CSV](../artifacts/logs/phase11_5_router_accuracy.csv) · [Hardware execution](../artifacts/logs/phase11_5_router_hardware.log) · [Source verification](../artifacts/logs/phase11_5_source_hashes.txt) · [Runtime architecture](VLLM_MLU_BACKEND_ARCHITECTURE.md#inference-runtime-architecture-audit)

<a id="phase-5-mlu-integration"></a>

## DeepSeek-V2-Lite MLU integration, phase 5

### Checkpoint contract

The local checkpoint config and model source define 27 layers with hidden width 2048, 16 heads, BF16 storage and untied 102,400-row embedding and logits tables. The local index and safetensors headers supply the shapes below. Linear weights are contiguous output-by-input `[N,K]), element strides `[K,1]`. Standalone GEMM uses contiguous A[M,K], B[K,N] and output C[M,N], with element strides `[K,1]`, `[N,1]`, and `[N,1]`. The host harness transposes checkpoint weights to B before device packing; production must preserve these logical layouts or introduce a tested transposed packer. FP32 results need an explicit activation-dtype cast at model boundaries.

| Model path | Checkpoint weight / activation contract |
| --- | --- |
| Q projection | `[3072,2048]`; output [B,S,16,192]. No Q LoRA rank. Split per head into nonrotary 128 and rotary 64. |
| KV-A and latent norm | `[576,2048]`; contiguous output row stride 576; latent columns 0–511 and shared rotary key columns 512–575. RMSNorm gamma `[512]`. Compact the strided latent slice before native RMSNorm. |
| KV-B and output projection | KV-B `[4096,512]`; output view [B,S,16,256], split key-noPE 128 and V 128. Output projection `[2048,2048]` follows attention concatenation of all V heads. |
| Dense FFN, layer 0 | Gate/up `[10944,2048]`, down `[2048,10944]`; SiLU(gate) × up. |
| MoE, layers 1–26 | Routed gate/up `[1408,2048]`, down `[2048,1408]`. Shared width 2816, two shared experts. 64 routed experts and six selected per token. Router `[64,2048]` is used as FP32 with softmax/greedy TopK, without top-k renormalization. Keep routing precision independent of low-precision GEMM. |
| Embedding, norms and LM head | Embedding and LM head `[102400,2048]`. Hidden RMSNorm vectors `[2048]`; epsilon 1e-6. Model RMSNorm rounds normalized values before gamma. |

MLA views, cache strides, masks and YaRN scale are in the [MLA design](NATIVE_ATTENTION.md#mlu-attention-mla-design).

### Execution evidence and phase-6 status

- Checkpoint-backed layer-zero Q, KV-A and KV-B projections passed on MLU 2 for FP16 and BF16, including all requested token counts. This compares BANG GEMM and CNNL as independent data paths.
- A connected native subgraph on MLU 2 kept persistent KV-A and KV-B weights and intermediates on device. It ran KV-A → low-precision cast and compact → native DeepSeek-order RMSNorm → KV-B for M=1 and 32. It copied back only the final tensor for validation, and all four combinations passed.
- Native YaRN rotation passed on MLU 4 against the exact checkpoint PyTorch CPU function. Standalone MLA on MLU 5 passed FP32 materialized and absorbed formulations, with one FP16 model-order discrepancy retained as a failed parity gate. No inference service was started.
- **End-to-end acceptance is incomplete.** The connected subgraph does not execute all decoder layers, attention/cache updates, MoE dispatch, logits and sampling as a native graph. Production routing and cache ownership have not been integrated. TTFT, TPOT, tokens/s, prefill/decode throughput, peak memory, multi-card inference, and a matched vLLM comparison remain unmeasured. No end-to-end result is inferred from operator timings.

[Projection acceptance](NATIVE_ATTENTION.md#mlu-qkv-projection-acceptance) · [RoPE acceptance](NATIVE_ATTENTION.md#mlu-rope-acceptance) · [Runtime design](MLU_RUNTIME_IMPLEMENTATION.md#mlu-runtime-design)
