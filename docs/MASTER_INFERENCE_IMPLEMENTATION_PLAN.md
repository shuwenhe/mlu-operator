# DeepSeek-V2-Lite native inference implementation plan

## Repository audit

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

## Implementation sequence

1. Add a standalone native QK launcher on the caller's queue. Use head-major materialized Q/K and a separate key capacity stride. Accumulate in FP32, round the matmul result to the input dtype, multiply by the configured YaRN scale, then round again and apply the causal mask. The local checkpoint's eager forward defines these two rounding points. Preserve raw FP32 dot products for diagnostics. Start with a scalar FP32 reduction over 192 products and measure parity before changing reduction order.
2. Compile for `mtp_592`. Test both dtypes with inputs that distinguish one rounding from two, and cover multiple heads, cache stride, masking and decode offsets. Replay materialized Layer 0 Q and K against the original checkpoint PyTorch path at `1e-5 + 1e-5 * abs(reference)`. Record mismatches, relative and absolute errors, bit differences, and unexpected nonfinite values separately from expected causal negative infinities.
3. After QK passes on MLU590-M9, implement stable FP32 Softmax and native AV. Verify each low-precision boundary. Keep the eager materialized MLA layout and independently validate any absorbed variant. Retain explicit CPU reference attention for validation and wire native attention without fallback.
4. Add per-request device cache ownership, queue-ordered append, committed length, rollback, cancellation and deferred reclaim. Validate interleaved prefill and decode and test concurrent request isolation.
5. Reuse projection and norm components for Layer 0 dense FFN and then layers 1–26. Validate the checkpoint's FP32 router, top-6 of 64, and two shared experts stage by stage. Connect validated device embedding, final RMSNorm, untied LM head, greedy decoding and EOS handling, followed by sampling.
6. After parity, measure loading, packing, TTFT, TPOT, throughput, peak device memory, operator latency and H2D costs. Compare against a compatible baseline on identical hardware and generation settings.

## Current hardware gate

The checkpoint source, config and four safetensors shards are available locally. The default shell lacks the compiler and device nodes. The historical Phase 8.2 SDK image compiled both QK variants for `mtp_592` and ran them with MLU590-M9 card 0 exposed.

Build with `bash runtime/native/phase9/build_qk.sh /tmp/qk` inside the SDK image. Run `/tmp/qk/qk_device_test 0` for the synthetic cases. Run `runtime/native/phase9/replay_qk.py` with the device binary, checkpoint config, and the Phase 8.2 `*attention_audit.npz` captures using `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for the CPU reference.

Four synthetic cases passed with zero mismatches. Replay of the captured Layer 0 Q/K tensors found 195 scaled-score and 97 masked-score mismatches beyond tolerance across 1,056,832 values per stage. All causal masks matched and there were no unexpected nonfinite values. The FP16 replay uses converted BF16 captures. The raw dot diagnostic also differs from PyTorch, consistent with the historical evidence that FP32 reduction order affects low-precision rounding. Native QK acceptance is open. The next acceptance run needs a checkpoint-compatible reduction and independently captured FP16 inputs. The software and hardware results are recorded in [the QK run log](../artifacts/logs/phase9_native_qk_hardware.log).
