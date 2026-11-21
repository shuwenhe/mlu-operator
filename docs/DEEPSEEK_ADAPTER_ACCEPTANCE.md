# DeepSeek-V2-Lite adapter hardware acceptance

**PASS for the tested Layer 0 projection and connected KV adapter paths. Layer 0 PyTorch parity remains FAIL.**

The executable compiled with warnings treated as errors and ran on authorized MLU590-M9 PCI `0000:4f:00.0`, CNRT ordinal 0 inside the one-card SDK container. The existing objects target `mtp_592` and were built with CNCC 5.7.6. The pinned Ubuntu 24.04 SDK image provides CNRT 7.7.3 and CNNL 2.2.18. Host driver is 6.5.52.

The fixture uses token 100000's real embedding and actual Layer 0 weights from the local safetensors checkpoint. Each projection runs a full output width for one row. Each GEMM comparison uses an independent CPU dot product over the original checkpoint layout, with FP64 accumulation and an FP32 result. Testing uses the existing `1e-5 + 1e-5 * abs(reference)` tolerance. Native and CNNL are explicitly invoked in distinct runs with separately prepared weight representations.

| Hardware check | BF16 | FP16 |
| --- | --- | --- |
| Q via Native and CNNL, 3,072 outputs per backend | PASS, max abs 1.1921e-7 | PASS, max abs 1.1921e-7 |
| O, Gate and Up via Native and CNNL | PASS, max abs 1.1921e-7 | PASS, max abs 1.1921e-7 |
| Native input RMSNorm and KV RMSNorm | PASS, zero error | PASS, zero error |
| Native connected KV, 4,096 final outputs | PASS, max abs 1.1921e-7 | PASS, max abs 3.7253e-6 |
| CNNL KV-A and KV-B on matched intermediate inputs | PASS, max abs 1.9074e-6 | PASS, max abs 4.7684e-7 |
| Native YaRN at positions 7 and 1 | PASS, all 2,176 words match | PASS, all 2,176 words match |
| Reject padded projection stride and unsupported CNNL connected KV route | PASS | PASS |
| Tracked device buffers and packed weights after close | 0 bytes | 0 bytes |

Every numeric comparison reported zero mismatches and zero nonfinite values. The native latent cast additionally matched all 512 expected low-precision words. RoPE was exercised separately with checkpoint-derived coefficients and embedding-derived inputs. CNNL RMSNorm is exposed by the adapter and was outside this executable's comparison matrix. The standalone O, Gate and Up cases consume an embedding and compare FP32 projection outputs. The preserved Layer 0 fixture uses the actual stage activations and tests subsequent low-precision rounding, FFN and residual paths.

## Reproduce

Run from the workspace root with the previously validated SDK runner, model path and accepted object directories. Use an authorized physical card matching the PCI argument.

```bash
out=$(cat artifacts/logs/mlu8_results_path.log)
stage=$(mktemp -d "$out/adapter-validation.XXXXXX")
PYTHONDONTWRITEBYTECODE=1 python3 integration/deepseek/adapter/checkpoint.py \
  /app/shuwen/model/DeepSeek-V2-Lite "$stage/checkpoint"
data="/results/${stage##*/}/checkpoint"
bash "$out/run_sdk.sh" 0 "
  export LD_LIBRARY_PATH=/usr/local/neuware/lib64
  bash integration/deepseek/adapter/build.sh /results
  for mode in q kv projections; do
    for dtype in bf16 fp16; do
      /results/deepseek_adapter --authorized 0000:4f:00.0 '$data' \"\$mode\" \"\$dtype\"
    done
  done"
```

- [Hardware execution log](../artifacts/logs/deepseek_adapter_hardware.log)
- [Checkpoint tensor names, shapes and shard offsets](../artifacts/logs/deepseek_adapter_checkpoint.log)
- [Layer 0 regression log](../artifacts/logs/deepseek_adapter_layer0_regression.log)
- [Post-test CNMON inventory](../artifacts/logs/deepseek_adapter_post_cnmon.log)
- [Existing Layer 0 numerical divergence](LAYER0_RESIDUAL_ROOT_CAUSE.md)
