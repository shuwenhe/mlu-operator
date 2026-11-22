# Phase 12.5.4 Root Cause Report — Native KV Cache Numerical Investigation

## Environment

- Repo: `/app/shuwen/mlu-operator`
- Vendor image: `teleavatar-mlu:v0.1`
- Python: `/torch/venv3/pytorch_infer/bin/python3`
- Hardware: `MLU590-M9 card0`
- Model: `/app/shuwen/model/DeepSeek-V2-Lite`

## P0 preservation

- Pre-run HEAD/status and source+fixture hashes were recorded in:
  - `artifacts/logs/phase12_5_4_source_hashes.txt`
- Historical fixtures (`phase11_5`, `phase12_3`, `phase12_4`, `phase12_5 reference*`) were only read, not modified.

## First divergence (experimental)

Using the frozen references and real vendor run:

1. **Initial baseline (before fix)**: first failing tensor is `request0 / 0-layer0-key.bin`.
2. Mismatch localization showed:
   - key `nope` slice `[0:128]`: no mismatch
   - key `rope` slice `[128:192]`: all mismatch entries
3. `step0` still had `attention/logits PASS` because prefill length was 1 (single-token softmax degenerates to 1.0).
4. At `step1`, divergence propagated to attention/logits.

Conclusion: the first true divergence appears in the **rope segment of K-path**, before KV cache readback becomes a confounder.

## KV write/read investigation conclusion

- Added optional diagnostic dumps for write inputs (`NATIVE_DEEPSEEK_DUMP_WRITE=1`):
  - `layer{i}-key_write.bin`, `layer{i}-value_write.bin`
- For layers `0,1,2,13,26`, write-tail vs read-tail checks are all PASS:
  - `artifacts/logs/phase12_5_4_kv_write_accuracy.csv`
- Readback vs historical reference for the same layers are all PASS:
  - `artifacts/logs/phase12_5_4_kv_read_accuracy.csv`

This rules out KV cache scatter/gather as the first divergence root cause in this incident.

## Root cause

Two fixture-construction defects in runtime setup:

1. **RoPE table generation mismatch** (did not use checkpoint’s YARN rotary implementation).
2. **Attention softmax scale mismatch** (used `1/sqrt(192)` only, missing YARN `mscale_all_dim` factor).

These setup errors caused key rope-segment mismatch first, then attention/logits mismatch at decode.

## Fix (smallest surgical change)

- Added `integration/vllm_native_mlu/phase12_5/prepare_rope.py` to generate:
  - `cos.bin`, `sin.bin` from checkpoint `DeepseekV2YarnRotaryEmbedding`
  - `scale.txt` with YARN-adjusted softmax scale
- Added optional diagnostics in `integration/vllm_native_mlu/phase12_5/model.py`:
  - `NATIVE_DEEPSEEK_DUMP_WRITE=1` gated write-input dumps only.

No kernel source change and no package/environment upgrade.

## Regression after fix

- Full 27-layer run against frozen references:
  - `reference tolerance failures=0`
  - `phase12_5_4_layer_accuracy.csv`: all PASS
  - `phase12_5_4_logits_accuracy.csv`: all PASS
  - `phase12_5_4_token_ids.csv`: `20/20` greedy parity
- Component regressions:
  - router/rmsnorm, attention chain, phase12_3 metadata adapter, phase12_4 cache+projection all PASS.

Raw logs:
- `artifacts/logs/phase12_5_4_hardware.log`
