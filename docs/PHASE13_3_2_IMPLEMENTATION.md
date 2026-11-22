# Phase 13.3.2 — Native MLA projections

Use an isolated projection module and rank-local safetensors slice reads. Reuse Phase 13.2 Column, LocalGemm and Row plus the frozen NativeRMSNorm. Preserve FP32 Q and KV outputs, BF16 latent narrowing before native RMSNorm, and FP32 Row SUM before one BF16 cast and residual.

1. Verify shard bounds and identity for each milestone.
2. Capture supplemental unsharded boundary diagnostics from exact frozen activation fixtures. Retain historical references in place. For O input use the frozen single-card attention helper and frozen K/V, solely in the test harness.
3. Run real MLU Q, KV-A, RMSNorm, KV-B and O in sequence, compare every boundary and then run stability and timing.

## Interfaces and fixture scope

`MLAProjection` keeps the five layer-local checkpoint parameters isolated from the model entrypoint. It reads safetensors slices by audited axes and validates BF16 metadata. The hardware probe executes layer 0. It verifies all 40 TP=8 device weight hashes against the Phase 13.3.1 inventory. The inventory and independent source review cover all 27 layers.

The probe reuses the exact frozen prompt bytes and layer-0 embedding, K/V and attention fixtures. Supplemental TP=1 intermediate boundaries are stored under `artifacts/build/phase13_3_2/` because the frozen run did not persist Q, raw KV-A, latent gamma output or KV-B boundaries. Independent FP32 MLU ATen GEMMs cross-check the native TP=1 GEMM outputs. Existing frozen tables and the frozen attention helper serve only to recover the O input in that isolated harness. Historical O output is read from the frozen Phase 12.5.4 reference. Production RoPE and attention, scheduling, cache and the model guard are unchanged.

```bash
set -o pipefail
runner=integration/vllm_native_mlu/phase13_3_2/run_probe.sh
logs=artifacts/logs/phase13_3_2
bash "$runner" 1 reference > "$logs/reference.log" 2>&1
for tp in 1 2 4 8; do
    bash "$runner" "$tp" test > "$logs/tp$tp.log" 2>&1
    printf 'TP=%s launcher status=%s\n' "$tp" "$?"
done
python3 integration/vllm_native_mlu/phase13_3_2/report.py "$logs"
```

Use fresh log paths for a subsequent experiment. The probe writes rank results and releases resources before returning a nonzero status on numerical mismatches.
