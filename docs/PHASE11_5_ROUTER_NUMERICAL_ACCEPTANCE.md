# Phase 11.5 FP32 router numerical acceptance

## Integrated baseline

The standalone model invokes `Model::run_moe` in `runtime/native/phase11/model_run.cpp`, which calls `project` and the linked `phase8_candidate_bf16` symbol from the Phase 10 `projection-bf16.o`. The isolated MoE harness calls the same symbol separately. In the model, each input row is the BF16 post-attention RMSNorm output. The logical activation matrix is `[M,2048]`, contiguous with strides `[2048,1]`. The checkpoint gate weight is BF16 `[64,2048]`, also contiguous with strides `[2048,1]`. The baseline transposes the gate weight before storing 128-by-128 packed tiles and pads the expert dimension from 64 to 128. The output buffer is contiguous FP32 `[M,64]` with strides `[64,1]`.

The compiled projection transposes each RHS tile, widens both operands to FP32, multiplies them, and accumulates 64 interleaved lanes. It folds those lanes in tiers of eight and then 4, 2 and 1. The router forwards its FP32 logits directly to the FP32 probability output of native Softmax. The original and candidate builds use CNCC `--bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off`, with the original projection additionally compiled using `-DPHASE8_BF16`. Host linking uses g++ `-O2 -ffp-contract=off`.

## Root cause and reproduction

The pinned independent PyTorch checkpoint widens BF16 activations and weights before calling FP32 linear. Profiling on the acceptance host identified oneMKL 2024.0 Update 2 generic CPU kernels with PyTorch using two threads. Different row counts use distinct addition sequences:

| Shape | Reference sequence |
|---|---|
| One row and packed SGEMM fringe rows | Accumulate products at features 0–3 in order into lane 0. Interleave the remaining products across four lanes and fold as `(lane0 + lane2) + (lane1 + lane3)`. |
| Two to seven rows | Accumulate all products in four interleaved lanes starting at feature 0 and use the same final fold. The sampled function was `mkl_blas_def_sgemm_pst`. |
| Eight to 128 rows | The sampled `mkl_blas_def_sgemm_inner` processes complete groups of four rows. Each dot sums the 32-feature tail panel first, then adds eighteen independent serial 112-feature sums in increasing feature order. Remaining rows use GEMV. The pinned two-thread row partition above 64 rows splits the range at `floor(M/2)`, giving each partition its own fringe. |

The independent device probe exported every product and each strategy's partial sums to `artifacts/build/phase11_5/layer*-m*/hardware-mode*.npz`. Products and partials matched the explicit host controls bit-for-bit. The original router reduction differed by 53/64 bits for the single row taken from the Layer 2 capture, 806/1,088 bits for Layer 2 17-token prefill and 49/64 bits for Layer 26 one-token prefill. The Layer 2 single-row case uses the first actual 17-token activation with a freshly calculated one-row PyTorch reference. The integrated one-token run also traces Layer 2. Every original router score was within tolerance. Small probability changes crossed BF16 rounding boundaries during expert aggregation. In the 17-token case the baseline first differed at row 0, expert 0, with native `0x3e326e4e` versus checkpoint `0x3e326e56`. Maximum absolute and relative errors and signed-order ULP distances for all tested strategies are in the CSV.

The isolated mixed-order CNRT probe matched all 1,216 logits bit-for-bit. Its queue-plus-synchronization timings, including writes of diagnostic products and partials, were approximately 0.16 ms for the one-row cases and 1.00 ms for 17 rows in this run. These are diagnostic timings with extra memory traffic. The 64-lane probe is an arithmetic control. Phase 11 standalone timings are in the same hardware log. FP64 followed by a single FP32 rounding still differed from the checkpoint on 794 of 1,088 logits for the 17-token case.

## Router-only build

The router device source is `runtime/native/phase11_5/router.mlu`. Build the alternate executable using `build_candidate.sh`, which patches copies of the model runner and MoE harness in the output directory. The baseline runner and its linked attention, projection, normalization and expert objects remain available for regression. The alternate call site uploads the original 64-by-2048 BF16 gate weight in checkpoint order. Expert selection, dispatch, expert weights and expert combine retain their existing implementations. The kernel performs FP32 accumulation on the device and has no CPU path.

The shape sweep uses repeated captured Layer 2 activation rows and the actual gate weight. It validates each supported row count from 1 through 128 on the physical card. Full model fixtures use the original prompt token IDs and original checkpoint, independent eager CPU PyTorch, and the existing `1e-5 + 1e-5 * abs(reference)` threshold.

## Hardware acceptance

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

## Reproduction

Use the pinned SDK image and the mounts listed in the vLLM compatibility audit, plus Phase 10 at `/phase10`. Mount the Phase 11.5 output at `/p115` with its captured operands. Export the cache settings from that audit and run from the repository root.

```bash
py=/torch/venv3/pytorch_infer/bin/python3
"$py" runtime/native/phase11/reference_model.py /checkpoint /phase11/model/prompt.bin 32 /p115/reference-full > /p115/reference-full.log
bash runtime/native/phase11_5/run_candidate.sh /p115/final /checkpoint /phase10 /phase11 \
    artifacts/build/mlu8-bringup.28JR1P/prepacked-final-build /p115 "$py" 0000:4f:00.0
```

[Accuracy CSV](../artifacts/logs/phase11_5_router_accuracy.csv) · [Hardware execution](../artifacts/logs/phase11_5_router_hardware.log) · [Source verification](../artifacts/logs/phase11_5_source_hashes.txt) · [Runtime architecture](INFERENCE_RUNTIME_ARCHITECTURE_AUDIT.md)
