# Phase 9.1.2 native QK hardware acceptance

**PASS for the QK bias-free, causal Layer 0 fixture matrix.** The promoted FP16 and BF16 kernels compiled with CNCC 5.7.6 targeting `mtp_592` and ran through CNRT 7.7.3 on MLU590-M9 card 0. All four synthetic smoke cases passed, including padded cache capacity. The captured matrix covers two request slots, lengths 1, 7, 32 and 128, and matching single-token decode calls at cache offsets 1, 7, 32 and 128. The replay exercises request data independently. It is a QK operator check.

| Observed checkpoint boundary | BF16 bit differences | FP16 bit differences |
| --- | ---: | ---: |
| First low-precision conversion | 0 / 564,160 | 0 / 564,160 |
| FP32 scaled product | 0 / 564,160 | 0 / 564,160 |
| Second conversion and final scaled score | 0 / 564,160 | 0 / 564,160 |
| Masked score | 0 / 564,160 | 0 / 564,160 |

There were zero causal-mask decision failures and zero unexpected NaN or Inf values. The separately compiled diagnostic device kernel emitted the widened first conversion and pre-cast FP32 scale product. Its low-precision scaled output matched the promoted kernel bitwise. The raw dot has separate FP32 and FP64 diagnostic rows in the CSV. The scale is derived from the original checkpoint instance and passed to the launcher as FP32.

The fixture generator traces the unmodified checkpoint eager attention source at its QK call inside the existing Layer 0 checkpoint runner. The original model's stored weights and embedding fixtures are loaded into each requested dtype before the forward pass. FP16 activation bits come from an independent FP16 forward. The captures record materialized Q/K as int16 bit patterns, with no-PE in features 0–127 and post-RoPE in features 128–191. Each capture contains positions, scaling and checkpoint scores. The surrounding runner separately captures Layer 0 intermediates. The capture files, provenance and `CAPTURE_SHA256SUMS` are in `artifacts/build/phase9_1_2`.

Single-launch CNRT notifier samples exclude transfers and process startup. For length 128, the original serial baseline measured 23.22–23.27 ms for prefill and 365.52–367.00 µs for decode. The isolated 64-lane experiment measured 21.17–21.25 ms and 334.16–335.20 µs respectively. These measurements establish operator reference points for later optimization.

To reproduce, use the Phase 8.2 SDK image with card 0 exposed. Run `qk_parity/capture.py` with the original checkpoint path, `runtime-full-fixtures`, a per-case output directory, dtype and prefill length. Run `qk_parity/reductions.py` with the experiment directory and build each generated variant with its `build_qk.sh`. Run `qk_parity/compare.py` with the capture root and each experiment directory. For the production replay, also pass the diagnostic executable path. Use `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for the reference.

Frozen Phase 8.2 and archived baseline source hashes verified after promotion.

[Device builds, smoke tests and notifier samples](../artifacts/logs/phase9_1_2_qk_hardware.log) · [Full accuracy matrix](../artifacts/logs/phase9_1_2_qk_accuracy.csv)
