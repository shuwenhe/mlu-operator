
The device-only candidate uses 32 lane accumulators within 512-element chunks, folds the chunk sums and lane groups in the pinned CPU reduction order, and computes square root followed by division. It uses independent symbols for FP16 and BF16 and accepts widths 512 and 2048. Both objects compiled for `mtp_592`. The standalone test on MLU 0 produced zero differing final words over 839,680 values across the 20 captured chunks. This evidence applies to the tested checkpoint operands and widths.

[Device-only execution](../artifacts/logs/phase8_2_native_norm_hardware.log)

The first device-only full matrix still had one FP16 length-128 post-norm error at request 1, prefill row 35, column 1203 (checkpoint `2559`, device `2558`). It propagated to 61 final outputs. A replay of the cascade sum matched checkpoint mean-square bits throughout that chunk. The pinned checkpoint's `rsqrt` also matched independently rounded FP32 square root followed by FP32 division on all 694 captured post-attention rows.

The revised experimental device kernel implements both round-to-nearest-even steps with integer significands for positive normal checkpoint variance inputs. This removes dependence on the original vector sqrt/div rounding in the experiment. On the targeted FP16 chunk, the revised device output matched all 262,144 words. The full revised FP16 length-128 fixture passed the original checkpoint gate. The earlier device-only result and source layout remain available under the recorded results path.

[Targeted and remaining hardware runs](../artifacts/logs/phase8_2_exact_rsqrt_remaining.log) · [Original FP16 length-128 rerun](../artifacts/logs/phase8_2_exact_rsqrt_hardware.log) · [CPU arithmetic cross-check](../artifacts/logs/phase8_2_rsqrt_cpu_audit.log)

The revised full-layer output also passed a direct check of all 36 KV latent RMSNorm chunks at width 512, with zero differing words. The reference uses the first 512 columns of the original checkpoint KV-A trace and the checkpoint latent gamma.

[Latent replay commands](../artifacts/logs/phase8_2_latent_replay.log)
