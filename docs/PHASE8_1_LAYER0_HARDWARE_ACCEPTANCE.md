# Phase 8.1 Layer 0 hardware acceptance

**FAIL.** MLU590-M9 PCI `0000:53:00.0` executed all nine candidate fixtures with two queues, two request slots, two repetitions, and one decoded token per request. The SDK is Ubuntu 24.04.3, CNCC 5.7.6 for `mtp_592`, CNRT 7.7.3 and CNNL 2.2.18 on driver 6.5.52. The checkpoint oracle was PyTorch 2.12.1+cpu. All gates retain the existing `1e-5 + 1e-5 * abs(reference)` threshold.

**Matched projection parity PASS.** All 252 seven-projection comparisons returned the exact low-precision PyTorch bits, including the original fixture. An additional 72 matched pointwise comparisons were bit exact. All 252 candidate FP32 projection comparisons against FP64 passed, as did the 252 widened PyTorch FP32 comparisons. Those references have separate CSV records.

**Original historical oracle FAIL.** The same original BF16 runtime fixture reported 400 / 24,576 final mismatches against the historical C++ oracle, compared with 404 / 24,576 for native default. Both counts include repetitions. The original BF16 fixture against checkpoint PyTorch passed all traced numerical gates and improved from 4 / 12,288 to 0 / 12,288 final mismatches across unique chunks.

| Dtype | Prefill | Final prefill failures | Final decode failures | Earliest stage outside tolerance across both slots |
| --- | ---: | ---: | ---: | --- |
| BF16 | 1 | 0 | 0 | None |
| BF16 original | 2 | 0 | 0 | None |
| BF16 | 7 | 0 | 0 | None |
| BF16 | 32 | 287 | 0 | Post-attention RMSNorm, slot 1 prefill, row 16 col 229, `bcad → bcae` |
| BF16 | 128 | 2384 | 384 | Reference attention, slot 0 prefill, row 39 col 382, `3cff → 3d00` |
| FP16 | 1 | 120 | 223 | Post-attention RMSNorm, slot 0 decode, row 0 col 341, `26e2 → 26e1` |
| FP16 | 7 | 3296 | 435 | Input RMSNorm, slot 1 prefill, row 2 col 152, `af80 → af7f` |
| FP16 | 32 | 18828 | 1042 | Input RMSNorm, slot 1 prefill, row 2 col 152, `af80 → af7f` |
| FP16 | 128 | 85313 | 701 | Input RMSNorm, slot 1 prefill, row 2 col 152, `af80 → af7f` |

FP16 input norm differs despite identical hidden input bits. Replaying the checkpoint's exact post-attention RMSNorm on the candidate's residual also reproduced the cited BF16 length 32 and FP16 length 1 first failures with identical input words. BF16 length 128 matches through the rounded recorded Q and KV-B projection tensors. The first recorded difference is at reference attention. The existing trace lacks separate rotary-output, softmax and detailed cache-update dumps, so the exact operation inside that interval remains unverified. The existing trace captures 16 original-source comparison stages and additional native FP32 and KV intermediate values. No nonfinite values appeared. The expanded checkpoint gate failed 188 / 576 stage comparisons.

**Lifecycle PASS.** The fixture reused persistent weights and kept two disjoint KV cache buffers. All candidate runs freed their tracked allocations. A fresh R3/R4 hardware run passed injected allocation rollback, queue synchronization failure recovery and cleanup. Final CNMON recorded 0 MiB of physical memory in use on all eight cards. Explicit backend selection preserved the independent CNNL path. No production integration is proposed while Layer 0 remains failed.

[Layer 0 accuracy CSV](../artifacts/logs/phase8_1_layer0_accuracy.csv) · [Matched norm replay](../artifacts/logs/phase8_1_matched_norm.csv) · [Hardware log](../artifacts/logs/phase8_1_hardware.log)
