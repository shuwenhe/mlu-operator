# Phase 10 KV cache acceptance

**PASS on MLU590-M9 card 0.** Each request owns physical head-major K `[16,capacity,192]` and V `[16,capacity,128]` buffers in its dtype. A device kernel writes only the appended token interval. The frozen QK and AV kernels read that same device memory using the physical head stride and proposed valid length. The request advances its committed length only after its queue completes. The 512-wide compressed latent is a projection intermediate under the eager checkpoint contract. Persistent cache storage uses expanded K/V.

All checked committed prefixes matched the original PyTorch eager cache bit-for-bit. Per dtype this covered 17,491,968 K values and 11,661,312 V values. Full-prefix comparisons after each append verified that earlier tokens remained intact. The device cache guards and unused suffix sentinels remained intact through prefill and capacity filling. The harness also checked positions, rejected a non-contiguous initial append and a capacity-overflow append, and rejected reset or append while work was pending.

Reset and reuse were tested with the other request's input tokens. Stale suffix storage remained outside the new committed length and the resulting probabilities and outputs matched checkpoint. Two independent cache allocations were exercised on one queue and on two queues. A mixed group used a 10-token capacity and a 131-token capacity at the same time. Device memory returned to zero managed bytes after cache, scratch and shared-table release. Packed weights were also closed.

The implementation uses logical reset. Erasing cache storage for secure reuse and concurrent host-thread access are outside the request API. Queue submission and cleanup are owned by the worker thread.

[Cache accuracy](../artifacts/logs/phase10_kv_cache_accuracy.csv) · [Decode and lifecycle log](../artifacts/logs/phase10_decode_hardware.log)
