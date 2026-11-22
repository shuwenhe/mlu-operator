# DeepSeek-V2-Lite per-layer cache specification

The checkpoint audit and vendor runtime trace verify identical attention dimensions and compressed cache layout on layers 0–26. Layer 0 has a dense FFN, while layers 1–26 have MoE FFNs. That difference does not alter the attention/cache contract.

| Decoder layers | Type | Native specification |
|---|---|---|
| 0 | Dense | BF16 FullAttentionSpec, 16 heads, K width 192, V width 128 |
| 1–26 inclusive | MoE | Same independent FullAttentionSpec for each of the 26 layers |

The implementation creates a separate spec entry for each layer and accepts a raw byte allocation in the vendor worker's `reshape_kv_cache(raw, spec, num_blocks, kernel_block_size, cache_dtype)` format. All K slots precede all V slots in a single allocation. Their tensor views have element strides `[3072,192,1]` and `[2048,128,1]`. With block size (B) and (N) blocks, both views have (N B) token slots and the page byte size is (10240 B). The physical token slot is `block_table[position // B] * B + position % B`. Cache writes use the scheduler's int32 slot mapping. Reads use block IDs and valid logical length. In a real binding the worker owns device allocation and the scheduler owns page IDs and lifetime. The views own no duplicate backing allocation.

**Component PASS on card 0.** The probe merged the 27 installed FullAttentionSpec entries and used their computed 163,840-byte page size at block size 16. For each layer, it verified zero-copy view pointers and ran the original native paged write/read across noncontiguous blocks 2 and 0. Scheduler allocation and the native backend reshape callback remain unbound to the actual model. [Component log](../artifacts/logs/phase12_4_model_binding_hardware.log)
