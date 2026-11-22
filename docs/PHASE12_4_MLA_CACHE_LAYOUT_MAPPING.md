# Vendor MLA cache layout mapping

**PASS for the layout audit.** An actual 65-token vendor request traced prefill and decode attention for each of the 27 layers on MLU590-M9 card 0. Both paths used BF16 cache shape `[1,28014,1,16,576]` with element strides `[258177024,9216,9216,576,1]` and an empty scale tensor. The logical axes are one cache plane, physical block, compressed KV head, offset within a block and feature. Each token stores 512 normalized latent components and 64 positionally encoded key components. One block holds 16 tokens and occupies 18,432 bytes per layer. For physical block `b` and in-block offset `j`, the feature row starts at `2 * (b * 9216 + j * 576)` bytes.

Vendor prefill passed materialized token-major Q and K `[65,16,192]` and V `[65,16,128]` while retaining the compressed persistent cache. Vendor decode passed Q `[1,16,576]`, K `[1,1,576]` and a 512-wide latent V input. The decode MLA cache write uses scheduler slots with the vendor paged-cache reshape path. Its read uses the block table and committed lengths. During tracing the request used blocks 1–5, with the remaining block-table entries padded.

The frozen native chain consumes materialized, head-major Q/K `[16,tokens,192]` and V `[16,tokens,128]`. The native page adapter uses token-major K and V planes before gathering into that layout.

| Option | Storage and execution tradeoff |
|---|---|
| A. Consume vendor compressed bytes directly | Retains 1,152 bytes per token per layer. Requires separate compressed-attention arithmetic, projected queries, latent reduction and output transformation. The frozen materialized kernels cannot read these bytes directly. |
| B. Expand on device | Retains compact persistent storage and can feed frozen kernels after KV-B projection and layout assembly. Gather and expansion of historical keys and values on every decode add compute and scratch. Correctness depends on preserving each rounding boundary. A persistent duplicate would increase memory and is outside this design. |
| C. Scheduler-owned materialized pages | Uses 10,240 bytes per token per layer, 8.89× the vendor logical cache size. It preserves the accepted native arithmetic and needs only per-request gather scratch. This is the tested component contract. The vendor model must select its spec and provide engine storage and metadata before deployment. |

A/B latency and the end-to-end cost of C were not measured. The cache layout cannot be converted with a view because the latent needs the original learned expansion. [Vendor hardware trace](../artifacts/logs/phase12_4_attention_hardware.log)
