# Phase 13.2 — Checkpoint sharding

Each weight is logically `[N,K]`. For column sharding, a projection section of width `S` at offset `B` contributes `S / TP` rows starting at `B + rank × (S / TP)`. Divide each packed section separately. The local tensor concatenates sections in checkpoint order. A gather for each section recovers section-major order. Column bias follows the same spans.

Row sharding reads input columns at `rank × (K / TP)` with width `K / TP`. Row bias is replicated and applied to the reduced result. The optional residual is likewise added on each rank after reduction. Replicated inputs are narrowed on-device. Pre-sharded inputs must have the exact local width.

Positive dimensions must divide the TP size on the sharded axis. In a packed column projection every section must divide independently. Native BF16 alignment pads each local shard's K and N to 64 and crops output to the logical local N. Logical output widths above 10,944 are tiled into separate local GEMMs. Padding and GEMM tiling preserve the checkpoint spans. The native packed buffer allocates complete 128×128 storage tiles.

The hardware fixture generates weight values directly at rank-local indexes. The real checkpoint probe separately uses safetensors slice reads for six layer-0 Attention and MLP projections and verifies the checksum after an MLU round trip. The host verifier reads the checkpoint independently and compares each reconstructed tensor byte-for-byte via SHA-256. It also verifies the ordered gate/up packing. See the [rank-level shard CSV](../artifacts/logs/phase13_2_weight_shards.csv) and [checkpoint reconstruction log](../artifacts/logs/phase13_2_checkpoint_device_verify.log).
