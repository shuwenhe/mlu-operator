# Phase 13.3.1 — MLA checkpoint weight mapping

The index and every referenced safetensors header identify exactly five BF16 MLA parameters per layer across 27 layers. The inventory contains 135 unique tensors and 1,080 rank records with file name, absolute byte range, original/local shape, axis, rank, offset, replication, reconstruction and SHA-256. Query A/B, Query RMSNorm and standalone K/V or rotary projection weights are absent. [Complete checkpoint inventory](../artifacts/logs/phase13_3_1_checkpoint_inventory.csv) · [Reconstruction evidence](../artifacts/logs/phase13_3_1/inventory_result.txt)

For rank r∈[0,7], own global heads 2r and 2r+1 in increasing order.

| Suffix under model.layers.L.self_attn | Global shape | TP=8 local and ownership |
|---|---|---|
| q_proj.weight | [3072,2048] | [384,2048], rows [384r,384r+384); each head has no-PE 128 then PE 64 |
| kv_a_proj_with_mqa.weight | [576,2048] | Replicate all rows; [0,512) produce latent, [512,576) produce one shared rotary K |
| kv_a_layernorm.weight | [512] | Replicate full gamma |
| kv_b_proj.weight | [4096,512] | [512,512], rows [512r,512r+512); each 256-row head stores K-noPE 128 then V 128 |
| o_proj.weight | [2048,2048] | [2048,256], columns [256r,256r+256); each head contributes 128 consecutive V channels |

Q and KV-B concatenate axis 0 in rank order. O concatenates axis 1 independently for each output row. A single full replica reconstructs KV-A and gamma. The byte audit reads candidate slices directly from the checkpoint and verifies exact reassembly against the original bytes for every layer. The rotated shared K has no separate K-B shard; it originates from the final 64 rows of KV-A. Keeping Q and KV-B as one Phase 13.2 Column section each preserves the checkpoint head-major packing.
