# DeepSeek-V2-Lite YaRN RoPE acceptance

**Hardware PASS on host MLU 4 (`0000:af:00.0`).** The test extracts the checkpoint's original PyTorch YaRN helper and rotation functions. The configuration uses factor 40, theta 10000, original context 4096, beta_fast 32, beta_slow 1 and both mscale parameters 0.707. It generates a compact coefficient table for the tested position IDs. The native BANG kernel looks up explicit indices into that table. A serving implementation must provide the matching position-to-table mapping.

Inputs pack 16 Q heads and one shared K head as contiguous `[17,S,64]` with element strides `[S*64,64,1]`. The original interleaved even/odd pairs are reordered to halves, then rotated. The two products round separately to the input dtype before addition and final rounding, matching the checkpoint expression. Coefficients are stored in the input dtype.

FP16 and BF16 sequence lengths 1, 7 and 65 passed. The cases include position 0, the original 4096 boundary, repeated positions, reverse order and positions up to 163839. Across **158,848** outputs, the native kernel matched all expected bits with zero signed-zero differences and zero absolute error. PyTorch generated the coefficient tables on CPU. No native coefficient generation or RoPE performance measurement was performed.

[Accuracy CSV](../artifacts/logs/phase5_rope_accuracy.csv) · [Fixture and compiler log](../artifacts/logs/phase5_rope_run.log)
