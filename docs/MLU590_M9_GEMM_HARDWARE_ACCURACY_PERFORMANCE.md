# MLU590-M9 GEMM hardware accuracy and performance

**PASS for the tested packed row-major FP16/BF16 fixtures on host MLU 2.** The native BANG C harness passed six shapes per dtype, including 32×64×128, 128×256×512, 256×128×512 and 127×193×65. The CNNL harness passed seven shapes per dtype. Both reported zero maximum absolute, maximum relative and mean absolute error and zero mismatches on their respective CPU FP32 oracles. Compiler output for BANG C shows separate FP32 multiply and add after input conversion. CNNL requests FP32 compute and output.

The shared shape below uses the same card, dtype, packed layout, M/N/K and effective FLOP count. Each harness uses its own deterministic fixture and timing scope. This is a same-shape observation, with no claim of a matched-input end-to-end benchmark.

| 127×193×65 | BANG C kernel µs | CNNL kernel µs | BANG C TFLOPS | CNNL TFLOPS |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 1985.71 | 2.754 | 0.00160468 | 1.15702 |
| BF16 | 1986.41 | 2.680 | 0.00160412 | 1.18897 |

BANG C used two warmups and 20 timed launches. CNNL used one warmup and 20 timed launches. TFLOPS uses 2MNK divided by kernel seconds and 1e12. For the common shape, BANG C transfer-inclusive single-call wall time was 2049.11 µs FP16 and 2063.79 µs BF16. CNNL measured H2D and D2H separately and reports a session wall time including allocation, setup and repeated launches. That session number is not comparable with the BANG C single-call transfer-inclusive interval. CNNL peaked at 143.664 FP16 and 143.933 BF16 TFLOPS at 1024 cubed. The standalone BANG C correctness kernel uses scalar FP32 arithmetic and is not a competitive substitute for the vendor matrix path.

[BANG C measured data](../artifacts/logs/gemm_mlu8_bang.csv) · [CNNL measured data](../artifacts/logs/gemm_mlu8_cnnl.csv) · [Commands and build evidence](../artifacts/logs/gemm_mlu8_hardware.log)
