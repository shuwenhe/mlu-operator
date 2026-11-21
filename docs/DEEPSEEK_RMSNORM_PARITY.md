# DeepSeek-V2-Lite RMSNorm numerical parity

## Exact operation order

The local checkpoint's `modeling_deepseek.py:94` widens the input to FP32, squares it, takes the mean over the last dimension, adds epsilon (1e-6), computes reciprocal square root and multiplies it into the FP32 input. It then casts that normalized activation back to the original input dtype **before** multiplying by the learned weight.

For checkpoint BF16 weights and BF16 activations, gamma and returned output are BF16. The FP16 tests explicitly cast both operands to FP16. With matched low-precision operands, the reference computes the gamma product from their rounded values and rounds the output to the shared dtype. A mixed dtype gamma would follow PyTorch promotion and is outside these fixtures.

| Stage | Dtype |
| --- | --- |
| Input activation and checkpoint gamma | BF16, or FP16 for the explicit conversion tests |
| Input after widening, square, mean, epsilon, reciprocal square root, normalized activation | FP32 |
| Normalized activation immediately before gamma | Original activation dtype, round to nearest ties to even |
| Gamma product returned with matching weight and activation dtype | BF16 or FP16 |

## CPU reference verification

[Parity script](../integration/deepseek/rmsnorm_parity.py) implements explicit FP32 arithmetic and BF16/FP16 rounding independently of PyTorch. The CPU differential extracts the exact checkpoint class and executes it with PyTorch 2.12.1 CPU in a device-free container. The existing fixtures were read without modification. They cover real input/final gamma at width 2048, real KV gamma at width 512, zero rows and synthetic widths 1 and 65. Scalar checks cover rounding ties and signed zero. The CPU reference uses a serial FP32 reduction, so the observed exact agreement applies to these fixtures and does not establish equality for arbitrary reduction orders or reciprocal-square-root implementations.

| Dtype | Comparison against independent reference | Elements | Max abs | Max rel | Mean abs | Different |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | Exact checkpoint PyTorch CPU class | 13,956 | 0 | 0 | 0 | 0 |
| FP16 | Exact checkpoint PyTorch CPU class | 13,956 | 0 | 0 | 0 | 0 |
| BF16 | FP32 post-gamma CPU proxy | 13,956 | 0.039490700 | 0.006147442 | 0.0005629144 | 8,983 |
| FP16 | FP32 post-gamma CPU proxy | 13,956 | 0.002482891 | 0.000752998 | 0.0000751652 | 8,982 |
| BF16 | Post-gamma cast CPU proxy | 13,956 | 0.0625 | 0.0078125 | 0.0003883666 | 1,915 |
| FP16 | Post-gamma cast CPU proxy | 13,956 | 0.00390625 | 0.000974659 | 0.0000568737 | 2,675 |

Relative error uses a denominator floor of 1e-6. The mean weights each element equally. The largest late-cast errors occur with real layer-zero input norm gamma. Full per-fixture results are in `artifacts/logs/rmsnorm_parity_torch.csv`.

```sh
python3 integration/deepseek/rmsnorm_parity.py artifacts/build/deepseek-fixtures
# In a CPU-capable PyTorch environment, also supply:
# --torch-source /app/shuwen/model/DeepSeek-V2-Lite/modeling_deepseek.py
```

## CNNL and BANG C device paths

The existing CNNL wrapper preserves an FP32-output contract using FP32 gamma. The CPU proxies isolate that ordering difference and a proposed final cast. Neither proxy is a measurement of CNNL. To test a CNNL-based parity route, run the norm with unit gamma and FP32 output, apply explicit low-precision rounding, multiply by low-precision gamma, and round the product. Compare the actual existing wrapper and the adapted route independently against the reference. Verify CNNL reduction and reciprocal-square-root numerical behavior before accepting the adapted route.

The independent [BANG C rounding stage](../probes/rmsnorm_round_gamma.mlu) takes an already normalized FP32 tile and same-dtype gamma. It explicitly narrows with round-to-nearest, widens the rounded values for FP32 multiply, then narrows the output. Both variants compiled with CNCC 5.7.6 at `--bang-mlu-arch=mtp_592 -O2 -std=c++11`, using `-DPARITY_BF16` for BF16. The objects contain device fatbins, and device assembly shows the pre-multiply and output narrowing conversions. The probe covers 128 lanes per launch and contains no RMS reduction or tail handling. It was compiled in a device-free container and has no measured output. See `artifacts/logs/rmsnorm_parity_bangc.log`.

**MLU parity remains blocked.** No authorized exclusive MLU allocation was established. The CNNL wrapper, existing fixtures, production vLLM and NeurX were preserved. Before accepting device substitution, verify CNRT transfers on an authorized card, compare intermediate normalized values, pre-gamma rounded bits and final values, then extend fixtures to cancellation-sensitive and wide-dynamic-range rows. CPU equality and device compilation are separate evidence.
