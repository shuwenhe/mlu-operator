# Native MLA acceptance

**BLOCKED for actual vLLM model invocation.** The Phase 12.3 fixture probe connects native paged writes and reads to the frozen QK, Softmax and AV launchers. The input fixtures provide projected and rotary-applied Q/K/V. Phase 10 MLA projection and RoPE are pending in the scheduled model path. [Native fixture trace](../artifacts/logs/phase12_3_native_mla_hardware.log) · [Numerical CSV](../artifacts/logs/phase12_3_accuracy.csv)

**Component PASS on MLU590-M9 card 0.** Two BF16 and two FP16 request fixtures each ran prefill and one decode step with noncontiguous pages. All 65,536 output values matched bit-for-bit. Four invalid live-slot trials were rejected before cache writes. The numerical threshold was `1e-5 + 1e-5 * abs(reference)`.

To replay in the pinned SDK with the mounts from the compatibility audit:

```bash
py=/torch/venv3/pytorch_infer/bin/python3
"$py" -m integration.vllm_native_mlu.phase12_3.probe \
    /vout/native-attention/libnative_mlu.so /vout/phase12/libpaged_kv.so /out /av-out \
    /vout/phase12_3-accuracy.csv
```
