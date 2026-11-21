
The QK comparison also found 24 out-of-tolerance positions that survive the causal mask. For example, slot 0 prefill head 1, query 64, key 9 produced PyTorch `-15.1875` and serial fixture `-15.25`. The device operands and both sets of intermediate arrays are preserved as `*attention_audit.npz` in the attention directory recorded by `artifacts/logs/phase8_2_results_path.log`.

[Causally visible QK differences](../artifacts/logs/phase8_2_causal_qk.log)
