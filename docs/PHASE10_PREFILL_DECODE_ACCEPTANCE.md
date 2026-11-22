# Phase 10 prefill and continuous decode acceptance

**PASS on card 0 for the supported materialized MLA matrix.** BF16 and FP16 each ran prefill lengths 1, 7, 32 and 128, then three single-token decode steps, then reset and reuse. The following configuration families were tested for both dtypes.

| Lengths | Requests | Queues | Runs |
| --- | ---: | ---: | ---: |
| 1, 7, 32, 128 | 1 | 1 | 8 |
| 1, 7, 32, 128 | 2 | 1 | 8 |
| 1, 7, 32, 128 | 2 | 2 | 8 |
| Mixed 7 and 128 | 2 | 1 or 2 | 4 |

There were 48 prefill, 144 decode, and 48 reset/reuse request steps. Every decode output matched both the independently captured incremental checkpoint and the same token in an original full-sequence PyTorch run. Prefill compares against same-shape checkpoint prefill. A cross-shape CPU-only diagnostic found 19 out-of-tolerance values when comparing FP16 request 1's seven-token prefill against the corresponding prefix of PyTorch's ten-token full run. The subsequent three continuation tokens match. This is a difference between reference execution shapes and is separate from native same-shape prefill acceptance.

All comparisons retained `1e-5 + 1e-5 * abs(reference)`. The causal masks, latent path, expanded K/V, Softmax, merged AV and final projection were checked per step. There were zero unexpected nonfinites, out-of-tolerance values, or cache guard failures. Logical batch sizes 1 and 2 are represented by independently owned requests. Other head counts, a fused multi-request batch kernel, prefill chunks above 128, and cache capacities above 131 are outside the measured matrix.

[Prefill hardware log](../artifacts/logs/phase10_prefill_hardware.log) · [Decode hardware log](../artifacts/logs/phase10_decode_hardware.log)
