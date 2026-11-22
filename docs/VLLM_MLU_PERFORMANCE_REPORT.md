# vLLM MLU performance report

One cold vendor vLLM run on MLU590-M9 card 0 used BF16, eager execution, maximum model length 128 and maximum four scheduled sequences. It loaded the original checkpoint, then generated four tokens each for two prompts submitted together.

| Vendor vLLM measurement | Result |
|---|---:|
| Initialization through offline API | 102.85 s |
| Weight loading reported by the engine | 13.12 s |
| Engine profile, cache creation and warmup | 2.73 s |
| Two-request generation wall time | 1.65 s |
| Aggregate tokens per second for this call, including prefill | 4.85 |

The engine logged slot-mapping Triton JIT during generation. Per-request TTFT, inter-token latency, warm measurements, peak host/device memory, transfer volume, expert scheduling overhead and cache memory in bytes were **NOT RUN** for the requested native vLLM model. There is no matched three-way performance comparison. Gate H is **NOT RUN**.

[Vendor engine execution log](../artifacts/logs/vllm_mlu_hardware.log) · [Standalone timing evidence](../artifacts/logs/phase11_hardware.log)
