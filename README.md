# NeurX MLU Operator

Native BF16 DeepSeek-V2-Lite inference on eight Cambricon MLU590-M9 cards. The repository also contains standalone BF16 and FP16 GEMM and RMSNorm operators.

## Build

```sh
make help
make cpu
make mlu NEUWARE_ROOT=/usr/local/neuware
make native NEUWARE_ROOT=/usr/local/neuware
```

`make cpu` builds and tests the standalone CPU reference. `make mlu` compiles the SDK-backed operator objects. `make native` links the SDK-backed operator tests and runtime probe. The SDK targets require the matching Cambricon headers and, for linking, CNRT and CNNL libraries. The DeepSeek runner loads the archived native libraries from the accepted inference environment.

## Inference

```sh
make deepseek-verify
make deepseek-check
make deepseek
make deepseek-c2
make deepseek-c3
make deepseek-chat
```

`make deepseek` selects the accepted Stage C.3 integration with fused and batched expert GEMMs, output-axis MLA, and CNCL TP8. The C.2 command selects output-axis MLA with the earlier expert decoder. The launcher uses the pinned image and requires eight idle MLU590-M9 cards. It validates each worker before accepting a request.

`make deepseek-chat` loads one Stage C.3 model and streams engine-generated text across turns. Use `/help`, `/stats`, `/reset`, `/exit` or `/quit`. Ctrl+C requests cancellation at the next engine step boundary and closes the session.

```sh
make deepseek-chat MAX_TOKENS=32 MAX_MODEL_LEN=128 TEMPERATURE=0.7
make deepseek-chat HISTORY_POLICY=reject
```

The frozen native model supports at most 128 context tokens, including the reserved response budget. The default `truncate` policy removes complete older turns silently and records the count in the request evidence. `reject` keeps history and asks you to shorten or reset the conversation. Set `CHAT_SYSTEM_PROMPT` to add a system message. Chat evidence, including prompts and responses, is written to a unique directory under `artifacts/logs/phase16/runs` or `DEEPSEEK_CHAT_LOG_ROOT`.

The checkpoint is the base DeepSeek-V2-Lite model. Chat stops when the model starts a new transcript role, but the base model can still give weak or off-topic answers.

Set a custom prompt and output length with:

```sh
make deepseek PROMPT='Explain tensor parallel inference on MLU.' MAX_TOKENS=32
```

For one-shot inference (`make deepseek*`), the tokenized prompt and output budget must fit the 128-token context. Inference uses TP8 and BF16. Set `DEEPSEEK_LOG_ROOT` to choose an evidence directory. Each invocation gets its own subdirectory containing the preflight snapshot, runtime log, and any resulting worker, generation and performance evidence.

## Benchmark

```sh
make deepseek-benchmark
make stage-c2
```

The benchmark uses the frozen 17-token prompt and 32-token generation. Its default is one cold request followed by two warm requests. Run each variant three times in separate invocations for comparison. `WARM_REQUESTS` controls the warm repeat count. Keep profiler runs separate from timing runs. [Eight-card reproducibility guide](docs/STAGE_D_REPRODUCIBILITY.md)

## Troubleshooting

- If preflight reports an occupied card, use that run's `cnmon_before.log` to identify the process. Run again once the card owner has released all eight devices.
- If an image or device node is unavailable, check Docker access, the pinned image ID, and the host Cambricon driver using the reproducibility guide.
- If initialization or generation fails, inspect `runtime.log` and `workers.csv` when available. Verify the archive using `make deepseek-verify` and confirm the checkpoint is present.
- If SDK builds fail, check `NEUWARE_ROOT` for the correct headers and library symlinks.

[Final architecture](docs/STAGE_D_ARCHITECTURE.md) · [Stage C.3 acceptance](artifacts/logs/phase15/stage_c3/final_report.md) · [Stage D verification status](artifacts/logs/phase15/stage_d/final_report.md) · [Operator design](docs/OPERATOR_DESIGN.md) · [Historical acceptance](docs/ACCEPTANCE_REPORT.md)

The standalone `make benchmark BENCH_AUTHORIZED=1 CARD=<ordinal>` workflow uses a reserved MLU device. Run the linked `cnrt_probe` with the matching SDK before measuring that device. [SDK environment](docs/ENVIRONMENT.md) · [Hardware inventory](docs/HARDWARE_INVENTORY.md)
