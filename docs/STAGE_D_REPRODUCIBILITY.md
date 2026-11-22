# Reproduce BF16 TP8 on eight MLU590-M9 cards

## Environment

Use a host with eight MLU590-M9 cards exposed as ordinals 0–7, the Cambricon runtime device nodes, and Docker access. The accepted host reported driver and CNMON 6.5.52. The frozen container reported CNCL 1.30.10 and Torch MLU 1.34.1 with Torch 2.12.1.

Load the accepted image into Docker from its approved archive and confirm its image ID. The launcher uses the image ID and disables pulls and container networking.

```sh
docker image inspect sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756 --format '{{.Id}}'
cnmon
make deepseek-verify
make deepseek-check
```

Preserve the repository, its C-stage source and evidence directories, and the archived `artifacts/build/phase12_5_4/nativelibs` and `rope_yarn` bundles. The runtime also uses `artifacts/logs/phase13_3_3/libordered.so` and `artifacts/logs/phase15/stage_c1/libpaged_two.so`. These include locally archived binary assets that normal Git ignore rules may exclude. Bring the verified artifacts along when transferring the checkout. Keep the original DeepSeek-V2-Lite checkpoint at `../model/DeepSeek-V2-Lite`. Include all four safetensors shards, configuration, model code and tokenizer. Frozen benchmark runners use this sibling path.

The card check requires a complete eight-device inventory, an empty CNMON process table and zero allocated card memory. A PID with zero KiB still occupies the card. Coordinate access with the site scheduler or card owner before running.

## Inference and evidence

```sh
make deepseek
make deepseek-c2
make deepseek-c3
```

The default and explicit C.3 invocation select the same accepted model and expert flags. All three commands use a 32-token output budget by default. Check `workers.csv` for the exact worker configuration, `generation_outputs.json` for emitted tokens, and `lifecycle.csv` for all exit codes. Product-launch validation results are listed in the Stage D report. [Stage D status](../artifacts/logs/phase15/stage_d/final_report.md)

## Comparable benchmarks

```sh
for run in 1 2 3; do
    make stage-c2 || break
    make stage-c3 || break
done
```

Use the same fixed 17-token input, 32 output tokens, and two warm requests per fresh engine. Keep the two warm samples grouped by engine when comparing runs. Cold TTFT is the first request after engine initialization. Compare the cold and warm rows separately. Each launch writes its own directory in `DEEPSEEK_LOG_ROOT` or `artifacts/logs/deepseek`. The Stage C.3 report links the original paired runs, the separate eight-rank profiling evidence, and the bitwise logit comparisons. [Stage C.3 benchmark](../artifacts/logs/phase15/stage_c3/final_report.md)

## Local SDK build

Use the SDK inside the frozen image or an equivalent installation with supported headers and linker libraries. Run `make cpu` for the CPU reference and `make mlu native NEUWARE_ROOT=/usr/local/neuware` for SDK objects and linked binaries. These build commands do not run the native device tests. Run `make deepseek-verify` again before archiving any reproduced result.
