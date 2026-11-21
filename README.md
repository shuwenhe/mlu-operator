# Cambricon MLU590-M9 standalone operators

CNNL-based BF16 and FP16 GEMM and RMSNorm development for DeepSeek inference. The project builds independently from NeurX and uses the vendor libraries for device execution.

```sh
scripts/environment_check.sh
scripts/test.sh
NEUWARE_ROOT=/path/to/neuware scripts/build.sh objects
```

`objects` compiles the native wrappers, harnesses and runtime probe without linking the SDK. For linked builds use `scripts/build.sh native` with an SDK compatible with the execution environment. CMake equivalents are `-DMLU_NATIVE=ON` for objects and `-DMLU_NATIVE=ON -DMLU_LINK=ON` for executables.

After explicit authorization, reserve one card, ensure it is the only device visible to the process, run `cnrt_probe` with the compatible SDK on the library path, then run `scripts/benchmark.sh --authorized <ordinal>`. The authorization flag records operator intent. It does not reserve a card or grant access.

- [Environment and SDK](docs/ENVIRONMENT.md)
- [Hardware inventory](docs/HARDWARE_INVENTORY.md)
- [Operator design and integration](docs/OPERATOR_DESIGN.md)
- [Technical acceptance](docs/ACCEPTANCE_REPORT.md)
