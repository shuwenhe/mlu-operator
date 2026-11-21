# Native MLU runtime implementation

The isolated [runtime source](../runtime/native) implements device workers, explicitly published packed weights, request slots, memory accounting and operator dispatch above the accepted kernels. It does not change production vLLM, NeurX, Team mappings, existing GEMM/RMSNorm kernels or earlier fixtures.

| Phase | Status |
| --- | --- |
| R1/R2 | Hardware PASS for PCI resolution, device properties, explicit publication, revision reload, leased invalidation and cleanup. |
| R3/R4 | Hardware PASS for two queues/slots, separate buffers, early-release prevention, fault injection and recovery, reserve policy and zero live allocations. Faults were injected, not caused by physical OOM or driver failure. |
| R5 | Hardware PASS for explicit Native/CNNL GEMM and RMSNorm plus native RoPE and native KV dispatch. CNNL RoPE is explicitly unsupported. The connected KV path uses the Native route; CNNL is not substituted upon error. |
| R6 | Full layer zero executed in BF16 with real weights and tokens, with CPU reference MLA and CPU SwiGLU/residuals explicitly selected. Layer numerical acceptance **FAIL**, 404 final elements outside existing tolerance across two repetitions. |
| R7 | Repeated execution, distinct cache storage, two queues, registry reload, injected allocation/queue errors and explicit reclamation observed. Model-level acceptance stays **FAIL** because of R6 parity. |

The current CNNL adapter creates transient descriptors and temporary buffers on each call, and its output contract can differ from native RMSNorm. For production use, add descriptor and queue-workspace reuse, verified caller-thread enforcement for all public metadata operations, consolidated packed-weight budget enforcement, fault recovery orchestration, and paged cache lifetime validation. These tests are single-card and single-layer. Full-model inference and serving performance were not measured.

## Reproduce inside an authorized compatible build context

Prepare accepted GEMM and RMSNorm object directories using their existing build scripts. Then build and run the phases in order. Use the assigned physical BDF, not an assumed ordinal.

```sh
export LD_LIBRARY_PATH=/usr/local/neuware/lib64
export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=integration/deepseek python3 runtime/native/layer_fixture.py \
  /app/shuwen/model/DeepSeek-V2-Lite /results/runtime-full-fixtures
runtime/native/build.sh /results/runtime /results/prepacked-final-build /results/rms-build
for phase in r12_test r34_test r5_test; do
  /results/runtime/"$phase" --authorized 0000:4f:00.0 || exit
done
/results/runtime/r6_test --authorized-reference-mla 0000:4f:00.0 /results/runtime-full-fixtures
```

The full-layer program exits 1 while the known parity condition persists. Keep its CSV and stderr, and keep the cleanup evidence even when the numerical gate fails.

[Worker](MLU_DEVICE_WORKER_DESIGN.md) · [Registry](MLU_WEIGHT_REGISTRY_ACCEPTANCE.md) · [Queue/workspace](MLU_QUEUE_WORKSPACE_ACCEPTANCE.md) · [Memory](MLU_MEMORY_MANAGER_ACCEPTANCE.md) · [Layer](MLU_SINGLE_LAYER_RUNTIME_ACCEPTANCE.md)
