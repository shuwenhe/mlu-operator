# Native MLU runtime design and acceptance

- [DeepSeek MLU native runtime design](#mlu-runtime-design)
- [Native MLU runtime implementation](#mlu-runtime-implementation)
- [MLU device worker](#mlu-device-worker-design)
- [MLU packed-weight registry acceptance](#mlu-weight-registry-acceptance)
- [Queue and workspace acceptance](#mlu-queue-workspace-acceptance)
- [MLU memory manager acceptance](#mlu-memory-manager-acceptance)

<a id="mlu-runtime-design"></a>

## DeepSeek MLU native runtime design

### Ownership and scheduling

Use one explicit worker context per assigned physical MLU and record its PCI identity and CNRT ordinal. Each context owns model metadata, a map of immutable packed weights, persistent layer buffers and queue handles. The registry key is the actual checkpoint revision, tensor identity, shard/offset, layout, dtype and device. Invalidate it when loading a different checkpoint. No process-global cache is needed.

During preparation, transpose checkpoint row-major `[N,K]` to logical `[K,N]` and pack once. Host pack can publish after its synchronous upload. Device pack remains asynchronous, so publish after its queue completes or after explicit event dependency. Record the logical and padded packed sizes. Queue-owning execution code selects the weight's device and retains both request scratch and weights until all dependent work completes. Concurrent read-only queue use is allowed after publication. Never free a referenced handle. Require successful drain or explicit runtime recovery before reclaiming buffers on error. Propagate raw CNRT/CNNL status with device, op, tensor, layer and request context.

Preallocate typed workspace per queue or request slot. Reuse descriptors and scratch only after the owning completion boundary. Keep FP32 GEMM output staging separate from BF16 activation buffers. Reserve cache blocks independently from short-lived activation scratch. Track persistent bytes, cache bytes, peak live workspace and allocator reserve separately.

### Dispatch and increment

1. Start with one device and explicit opt-in native projections and RMSNorm. Verify shape, stride, dtype and ownership before dispatch. Preserve CNNL as an independent selectable path. Do not silently substitute it for a failed native call.
2. The implemented KV subgraph passes persistent weights, cast and normalization on one queue. To extend it, carry the shared 64-wide rotary key into the validated standalone rotation interface, then publish cache entries after writes are ordered. Embed and resolve fixed input token IDs, validate layer intermediates, and test one prefill followed by token-at-a-time decode. Integrate MLA only after low-precision parity and cache append/rollback checks. Keep FP32 router, optional collectives and unsupported operators on explicit approved backends.
3. Layer plans own transient views, normalized latent layout, backend choice and output semantics. Integrate the remaining output projection, residuals, expert dispatch, gated MLP, final norm, logits and sampling before measuring end-to-end output. A vLLM adapter would bind these operations to existing device-resident tensors and the current queue in a separate development checkout. The production deployment is unchanged.

### Acceptance protocol

Compare identical BF16 checkpoint weights, token IDs, batch sizes, prompt and generation lengths, sampling policy, runtime versions and card placement. Collect setup/load time separately. Report TTFT to first token and TPOT after the first token, plus prefill tokens/s, decode tokens/s, aggregate tokens/s, latency distributions and peak device allocation. Establish single-card token/logit parity before multi-card comparison. Independent throughput and model sharding require separate measurements. None of these inference-level metrics was collected in Phase 5. [Integration status](DEEPSEEK_V2_LITE_E2E_ACCEPTANCE.md#phase-5-mlu-integration)

<a id="mlu-runtime-implementation"></a>

## Native MLU runtime implementation

The isolated [runtime source](../runtime/native) implements device workers, explicitly published packed weights, request slots, memory accounting and operator dispatch above the accepted kernels. It does not change production vLLM, NeurX, Team mappings, existing GEMM/RMSNorm kernels or earlier fixtures.

| Phase | Status |
| --- | --- |
| R1/R2 | Hardware PASS for PCI resolution, device properties, explicit publication, revision reload, leased invalidation and cleanup. |
| R3/R4 | Hardware PASS for two queues/slots, separate buffers, early-release prevention, fault injection and recovery, reserve policy and zero live allocations. Faults were injected, not caused by physical OOM or driver failure. |
| R5 | Hardware PASS for explicit Native/CNNL GEMM and RMSNorm plus native RoPE and native KV dispatch. CNNL RoPE is explicitly unsupported. The connected KV path uses the Native route; CNNL is not substituted upon error. |
| R6 | Full layer zero executed in BF16 with real weights and tokens, with CPU reference MLA and CPU SwiGLU/residuals explicitly selected. Layer numerical acceptance **FAIL**, 404 final elements outside existing tolerance across two repetitions. |
| R7 | Repeated execution, distinct cache storage, two queues, registry reload, injected allocation/queue errors and explicit reclamation observed. Model-level acceptance stays **FAIL** because of R6 parity. |

The current CNNL adapter creates transient descriptors and temporary buffers on each call, and its output contract can differ from native RMSNorm. For production use, add descriptor and queue-workspace reuse, verified caller-thread enforcement for all public metadata operations, consolidated packed-weight budget enforcement, fault recovery orchestration, and paged cache lifetime validation. These tests are single-card and single-layer. Full-model inference and serving performance were not measured.

### Reproduce inside an authorized compatible build context

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

[Worker](#mlu-device-worker-design) · [Registry](#mlu-weight-registry-acceptance) · [Queue/workspace](#mlu-queue-workspace-acceptance) · [Memory](#mlu-memory-manager-acceptance) · [Layer](LAYER0_RESIDUAL_ROOT_CAUSE.md#mlu-single-layer-runtime-acceptance)

<a id="mlu-device-worker-design"></a>

## MLU device worker

[Worker and registry source](../runtime/native/runtime.cpp) binds one context to one owner thread and physical PCI identifier. Construction resolves the BDF through `cnrtDeviceGetByPCIBusId`, reads it back with `cnrtDeviceGetPCIBusId`, and records CNRT properties. The ordinal is scoped to the device mapping in that process. Every queue operation reselects that worker's ordinal and rejects calls originating on another thread. Queues are indexed within the worker and destroyed only after synchronization. Metadata and registry operations are intended to run on the same owning thread.

Lifecycle is construction, ready, faulted, recovered or closed. A failed synchronization stores the original CNRT status and blocks queue acquisition and managed allocation. Recovery checks all surviving queues before clearing the recorded fault. If a workspace cannot guarantee safe teardown on destruction, the process terminates without freeing potentially referenced resources. This is a fail-stop behavior. Runtime restart and unrecoverable device-fault remediation are external responsibilities.

**Hardware PASS, R1.** In a disposable SDK context mapping one card, the worker resolved `0000:4f:00.0` to ordinal 0 and reported MLU590-M9, 98,304 MiB. Both queue initialization and destruction were exercised. The image uses CNRT 7.7.3/CNNL 2.2.18 and CNCC 5.7.6. Existing Team mappings and software were untouched.

[R1/R2 hardware log](../artifacts/logs/runtime_r12_hardware.log)

<a id="mlu-weight-registry-acceptance"></a>

## MLU packed-weight registry acceptance

The registry key comprises revision, tensor identity, shard, byte offset, logical layout, dtype, canonical device PCI identity, K and N. The runtime fixture uses the checkpoint index SHA-256 as its local revision key, with actual shard names and payload offsets. It does not claim a full checkpoint content hash. Published weights are immutable views. Each lookup returns an aliasing reference-counted lease. Invalidation rejects subsequent lookups and retains old entries while leases exist.

Host packing is synchronous. Device packing submits to a worker-owned queue and is initially invisible to readers. Explicit publication synchronizes that queue before exposing the handle. The caller keeps the source live through publication and must also keep it live if submission fails after allocation. Pending invalidated entries synchronize their packing queue before collection. Request dispatch pins leases until its queue completes. The registry may reclaim only invalidated entries whose leases have ended. Multiple revisions can coexist. Duplicate exact keys are rejected.

**Hardware PASS, R2.** Host packing, asynchronous device packing, hidden pre-publication entry, publication, two-revision visibility, invalidation with a live lease, collection after lease release, and zero registry bytes were exercised. The dispatch hardware test additionally invalidated a weight after GEMM submission and confirmed that it was reclaimed only after request completion. Source buffers, queue and packed allocations were released in the test.

[R1/R2 transcript](../artifacts/logs/runtime_r12_hardware.log) · [Final suite](../artifacts/logs/runtime_final_hardware.log)

<a id="mlu-queue-workspace-acceptance"></a>

## Queue and workspace acceptance

[Workspace](../runtime/native/workspace.cpp) provisions separately allocated low-precision activation and FP32 scratch per slot. Each slot belongs to one worker queue, rejects re-entrance while active, and retains input/output allocation pins, packed-weight leases and temporary CNNL resources through completion. The dispatch layer accepts only whole owned allocation bases with sufficient capacity and contiguous row-major strides. Buffer reclamation fails while any active request holds a pin. CNNL device dispatch uses the caller's queue and synchronizes before returning from its operator. Native device dispatch normally enqueues. Native RoPE additionally drains its queue before a host validation read of the position indices.

**Hardware PASS, R3.** Two queues and distinct request slots were created. A fault injected into `cnrtQueueSync` left the request pending and preserved the raw error. Recovery reran real synchronization before release and reuse. The native GEMM dispatch test confirmed early output release is rejected. The connected decoder fixture submitted the two requests before synchronizing either. No actual driver corruption or queue hang was induced.

[R3/R4 transcript](../artifacts/logs/runtime_r34_hardware.log) · [Final suite](../artifacts/logs/runtime_final_hardware.log)

<a id="mlu-memory-manager-acceptance"></a>

## MLU memory manager acceptance

[Memory manager](../runtime/native/workspace.cpp) tracks explicitly owned allocations in weight, cache and workspace buckets, with live and peak counters and a configurable driver free-memory floor. The floor is a policy reserve, not an allocation or measurement of CNRT internal reservation. Driver total/free is captured independently with `cnrtMemGetInfo`. The packed registry reports a separate byte total because allocations occur in the accepted packing API. The managed peak excludes registry bytes and vendor internal allocations. Reserve enforcement currently applies only to managed allocation calls.

**Hardware PASS, R4.** The small failure-injection test forced the second slot allocation to return `cnrtErrorNoMem` and observed rollback to zero live bytes. It measured 288 peak managed bytes with the configured 4,096-byte reserve floor. It exercised distinct weight/cache/workspace buckets and released all managed memory. Actual exhaustion was not provoked. Queue fault recovery used an injected sync status and subsequent real synchronization. See the single-layer report for real checkpoint allocation sizes and the final residual counters.

[R3/R4 transcript](../artifacts/logs/runtime_r34_hardware.log) · [Final suite](../artifacts/logs/runtime_final_hardware.log)
