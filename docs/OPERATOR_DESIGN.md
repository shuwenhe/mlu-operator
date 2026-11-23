# Operator design and hardware acceptance

- [Operator design](#operator-design)
- [MLU590-M9 custom GEMM and RMSNorm design](#native-kernel-design)
- [MLU590-M9 eight-card hardware acceptance](#mlu590-m9-eight-card-hardware-acceptance)
- [MLU590-M9 eight-card operator development plan](#mlu590-m9-eight-card-operator-plan)
- [MLU590-M9 single-card readiness](#mlu590-m9-single-card-readiness)

<a id="operator-design"></a>

## Operator design

P0 uses CNNL from the verified Ubuntu 24.04 package interface. Keep the host and workload configuration unchanged. First compile CPU references and native objects. After an exclusive device reservation and explicit approval, link in a compatible SDK environment and validate CNRT before executing operators.

GEMM accepts packed row-major FP16/BF16 A[M,K] and B[K,N] and returns FP32 D[M,N]. An independent FP32 reference checks rectangular and square outputs. RMSNorm accepts packed FP16/BF16 rows, FP32 gamma and epsilon, and returns FP32 rows. CNNL documents the storage combinations, and its internal reduction precision must be confirmed separately. Both wrappers return the first vendor failure and perform cleanup.

Future vLLM-MLU integration belongs in an authorized development checkout. Use device-resident tensor buffers, the current stream and cached descriptors/workspace, and register a custom op with an explicit dtype and layout contract. Keep host staging in the standalone tool. Compare against the existing compiled implementation under identical dtype, shape and synchronization. Existing serving uses TMO and CNNL, including MatMulEx, batched GEMM and grouped GEMM. Use CNNL rather than adding a custom BANG kernel until a measured gap justifies one.

P1 should compare existing YaRN RoPE, MLA prefill/decode and paged cache paths against the DeepSeek reference. P2 should address softmax top-k, expert gather/combine and grouped GEMM after dense GEMM and MLA pass.

<a id="native-kernel-design"></a>

## MLU590-M9 custom GEMM and RMSNorm design

### Decision and evidence

Keep the existing CNNL operators as explicit baselines. Develop two independent BANG C device kernels in an approved development environment after confirming CNCC lowering for `mtp_592`. Start with a single-task RMSNorm and a dense rectangular GEMM with one task per output tile. Optimize only after device correctness and profiling establish a measurable need.

- **Current GEMM.** `operators/gemm/src/mlu_gemm.c` uses CNRT allocation and copies, CNNL descriptors and heuristic selection, and `cnnlMatMul_v2`. It requests FP32 compute and output for packed FP16/BF16 inputs. CNRT notifiers bracket repeated launches.
- **Current RMSNorm.** `operators/rmsnorm/src/mlu_rmsnorm.c` uses `cnnlRmsNormForward_v2` over the last axis with FP32 gamma, output and saved RMS. Precision inside the vendor reduction remains unverified.
- **Validation status.** The wrappers and harnesses compiled to host objects against SDK headers. The GEMM CPU oracle passed 20 cases. No custom kernel was compiled or run during this investigation. Host linking of the deployed SDK is blocked by its glibc requirements. An authorized exclusive device is still required. [Existing acceptance report](ACCEPTANCE_REPORT.md)
- **Production comparison.** The deployed vLLM-MLU linear path goes from `model_executor/layers/linear.py` to `vllm_mlu/_mlu_ops.py` and `torch_mlu_ops/_ops.py`, then into the TMO extension. Its compiled library contains CNNL `cnnlMatMulEx_v2`, `cnnlBatchMatMulEx` and `cnnlGroupGemm` call evidence. Earlier symbol inspection identified a custom grouped-GEMM offset helper. This is evidence of a custom helper alongside CNNL compute paths. The deployed RMSNorm wrapper delegates to TMO fused layernorm. Its underlying kernel is unresolved. The inspected deployed `csrc/libtorch_stable/cache_kernels.mlu` uses the current MLU queue and CNRT copies, illustrating stream ownership for integration. It does not establish custom GEMM or RMSNorm compute. [Cambricon vLLM-MLU upstream](https://github.com/Cambricon/vllm-mlu)

### Programming APIs and precision

The inspected production image contains CNCC 5.7.6, CNRT 7.7.3 and compiler headers in `/usr/local/neuware/lib/clang/11.1.0/include`. Evidence below comes from those deployed headers and the SDK examples read through Docker archives, without starting a process in the production container.

| Surface | Verified evidence and limit |
| --- | --- |
| Kernel declaration and launch | The installed `samples/BANG/0_Concepts_and_Techniques/convolution/main.mlu` uses `__mlu_global__`, a Block function type, `cnrtDim3_t`, queue launch and synchronization. The device-link example uses `__mlu_device__`. Cambricon's development guide also documents `__mlu_entry__` and `__mlu_func__`. Follow the installed sample's compiler dialect. |
| Task indices | `__clang_bang_builtin_vars.h` exposes `taskId`, `taskDim`, component indices, `coreId` and `clusterId`. Task-stride loops permit a bounded grid to process a larger logical tile count. |
| Local memory and data movement | The convolution example uses `__nram__`, `__wram__`, and `__memcpy` with `GDRAM2NRAM`, `GDRAM2WRAM` and `NRAM2GDRAM`. The guide documents SRAM staging and asynchronous copies. Buffer reuse requires the appropriate completion synchronization for the selected transfer path. |
| FP16/BF16 matmul | `bang_device_functions_decls.h` declares `__bang_matmul(float*, const half*, const half*, ...)` and `__bang_matmul(float*, const bfloat16_t*, const bfloat16_t*, ...)` near lines 15947 and 15959. `bang_bf16.h` names `compute_50` and `compute_60`. These declarations establish a candidate FP32-output path. They do not prove the operand packing, alignment, instruction lowering or internal accumulator precision on MLU590-M9. |
| FP32 RMSNorm path | The same header declares `__bang_half2float`, `__bang_bfloat162float`, FP32 `__bang_reduce_sum` and FP32 `__bang_rsqrt`. Use FP32 conversion before squaring and reducing. Verify the exact reduction signature, vector lengths and rsqrt error for the chosen target. Output starts in FP32, with optional explicit round-to-nearest conversion added under a separate contract. |
| Kernel/runtime interface | The SDK sample sets the device, creates a queue, allocates and copies memory, launches, synchronizes and cleans up via CNRT. Keep those responsibilities in the standalone host driver. Custom kernels require CNCC and CNRT. CNNL remains needed only for the comparison harness. |

Sources for the execution model and examples. [Cambricon BANG C operator guide](https://github.com/Cambricon/mlu-ops/blob/master/docs/BANG-C-OPS-Develop-Guide.md) · [Vendor GEMM lab](https://developer.cambricon.com/index/curriculum/expdetails/id/8/classid/) · [Vendor compiler target guide](https://sdk.cambricon.com/static/independent/CNToolkit/3.8.4/releasenote/3.8.4/build/html/chapter_components/index.html)

### Hardware execution and memory model

GDRAM is device-global memory. NRAM is core-local scratch for vectors and intermediate results. WRAM is core-local storage used by suitable matrix or convolution instructions, often for a packed right-hand operand. SRAM can serve as cluster-shared staging in compatible Union launches. Start with Block tasks and no cluster barriers. A Block task can execute independently. Union resource sizing and synchronization must follow the installed CNRT and CNCC contracts.

Confirm per-target usable scratch, alignment and legal launch sizes through the compiler, documented runtime attributes and a probe. Do not budget storage from an older architecture example or assume the task grid equals physical core count. Measure transfer and compute overlap before adding ping-pong buffers.

### GEMM kernel

1. Preserve the standalone baseline contract `D[M,N] = A[M,K] × B[K,N]`, with packed row-major FP16/BF16 A and B and FP32 D. Validate positive dimensions, byte multiplication and offsets in the host dispatcher. Report unsupported sizes or vendor failures explicitly.
2. Establish a minimal intrinsic probe for each dtype using an aligned tile, one-hot inputs and distinctive rows and columns. Confirm right-operand packing, argument order and output layout. Compile for `mtp_592`, inspect diagnostics and scratch use, then validate the exact output on the reserved hardware.
3. Give each Block task exclusive ownership of a `TM × TN` D tile. For each K slab, stage A in NRAM, pack B into the vendor-required layout, and move it to the required scratch region. Zero-fill invalid lanes in M, N and K tails before the intrinsic. Write only the logical output rectangle. Use wide offset arithmetic and avoid reading adjacent rows outside the logical extent.
4. Produce FP32 slab outputs and add them into FP32 tile storage across K. Budget at least `2×TM×TK + 4×TM×TN` bytes of NRAM for A and the accumulator, plus another FP32 tile if the intrinsic needs separate output, packing buffers and alignment. Budget the packed B slab independently. Begin without split-K. Benchmark prepacked static weights and reuse, then double buffering or cluster sharing only after profiling demonstrates movement overhead.
5. The first test gate requires both FP32 slab accumulation and evidence of sufficient precision inside a slab. If the target intrinsic does not meet the required precision, reduce slab length only if that is documented to help or use a separate FP32 arithmetic reference kernel on device for correctness research. Keep the CNNL route available and label each result by the actual arithmetic mode.

Test both dtypes on small shapes, odd rectangular dimensions, tile boundaries of ±1 and representative DeepSeek shapes, including M=1 decode and projection widths 3072 and 1408. Compare with the independent increasing-K FP32 oracle, and a higher-precision diagnostic reference for cancellation or exponent-range cases. Include k-slab cancellation, signed values, zero, subnormal policy and NaN/Inf detection. Set tolerances per dtype and workload before collecting results. Report max absolute, max relative with a stated near-zero floor, mean absolute error, nonfinite counts and repeatability.

### RMSNorm kernel

1. Accept packed FP16/BF16 rows, FP32 gamma, positive finite epsilon and FP32 output. Map each Block task to one or more rows. Convert loaded lanes to FP32, square and reduce in FP32, compute `inv = rsqrt(sum / width + epsilon)`, and multiply FP32 input by inv and gamma before storage. Divide by logical width so padded zeros do not alter the mean. Use an FP32 scalar square root or a validated FP32 reciprocal-square-root path with a measured error bound.
2. Keep a whole row local when it fits. Reuse gamma across rows assigned to the task when profitable. For widths beyond the scratch budget, accumulate FP32 chunk sums on one task and reread input in a second pass before writing the result. Use FP32 partial sums with a defined order. Add cross-task reduction only where a wide-row benchmark justifies the additional synchronization.
3. Test widths 1, 65, 512, 2048 and 4096 with zero rows, signed inputs, nonuniform gamma, epsilon variation and repeated launches. Test near-constant and large dynamic-range rows. Compare the independent FP32 formula and a higher-precision diagnostic oracle. FP32 squaring can overflow for extreme finite BF16 values. Establish the supported range, then evaluate a scale-first algorithm if wider-range stability is required. Validate subnormal and nonfinite policies explicitly.

### Build and hardware acceptance

Use a digest-pinned, approved development image with matching CNCC, CNAS, CNRT and CNDrv, its own compatible libc, and explicit access to only the reserved card and required control nodes. The deployed Ubuntu 24.04 userspace is evidence for an SDK candidate. The Focal host remains on glibc 2.31 and the deployed libraries require newer symbols. No host upgrade is part of this design.

The installed convolution README demonstrates `cncc --bang-mlu-arch=mtp_592 -O3 -std=c++11 -lstdc++ -lm main.mlu`. For the prototypes, adapt that invocation to the pinned SDK's include and link paths. Retain the compiler and runtime versions, effective target and flags, diagnostics, executable hash, and known precision configuration. Build separate intrinsic probes before full operator kernels. Confirm device allocation and CNRT H2D/D2H round trip before launching either operator.

Hardware acceptance requires successful launch and synchronization, output comparison for both dtypes, repeated runs, cleanup and explicit errors for invalid requests. Capture original vendor statuses, device PCI identity, and evidence of the custom kernel symbol executing via an approved profiler or device trace. Compilation and library loading are insufficient evidence of execution.

Compare custom and CNNL kernels with identical dtype, layout, output precision, input residency and correctness thresholds. Measure synchronized kernel latency separately from weight packing, allocation and transfers. Report warmups, samples, median and tail latency, GEMM TFLOPS, RMSNorm effective bandwidth using an explicit byte-count definition, and peak workspace. CNNL offers a mature dispatch and breadth baseline. Custom kernels are justified by measured gains on fixed shapes, reusable weight packing or fusion, especially small-M decode and residual-plus-RMSNorm. Keep integration opt-in until accuracy and performance gates pass.

<a id="mlu590-m9-eight-card-hardware-acceptance"></a>

## MLU590-M9 eight-card hardware acceptance

Executed on 2026-11-21 on `tza-0a09-ai01-em9`, after the user confirmed formal authorization for all eight cards. The dedicated disposable contexts granted only the requested compute and IPCM nodes and shared control. Existing Team containers retained their mappings and were left untouched. GLM53 remained stopped.

| Phase | Status | Actual evidence |
| --- | --- | --- |
| 1. Inventory and SDK | PASS | Host CNMON returned 0, eight MLU590-M9 reported Good with driver Running. Host PCI addresses in card order were `4f:00.0, 53:00.0, 6f:00.0, 73:00.0, af:00.0, b3:00.0, cf:00.0, d3:00.0`. Direct CNCC executable reported 5.7.6, MLVM 1.4, Clang 11.1.0. The Ubuntu 24.04 image reported installed CNNL 2.2.18 and CNRT 7.7.3. The in-container CNMON filter produced no card lines; CNDEV health evidence is from the host. |
| 2. Independent CNRT smoke | PASS | Runtime version 7.7.3 enumerated ordinals 0–7, each MLU590-M9 with 98,304 MiB. All device selections, properties, 4 KiB allocations, byte-exact H2D/D2H round trips, queue synchronization, frees and queue destroys passed. |
| 3. Multi-card | PASS | Eight simultaneously live buffers and independent queues held distinct patterns. Reverse-order readback matched all patterns and cleanup succeeded. The first negative-control check expected a different invalid-device enum. CNRT returned 100006 for ordinal 8, which the rerun accepted as an invalid-argument rejection. |
| 4. RMSNorm | PASS for fixtures | MLU 0 ran FP16 and BF16 at all five widths. All 17,540 pre-gamma values and final values matched expected bits, normalized FP32 passed, guards passed, and CNNL passed its separately evaluated FP32 contract. |
| 5. GEMM | PASS for fixtures | MLU 2 ran standalone BANG C and CNNL independently. BANG C passed 12 cases and CNNL passed 14, all with zero numerical error on their respective fixtures. |
| 6. Performance | PASS for recorded scopes | Single-card RMSNorm and GEMM timings recorded. Independent BF16 RMSNorm jobs measured 1/2/4/8-card throughput. GEMM scaling and CNCL collectives were outside this measurement. |

[Host inventory](../artifacts/logs/eight_card_inventory.log) · [CNRT smoke](../artifacts/logs/cnrt_smoke_test.log) · [Multi-card probe](../artifacts/logs/cnrt_multi_card.log) · [Initial negative-control log](../artifacts/logs/cnrt_multi_card_first_attempt.log)

### Scaling and cleanup

The [scaling harness](../operators/rmsnorm/native/scaling.cpp) ran one thread and queue per selected CNRT ordinal with resident BF16 256×2048 input. Each card used distinct gamma, five warmups and 100 timed launches. Elapsed time starts at common release and ends at the last queue completion. Allocation, initial transfers, readback and cleanup are excluded. Every readback matched its own weight pattern exactly. This measures aggregate throughput for independent resident jobs.

| Cards | Aggregate rows/s | Speedup |
| ---: | ---: | ---: |
| 1 | 621,780 | 1.000× |
| 2 | 1,242,277 | 1.998× |
| 4 | 2,484,242 | 3.995× |
| 8 | 4,948,901 | 7.959× |

Post-test CNMON reported 0 MiB used on all eight cards. Both Team containers remained running and GLM53 remained exited. Existing operator sources, fixtures and CPU evidence were preserved. Added only independent multi-card and scaling harnesses, reports and hardware evidence. No numerical tolerances changed. Device correctness claims above apply to the fixtures run; no end-to-end model inference was attempted.

[RMSNorm accuracy and timings](MLU590_M9_RMSNORM_HARDWARE_ACCURACY.md#mlu590-m9-rmsnorm-hardware-accuracy) · [GEMM accuracy and performance](GEMM_NATIVE_PERFORMANCE_REPORT.md#mlu590-m9-gemm-hardware-accuracy-performance) · [Scaling CSV](../artifacts/logs/rmsnorm_mlu8_scaling.csv) · [Post-test device inventory](../artifacts/logs/eight_card_after_tests.log)

<a id="mlu590-m9-eight-card-operator-plan"></a>

## MLU590-M9 eight-card operator development plan

### Phase 1 — Read-only audit

Audit date 2026-11-21, host `tza-0a09-ai01-em9`. The user reported an allocation covering all eight cards. The host confirms the cards are currently idle. The reservation's access policy remains unresolved because both Team containers retain effective mappings to the cards. Neither container was changed.

| Check | Status | Evidence |
| --- | --- | --- |
| GLM53 | PASS | `glm53_serve` is exited. Container ID `6361c56e3aea1a4388776ad090634480587f8a21c760b4ff34131173be76d147`. |
| Team 1 | PASS | `ta-mlu-team1`, ID `0ed98a6d307fa064c17cc11be78d6fcd79a13e1cb56a00c8d3cee579db6a9349`, running with nonprivileged runc. Explicit `rwm` mappings for compute and IPCM nodes 0–3 and shared control. `docker top` showed `start.sh`, SSH processes and Python PID 785760. |
| Team 2 | PASS | `ta-mlu-team2`, ID `ffa00877140832f8d953f1975f95c86c6bbf7eacffbad04239545be1a7749c1e`, running with nonprivileged runc. Explicit `rwm` mappings for compute and IPCM nodes 4–7 and shared control. `docker top` showed `start.sh`, SSH and Python PID 652279. |
| Card occupancy | PASS | `cnmon info` exited 0. All eight cards reported MLU590-M9, initialized, 0 MiB used and 97,061 MiB free per card. This is a point-in-time result. |
| Host device files | PASS | Control `509:0`, compute `508:0–7`, and IPCM `505:0–7` were root:root, mode 0666. Existing container mappings can grant access even while CNMON shows 0 MiB used. |
| Host driver and SDK | BLOCKED for host CNRT | Driver module 6.5.52 and glibc 2.31. Host loader inventory showed CNDev. Local Ubuntu 24.04 image contains installed Neuware 4.7.6, CNCC 5.7.6, CNRT 7.7.3, CNDrv 3.7.4, CNNL 2.2.18 and CNCL 1.30.10. Previous builds linked in this image. CNRT runtime execution in the proposed context is pending. |
| Exclusive access and conflicting mappings | BLOCKED | Team 1 still maps cards 0–3 and Team 2 maps cards 4–7. The allocation report alone does not explain how these live mappings coexist with a protected test window. Obtain administrator confirmation of the allocation and enforceable access policy. |

### Phase 2 — Administrator resolution

Record the physical PCI IDs, reservation owner, test window and how the platform prevents concurrent use by the mapped Team workloads. If enforcing the reservation requires changing a Team container, the administrator and workload owner must explicitly authorize that action and coordinate it. Do not modify the Team containers as part of this plan's read-only phase. Do not restart GLM53. Recheck CNMON and the effective mappings after the platform confirms the reservation.

### Phase 3 — Isolated development context

Prepare an isolated runc container from the existing Linux/amd64 image `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`. Use UID/GID 1001:1001, default seccomp, capability drop, no network, a read-only root and workspace, and a writable results directory. No production mounts or host SDK installation are required. Grant only the selected compute nodes, their IPCM companions and the shared control node. Shared control can expose broader monitoring visibility. Verify device access inside each test context. The proposal assumes these support nodes are sufficient, pending runtime verification.

After Phase 2 approval only, a suitable command for the eight-card smoke context is:

```bash
cd /app/shuwen/mlu-operator
results=$(mktemp -d "$PWD/artifacts/build/mlu8-smoke.XXXXXX")
devices=(--device=/dev/cambricon_ctl:/dev/cambricon_ctl:rw)
for i in {0..7}; do
    devices+=(--device="/dev/cambricon_dev$i:/dev/cambricon_dev$i:rw")
    devices+=(--device="/dev/cambricon_ipcm$i:/dev/cambricon_ipcm$i:rw")
done
docker run --rm -it --pull=never --name mlu8-operator-smoke \
    --runtime=runc --user 1001:1001 --network=none --read-only \
    --cap-drop=ALL --security-opt=no-new-privileges \
    --tmpfs /tmp:rw,nosuid,nodev,exec,mode=1777 "${devices[@]}" \
    --mount "type=bind,src=$PWD,dst=/project,readonly" \
    --mount "type=bind,src=$results,dst=/results" \
    --workdir /project --entrypoint /bin/bash \
    sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756
```

Run as the workspace owner and ensure the results directory is writable by UID 1001. Exiting removes only this disposable test container. For later operator runs, create a separate context with the allowlist reduced to the assigned card or pair and a separate output path. Verify CNRT ordinal-to-PCI mapping in each context. Do not infer ordinal identity from the host node number.

### Phase 4 — Eight-card CNRT smoke gate

Inside the approved eight-card context, verify node permissions and allowlist against host PCI identities. Capture CNMON and loader resolution. The existing `common/runtime/cnrt_probe.c` performs device count and properties, selection, queue creation, allocation, a byte-exact 4 KiB H2D/D2H round trip, queue synchronization, free and queue destroy on each visible ordinal. Its copies are synchronous and the queue synchronization check is separate.

```sh
set -eo pipefail
sdk=/usr/local/neuware
export LD_LIBRARY_PATH="$sdk/lib64"
id
ls -l /dev/cambricon*
cnmon info > /results/cnmon-before.log
cc -std=c11 -Wall -Wextra -Werror -I"$sdk/include" \
    common/runtime/cnrt_probe.c -L"$sdk/lib64" -lcnrt -o /tmp/cnrt_probe
ldd /tmp/cnrt_probe | tee /results/loader.log
/tmp/cnrt_probe | tee /results/cnrt-smoke.log
grep -Fx 'device_count=8' /results/cnrt-smoke.log
```

Accept only if the count is eight, all eight PCI identities match the reservation and every ordinal passes all checks. Archive the original CNRT codes and device identities. Stop operator validation on a failed gate. Zero allocator usage and a successful CNDEV query are separate from CNRT acceptance.

### Phases 5–7 — Operator validation

| Host MLU | PCI BDF | Assigned lane | First acceptance |
| --- | --- | --- | --- |
| 0–1 | 0000:4f:00.0, 0000:53:00.0 | RMSNorm FP16/BF16 | Start on MLU 0 alone. Run widths 512, 2048, 65, 127 and 193, compare FP32 intermediates, pre-gamma and final bits, and independently evaluate CNNL. Check canaries and odd-width transfer behavior. Repeat on MLU 1 after correctness. |
| 2–3 | 0000:6f:00.0, 0000:73:00.0 | GEMM FP16/BF16 | Start on MLU 2. Run shape and dtype oracle comparisons for independent BANG C and CNNL. Compare synchronized kernel and transfer-inclusive performance on matched inputs. Repeat on MLU 3. |
| 4–5 | 0000:af:00.0, 0000:b3:00.0 | Attention/MLA | Development lane. Prepare correctness fixtures for prefill, cached decode, YaRN and masking before native acceptance. |
| 6–7 | 0000:cf:00.0, 0000:d3:00.0 | MoE/grouped GEMM | Development lane. Preserve FP32 routing semantics and validate top-k, empty experts, dispatch and combine before native grouped-GEMM comparison. |

Gate RMSNorm first, then GEMM. After single-card gates pass, launch independent jobs on disjoint device allowlists, with distinct results directories and recorded ordinal maps. Measure isolated timings before comparing shared-host parallel runs. Hardware multi-card collectives and cross-device communication require their own CNCL correctness gate. Independent jobs on different cards do not exercise collectives.

### Current acceptance status

- **Compile PASS, prior evidence.** Existing RMSNorm and GEMM BANG C builds generated `mtp_592` device objects. The existing SDK image linked operator harnesses offline.
- **CPU PASS, prior evidence.** Existing RMSNorm PyTorch parity and GEMM reference reports remain the baseline.
- **Actual CNRT smoke, native MLU execution and performance BLOCKED.** No test context was created. No kernels or benchmarks ran, and no existing container or mapping was changed.

[MLU 0 readiness](#mlu590-m9-single-card-readiness) · [RMSNorm acceptance](MLU590_M9_RMSNORM_HARDWARE_ACCURACY.md#deepseek-rmsnorm-hardware-acceptance) · [GEMM build and harness](BANGC_BUILD_AND_INSTRUCTIONS.md#bangc-gemm-build-report)

<a id="mlu590-m9-single-card-readiness"></a>

## MLU590-M9 single-card readiness

**BLOCKED before container creation.** On 2026-11-21, host `tza-0a09-ai01-em9` reported eight initialized MLU590-M9 cards with 0 MiB used each. MLU 0 is PCI `0000:4f:00.0`. The running Team 1 container retains access to MLU 0. Current inactivity does not establish exclusive reservation. The required device policy and administrator authorization for the proposed test environment are still unconfirmed.

### Read-only audit

| Check | Result | Evidence |
| --- | --- | --- |
| GLM53 remains stopped | PASS | `glm53_serve` is exited. |
| Team 1 inventory | PASS | `ta-mlu-team1`, ID `0ed98a6d307fa064c17cc11be78d6fcd79a13e1cb56a00c8d3cee579db6a9349`, running under nonprivileged runc. Same-path `rwm` maps for `cambricon_dev0–3`, `cambricon_ipcm0–3` and `cambricon_ctl`. `docker top` showed `start.sh`, `sshd` and a Python Uvicorn web UI process (host PID 785760). |
| Team 2 inventory | PASS | `ta-mlu-team2`, ID `ffa00877140832f8d953f1975f95c86c6bbf7eacffbad04239545be1a7749c1e`, also nonprivileged runc. Same-path `rwm` maps for `cambricon_dev4–7`, `cambricon_ipcm4–7` and `cambricon_ctl`. Processes were `start.sh`, `sshd` and a Python Uvicorn web UI process (host PID 652279). |
| Device state and permissions | PASS | `cnmon info` exited 0. All eight cards were initialized with the driver running, 0 MiB used and 97,061 MiB free per card. `fuser` on MLU 0 compute/IPCM paths reported no matches as the invoking user. This is a snapshot with limited visibility. All inspected device files were character devices, root:root, mode 0666. The observed decimal major:minor values were ctl 509:0, dev0 508:0 and ipcm0 505:0. |
| Host SDK | BLOCKED for host CNRT/CNNL | Kernel driver 6.5.52, eight procfs firmware fields reporting 6.5.52, and host glibc 2.31. The host Neuware prefix and loader inventory expose CNDev 3.12.4. No host CNRT/CNNL library was found there. The prior Ubuntu 24.04 SDK build evidence records newer glibc dependencies. |
| Existing image SDK | PASS for package and compile evidence | The local Linux/amd64 image is `sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756`. Team 1 package metadata reports Neuware 4.7.6, CNCC 5.7.6, CNDev 6.5.52, CNDrv 3.7.4, CNRT 7.7.3 and CNNL 2.2.18, all Ubuntu 24.04. The independent RMSNorm executable previously linked against this image's SDK. Runtime compatibility in the proposed one-card context remains unverified. |
| Exclusive MLU 0 allocation | BLOCKED | Team 1 retains a live mapping for MLU 0. The platform owner must confirm a reservation covering the test window and how concurrent access is prevented under the existing mapping, with the Team container left untouched. |

Host evidence came from `docker inspect`, `docker top`, `ps`, `stat`, `fuser`, `cnmon info`, `ldconfig -p`, sysfs and procfs. SDK package status was read through a Docker archive request. No command was executed inside a Team container. Docker reports runc with the cgroupfs driver and cgroup v1.

### Proposed isolated environment

After authorization, use the pinned local image without pulling. Run as UID/GID 1001:1001, with default seccomp, all capabilities dropped, no network and a read-only root filesystem. Give the container a read-only view of this workspace and one writable results subdirectory. Map only the MLU 0 compute node and the shared control and card-0 IPCM support nodes. No wildcard device rules, GDR, other-card nodes or production volume mounts. The shared control node may expose host-wide management telemetry. The IPCM node is part of the proposed support set based on the existing per-card mappings. Its necessity for this SDK has not been verified.

Prepare the fixture set in a fresh workspace directory using the existing generator. Once the administrator confirms the proposed access policy and reservation, the following is the proposed container command. It opens a shell so visibility and SDK checks can precede any kernel. Use a fresh results path and verify that its owner is UID 1001 before launch.

```sh
cd /app/shuwen/mlu-operator
results="$PWD/artifacts/build/mlu0-rmsnorm-acceptance"
mkdir -p "$results"
docker run --rm -it --pull=never --name mlu0-rmsnorm-acceptance \
  --runtime=runc --user 1001:1001 --network=none --read-only \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,exec,mode=1777 \
  --device=/dev/cambricon_ctl:/dev/cambricon_ctl:rw \
  --device=/dev/cambricon_dev0:/dev/cambricon_dev0:rw \
  --device=/dev/cambricon_ipcm0:/dev/cambricon_ipcm0:rw \
  --mount "type=bind,src=$PWD,dst=/project,readonly" \
  --mount "type=bind,src=$results,dst=/results" \
  --workdir /project --entrypoint /bin/bash \
  sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756
```

Exiting removes the test container and its temporary mappings. If cleanup is needed, verify the container name and identity before stopping only the dedicated test container. Keep results in the workspace and release the reservation after cleanup.

### CNRT smoke gate

Inside the approved context, verify `id`, the three mapped nodes, the effective device policy and PCI identity. Check SDK SONAME resolution with `ldd`. The shared control node may let `cnmon info` report broader inventory. Require exactly one CNRT-visible device and confirm it matches the reserved MLU 0 before running the existing probe, which tests every visible ordinal.

```sh
set -eo pipefail
sdk=/usr/local/neuware
export LD_LIBRARY_PATH="$sdk/lib64"
id
find /dev -maxdepth 1 -name 'cambricon*' -print
cnmon info > /results/cnmon-before.txt

cat > /tmp/count.c <<'C'
#include <cnrt.h>
#include <stdio.h>
int main(void) {
    unsigned int count = 0;
    cnrtRet_t status = cnrtGetDeviceCount(&count);
    printf("cnrtGetDeviceCount=%u count=%u\n", (unsigned int)status, count);
    return status == cnrtSuccess && count == 1 ? 0 : 1;
}
C
cc -I"$sdk/include" /tmp/count.c -L"$sdk/lib64" -lcnrt -o /tmp/count
ldd /tmp/count
/tmp/count | tee /results/cnrt-count.log
# Confirm the reserved PCI identity from the approved mapping and device properties.
cc -std=c11 -Wall -Wextra -Werror -I"$sdk/include" common/runtime/cnrt_probe.c \
  -L"$sdk/lib64" -lcnrt -o /tmp/cnrt_probe
/tmp/cnrt_probe | tee /results/cnrt-smoke.log
```

The existing probe checks version, count, device selection and properties, queue creation, allocation, a byte-exact 4 KiB H2D/D2H round trip, synchronization, free and queue cleanup. It logs original return codes and fails on any check. Stop before RMSNorm if enumeration, identity, loader checks or the probe fail.

### RMSNorm gate

**BLOCKED.** After the smoke gate passes, use the [five-width hardware procedure](MLU590_M9_RMSNORM_HARDWARE_ACCURACY.md#deepseek-rmsnorm-hardware-acceptance) with `/results` for builds and diagnostic output. Run FP16 and BF16 widths 512, 2048, 65, 127 and 193. Check canaries, normalized FP32 errors, pre-gamma bits, final output against the preserved CPU reference, and the separate CNNL correctness baseline. Benchmark after correctness with five warmups and 30 measured launches, reporting kernel and transfer-inclusive latency separately. No CNRT smoke tests, kernels or benchmarks ran in this phase. The Team containers, GLM53 configuration and existing fixtures were left untouched.
