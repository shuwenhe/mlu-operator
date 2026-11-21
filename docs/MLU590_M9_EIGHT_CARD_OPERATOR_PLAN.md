# MLU590-M9 eight-card operator development plan

## Phase 1 — Read-only audit

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

## Phase 2 — Administrator resolution

Record the physical PCI IDs, reservation owner, test window and how the platform prevents concurrent use by the mapped Team workloads. If enforcing the reservation requires changing a Team container, the administrator and workload owner must explicitly authorize that action and coordinate it. Do not modify the Team containers as part of this plan's read-only phase. Do not restart GLM53. Recheck CNMON and the effective mappings after the platform confirms the reservation.

## Phase 3 — Isolated development context

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

## Phase 4 — Eight-card CNRT smoke gate

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

## Phases 5–7 — Operator validation

| Host MLU | PCI BDF | Assigned lane | First acceptance |
| --- | --- | --- | --- |
| 0–1 | 0000:4f:00.0, 0000:53:00.0 | RMSNorm FP16/BF16 | Start on MLU 0 alone. Run widths 512, 2048, 65, 127 and 193, compare FP32 intermediates, pre-gamma and final bits, and independently evaluate CNNL. Check canaries and odd-width transfer behavior. Repeat on MLU 1 after correctness. |
| 2–3 | 0000:6f:00.0, 0000:73:00.0 | GEMM FP16/BF16 | Start on MLU 2. Run shape and dtype oracle comparisons for independent BANG C and CNNL. Compare synchronized kernel and transfer-inclusive performance on matched inputs. Repeat on MLU 3. |
| 4–5 | 0000:af:00.0, 0000:b3:00.0 | Attention/MLA | Development lane. Prepare correctness fixtures for prefill, cached decode, YaRN and masking before native acceptance. |
| 6–7 | 0000:cf:00.0, 0000:d3:00.0 | MoE/grouped GEMM | Development lane. Preserve FP32 routing semantics and validate top-k, empty experts, dispatch and combine before native grouped-GEMM comparison. |

Gate RMSNorm first, then GEMM. After single-card gates pass, launch independent jobs on disjoint device allowlists, with distinct results directories and recorded ordinal maps. Measure isolated timings before comparing shared-host parallel runs. Hardware multi-card collectives and cross-device communication require their own CNCL correctness gate. Independent jobs on different cards do not exercise collectives.

## Current acceptance status

- **Compile PASS, prior evidence.** Existing RMSNorm and GEMM BANG C builds generated `mtp_592` device objects. The existing SDK image linked operator harnesses offline.
- **CPU PASS, prior evidence.** Existing RMSNorm PyTorch parity and GEMM reference reports remain the baseline.
- **Actual CNRT smoke, native MLU execution and performance BLOCKED.** No test context was created. No kernels or benchmarks ran, and no existing container or mapping was changed.

[MLU 0 readiness](MLU590_M9_SINGLE_CARD_READINESS.md) · [RMSNorm acceptance](DEEPSEEK_RMSNORM_HARDWARE_ACCEPTANCE.md) · [GEMM build and harness](BANGC_GEMM_BUILD_REPORT.md)
