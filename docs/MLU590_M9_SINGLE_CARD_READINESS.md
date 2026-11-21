# MLU590-M9 single-card readiness

**BLOCKED before container creation.** On 2026-11-21, host `tza-0a09-ai01-em9` reported eight initialized MLU590-M9 cards with 0 MiB used each. MLU 0 is PCI `0000:4f:00.0`. The running Team 1 container retains access to MLU 0. Current inactivity does not establish exclusive reservation. The required device policy and administrator authorization for the proposed test environment are still unconfirmed.

## Read-only audit

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

## Proposed isolated environment

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

## CNRT smoke gate

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

## RMSNorm gate

**BLOCKED.** After the smoke gate passes, use the [five-width hardware procedure](DEEPSEEK_RMSNORM_HARDWARE_ACCEPTANCE.md) with `/results` for builds and diagnostic output. Run FP16 and BF16 widths 512, 2048, 65, 127 and 193. Check canaries, normalized FP32 errors, pre-gamma bits, final output against the preserved CPU reference, and the separate CNNL correctness baseline. Benchmark after correctness with five warmups and 30 measured launches, reporting kernel and transfer-inclusive latency separately. No CNRT smoke tests, kernels or benchmarks ran in this phase. The Team containers, GLM53 configuration and existing fixtures were left untouched.
