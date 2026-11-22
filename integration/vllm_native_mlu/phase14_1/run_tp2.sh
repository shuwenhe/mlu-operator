#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
out=$(realpath -m -- "${NEURX_PHASE14_1_OUT:-$root/artifacts/logs/phase14_1}")
mkdir -p "$out/tp2_dump"
devices=(--device=/dev/cambricon_ctl --device=/dev/cambricon_gdr)
for card in 0 1; do
    devices+=(--device="/dev/cambricon_dev$card" --device="/dev/cambricon_ipcm$card")
done
lib=/project/artifacts/build/phase12_5_4
timeout 900s docker run --rm --pull=never --network=none --read-only     --tmpfs /tmp:rw,exec,mode=1777 --shm-size=1g --user "$(id -u):$(id -g)" "${devices[@]}"     --mount "type=bind,src=$root,dst=/project,readonly"     --mount "type=bind,src=$root/../model/DeepSeek-V2-Lite,dst=/checkpoint,readonly"     --mount "type=bind,src=$out,dst=/results" --workdir /project     -e PYTHONPATH=/project:/project/integration/vllm_native_mlu/phase14_1     -e HOME=/tmp -e LOGNAME=phase14 -e HF_HOME=/tmp/hf -e TRITON_CACHE_DIR=/tmp/triton     -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor -e VLLM_CACHE_ROOT=/tmp/vllm     -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=1 -e VLLM_NO_USAGE_STATS=1     -e "VLLM_ENABLE_V1_MULTIPROCESSING=${NEURX_ENGINE_MULTIPROC:-0}" -e VLLM_WORKER_MULTIPROC_METHOD=spawn     -e "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=${VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS:-30}" \
    -e "NEURX_TP2_BOUNDARIES=${NEURX_TP2_BOUNDARIES:-1}" \
    -e "VLLM_ALLOW_INSECURE_SERIALIZATION=${NEURX_FAULT_LOCAL_RPC:-0}" \
    -e VLLM_HOST_IP=127.0.0.1 -e GLOO_SOCKET_IFNAME=lo     -e NATIVE_DEEPSEEK_LIBRARIES="$lib/nativelibs" -e NATIVE_DEEPSEEK_ROPE="$lib/rope_yarn"     -e NEURX_ORDERED_O=/project/artifacts/logs/phase13_3_3/libordered.so     -e NEURX_PAGED_EIGHT=/results/libpaged_eight.so -e NEURX_TP2_TRACE=1 -e NEURX_TP2_DUMP=/results/tp2_dump     --entrypoint /bin/bash sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756     integration/vllm_native_mlu/phase14_1/worker.sh "$@"
