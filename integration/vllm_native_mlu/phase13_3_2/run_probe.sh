#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
tp=${1:?TP size}
shift
mode=${1:?mode}
case "$tp" in
    1|2|4|8) ;;
    *) printf 'Expected TP size 1, 2, 4 or 8\n' >&2; exit 2 ;;
esac
if [[ "$mode" == reference && "$tp" != 1 ]]; then
    printf 'Reference mode requires TP=1\n' >&2
    exit 2
fi
out="$repo_root/artifacts/build/phase13_3_2"
mkdir -p "$out"
devices=(--device=/dev/cambricon_ctl --device=/dev/cambricon_gdr)
for ((card = 0; card < tp; ++card)); do
    devices+=(--device="/dev/cambricon_dev$card" --device="/dev/cambricon_ipcm$card")
done

timeout 900s docker run --rm --pull=never --network=none --cap-drop=ALL \
    --security-opt=no-new-privileges --user "$(id -u):$(id -g)" --read-only \
    --tmpfs /tmp:rw,exec,mode=1777 --shm-size=1g "${devices[@]}" \
    --mount "type=bind,src=$repo_root,dst=/project,readonly" --workdir /project \
    --mount "type=bind,src=$out,dst=/results" \
    --mount "type=bind,src=$repo_root/../model/DeepSeek-V2-Lite,dst=/checkpoint,readonly" \
    -e HOME=/tmp -e LOGNAME=phase13 -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    -e TRITON_CACHE_DIR=/tmp/triton -e VLLM_CACHE_ROOT=/tmp/vllm \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 \
    -e VLLM_NO_USAGE_STATS=1 -e GLOO_SOCKET_IFNAME=lo \
    --entrypoint /torch/venv3/pytorch_infer/bin/python3 \
    sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756 \
    -m torch.distributed.run --master-addr=127.0.0.1 --master-port=29513 \
    --nproc-per-node="$tp" --module integration.vllm_native_mlu.phase13_3_2.probe \
    "$mode" /results "${@:2}"
