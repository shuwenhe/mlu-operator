#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
out="$root/artifacts/logs/phase15/stage_c1"
mode=${1:?Expected unit, distributed, or inference}
shift
devices=(--device=/dev/cambricon_ctl --device=/dev/cambricon_gdr)
for card in {0..7}; do
    devices+=(--device="/dev/cambricon_dev$card" --device="/dev/cambricon_ipcm$card")
done
lib=/project/artifacts/build/phase12_5_4
case "$mode" in
    unit) args=(-m unittest integration.vllm_native_mlu.phase15.tp8.test_contracts
                   integration.vllm_native_mlu.phase14_1.test_contracts
                   integration.vllm_native_mlu.phase13_2.test_shards
                   integration.vllm_native_mlu.phase13_3_2.test_mapping
                   integration.vllm_native_mlu.phase13_4.test_schedule) ;;
    distributed) args=(-m torch.distributed.run --nproc-per-node=8 --master-addr=127.0.0.1
                       --master-port=29513 --module integration.vllm_native_mlu.phase15.tp8.validate) ;;
    inference) args=(-m integration.vllm_native_mlu.phase15.tp8.inference) ;;
    *) exit 2 ;;
esac
timeout 900s docker run --rm --pull=never --network=none --cap-drop=ALL \
    --security-opt=no-new-privileges --user "$(id -u):$(id -g)" --read-only \
    --tmpfs /tmp:rw,exec,mode=1777 --shm-size=1g "${devices[@]}" \
    --mount "type=bind,src=$root,dst=/project,readonly" --workdir /project \
    --mount "type=bind,src=$out,dst=/results" \
    --mount "type=bind,src=$root/../model/DeepSeek-V2-Lite,dst=/checkpoint,readonly" \
    -e PYTHONPATH=/project:/project/integration/vllm_native_mlu/phase14_1:/project/integration/vllm_native_mlu/phase13_2:/project/integration/vllm_native_mlu/phase13_3_2:/project/integration/vllm_native_mlu/phase13_4:/project/integration/vllm_native_mlu/phase15/tp8 \
    -e HOME=/tmp -e LOGNAME=phase15 -e HF_HOME=/tmp/hf -e VLLM_CACHE_ROOT=/tmp/vllm \
    -e TRITON_CACHE_DIR=/tmp/triton -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=1 -e VLLM_NO_USAGE_STATS=1 \
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_HOST_IP=127.0.0.1 -e GLOO_SOCKET_IFNAME=lo -e PYTHONUNBUFFERED=1 \
    -e NATIVE_DEEPSEEK_LIBRARIES="$lib/nativelibs" -e NATIVE_DEEPSEEK_ROPE="$lib/rope_yarn" \
    -e NEURX_ORDERED_O=/project/artifacts/logs/phase13_3_3/libordered.so \
    -e NEURX_PAGED_TWO=/results/libpaged_two.so \
    --entrypoint /torch/venv3/pytorch_infer/bin/python3 \
    sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756 "${args[@]}" "$@"
