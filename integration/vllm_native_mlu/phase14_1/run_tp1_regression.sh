#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
snapshot="$root/artifacts/logs/phase14_1/frozen"
out=$(realpath -m -- "${NEURX_PHASE14_1_OUT:-$root/artifacts/logs/phase14_1}")
mkdir -p "$out"
mkdir "$out/tp1_dump"
legacy=integration/vllm_native_mlu/phase12_5
lib=/project/artifacts/build/phase12_5_4
timeout 900s docker run --rm --pull=never --network=none --read-only     --tmpfs /tmp:rw,exec,mode=1777 --shm-size=1g --user "$(id -u):$(id -g)"     --device=/dev/cambricon_ctl --device=/dev/cambricon_gdr --device=/dev/cambricon_dev0     --device=/dev/cambricon_ipcm0     --mount "type=bind,src=$root,dst=/project,readonly"     --mount "type=bind,src=$root/../model/DeepSeek-V2-Lite,dst=/checkpoint,readonly"     --mount "type=bind,src=$snapshot/model.py,dst=/project/$legacy/model.py,readonly"     --mount "type=bind,src=$snapshot/run.py,dst=/project/$legacy/run.py,readonly"     --mount "type=bind,src=$out,dst=/results" \
    --workdir /project -e PYTHONPATH=/project -e HOME=/tmp -e LOGNAME=phase14     -e HF_HOME=/tmp/hf -e TRITON_CACHE_DIR=/tmp/triton -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor     -e VLLM_CACHE_ROOT=/tmp/vllm -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=1     -e VLLM_NO_USAGE_STATS=1 -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_HOST_IP=127.0.0.1     -e NATIVE_DEEPSEEK_LIBRARIES="$lib/nativelibs" -e NATIVE_DEEPSEEK_ROPE="$lib/rope_yarn"     -e NATIVE_DEEPSEEK_TRACE=1 -e NATIVE_DEEPSEEK_DUMP=/results/tp1_dump \
    -e NATIVE_DEEPSEEK_PROMPTS=/project/artifacts/build/vllm_mlu/phase12_5/prompts \
    --entrypoint /bin/bash \
    sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756 -ec '
python=/torch/venv3/pytorch_infer/bin/python3
"$python" -m integration.vllm_native_mlu.phase12_5.run
"$python" -m integration.vllm_native_mlu.phase12_5.compare /project/artifacts/build/vllm_mlu/phase12_5 /results/tp1_dump /results/tp1_accuracy.csv reference
'
