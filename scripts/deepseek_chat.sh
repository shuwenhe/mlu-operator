#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

frozen=sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756
image=${FROZEN_IMAGE:-$frozen}
checkpoint=$(realpath -m "$root/../model/DeepSeek-V2-Lite")
model=$(realpath -m "${MODEL_DIR:-$checkpoint}")
tp=${TP:-8}
dtype=${DTYPE:-bfloat16}
max_tokens=${MAX_TOKENS:-32}
max_model_len=${MAX_MODEL_LEN:-128}
temperature=${TEMPERATURE:-0.7}
variant=${VARIANT:-stage_c3}
experiment=${DEEPSEEK_EXPERIMENT:-fused,batched}

reject() { echo "$*" >&2; exit 2; }
[[ "$variant" == "stage_c3" ]] || reject "deepseek-chat requires VARIANT=stage_c3"
[[ "$experiment" == "fused,batched" ]] || reject "deepseek-chat requires DEEPSEEK_EXPERIMENT=fused,batched"
[[ "$tp" == "8" && "$dtype" == "bfloat16" ]] || reject "deepseek-chat requires TP=8 and DTYPE=bfloat16"
[[ "$image" == "$frozen" ]] || reject "deepseek-chat requires frozen image $frozen"
[[ "$max_tokens" =~ ^[1-9][0-9]{0,3}$ ]] || reject "MAX_TOKENS must be a positive integer"
[[ "$max_model_len" =~ ^[1-9][0-9]{0,2}$ ]] && ((max_tokens < max_model_len && max_model_len <= 161)) ||
    reject "Frozen Stage C.3 requires 0 < MAX_TOKENS < MAX_MODEL_LEN <= 161"
[[ "$temperature" =~ ^[0-9]+(\.[0-9]+)?$ ]] || reject "TEMPERATURE must be a non-negative number"

for command in docker cnmon python3; do
    command -v "$command" >/dev/null || reject "Missing required command: $command"
done
for file in config.json model.safetensors.index.json tokenizer.json model-{00001,00002,00003,00004}-of-000004.safetensors; do
    [[ -r "$model/$file" && -s "$model/$file" ]] || reject "Missing checkpoint file: $model/$file"
done
devices=(--device=/dev/cambricon_ctl --device=/dev/cambricon_gdr)
for rank in {0..7}; do
    devices+=(--device="/dev/cambricon_dev$rank" --device="/dev/cambricon_ipcm$rank")
done
for option in "${devices[@]}"; do
    [[ -c "${option#--device=}" ]] || reject "Missing device node: ${option#--device=}"
done
docker image inspect "$image" >/dev/null || reject "Unable to inspect frozen image: $image"

log_root=${DEEPSEEK_CHAT_LOG_ROOT:-$root/artifacts/logs/phase16/runs}
mkdir -p "$log_root"
run_dir=$(mktemp -d "$(realpath "$log_root")/chat.XXXXXXXX")
printf 'Evidence: %s\n' "$run_dir"

cnmon > "$run_dir/device_state_before.log" 2>&1
python3 scripts/deepseek_preflight.py "$run_dir/device_state_before.log" "$run_dir"
printf '%s\n' \
    "mode=chat" "image=$image" "model=$model" "tp=$tp" "dtype=$dtype" \
    "variant=$variant" "experiment=$experiment" "max_tokens=$max_tokens" \
    "max_model_len=$max_model_len" "temperature=$temperature" | tee "$run_dir/launch.txt"

phase=integration/vllm_native_mlu/phase15
python_path=/project:/project/$phase/tp8:/project/$phase/stage_c2:/project/$phase/stage_c3
tty_flags=(-i)
if [[ -t 0 && -t 1 ]]; then
    tty_flags=(-it)
fi
launch=(docker run --rm --pull=never --network=none --cap-drop=ALL
    --security-opt=no-new-privileges --user "$(id -u):$(id -g)" --read-only
    --tmpfs /tmp:rw,exec,mode=1777 --shm-size=1g "${tty_flags[@]}" "${devices[@]}"
    --mount "type=bind,src=$root,dst=/project,readonly" --workdir /project
    --mount "type=bind,src=$run_dir,dst=/results"
    --mount "type=bind,src=$model,dst=/checkpoint,readonly"
    -e "PYTHONPATH=$python_path" -e HOME=/tmp -e LOGNAME=phase15 -e HF_HOME=/tmp/hf
    -e VLLM_CACHE_ROOT=/tmp/vllm -e TRITON_CACHE_DIR=/tmp/triton -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=1 -e VLLM_NO_USAGE_STATS=1
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_WORKER_MULTIPROC_METHOD=spawn
    -e VLLM_HOST_IP=127.0.0.1 -e GLOO_SOCKET_IFNAME=lo -e PYTHONUNBUFFERED=1
    -e DEEPSEEK_EXPERIMENT="$experiment"
    -e NATIVE_DEEPSEEK_LIBRARIES=/project/artifacts/build/phase12_5_4/nativelibs
    -e NATIVE_DEEPSEEK_ROPE=/project/artifacts/build/phase12_5_4/rope_yarn
    -e NEURX_ORDERED_O=/project/artifacts/logs/phase13_3_3/libordered.so
    -e NEURX_PAGED_TWO=/project/artifacts/logs/phase15/stage_c1/libpaged_two.so
    --entrypoint /torch/venv3/pytorch_infer/bin/python3 "$image"
    -m integration.vllm_native_mlu.phase16.deepseek_chat
    --model-dir /checkpoint
    --output-dir /results --max-tokens "$max_tokens" --max-model-len "$max_model_len"
    --temperature "$temperature" --history-policy "${HISTORY_POLICY:-truncate}" --system "${CHAT_SYSTEM_PROMPT:-}")

status=0
"${launch[@]}" || status=$?

printf '%s\n' "$status" > "$run_dir/exit_code.txt"
cnmon > "$run_dir/device_state_after.log" 2>&1
if ((status)); then
    echo "DeepSeek chat failed with exit=$status. See $run_dir" >&2
    exit "$status"
fi
echo "DeepSeek chat completed. Evidence: $run_dir"
