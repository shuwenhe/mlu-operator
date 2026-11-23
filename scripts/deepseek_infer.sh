#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
target=${1:-deepseek}
variant=${VARIANT:-stage_c3}
experiment=${DEEPSEEK_EXPERIMENT:-}
mode=infer
case "$target" in
    deepseek|infer) ;;
    deepseek-check|check) mode=check ;;
    deepseek-c2) variant=stage_c2; experiment=output-axis ;;
    deepseek-c3) variant=stage_c3; experiment=fused,batched ;;
    deepseek-tp8) variant=stage_c1; experiment=baseline ;;
    deepseek-benchmark) mode=benchmark ;;
    stage-c2) mode=benchmark; variant=stage_c2; experiment=output-axis ;;
    stage-c3) mode=benchmark; variant=stage_c3; experiment=fused,batched ;;
    moe-audit) mode=probe; variant=stage_c3; experiment=fused,batched ;;
    *) echo "Unknown DeepSeek target: $target" >&2; exit 2 ;;
esac

frozen=sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756
image=${FROZEN_IMAGE:-$frozen}
checkpoint=$(realpath -m "$root/../model/DeepSeek-V2-Lite")
model=$(realpath -m "${MODEL_DIR:-$checkpoint}")
tp=${TP:-8}
dtype=${DTYPE:-bfloat16}
max_tokens=${MAX_TOKENS:-32}
max_model_len=${MAX_MODEL_LEN:-128}
stock_reference=${STOCK_REFERENCE:-0}
capture_logits=${CAPTURE_LOGITS:-0}
extra=()
[[ "$stock_reference" != 1 ]] || extra+=(--stock-reference)
[[ "$capture_logits" != 1 ]] || extra+=(--capture-logits)
[[ ! -v PROMPT_TOKEN_IDS ]] || extra+=(--prompt-token-ids "$PROMPT_TOKEN_IDS")
warm=${WARM_REQUESTS:-2}
prompt=${PROMPT:-"Explain tensor parallel inference on MLU in one short paragraph."}

reject() { echo "$*" >&2; exit 2; }
case "$variant" in
    stage_c1|tp8|stage_c2|stage_c3) ;;
    *) reject "Unsupported DeepSeek variant: $variant" ;;
esac
[[ "$variant" == "tp8" ]] && variant=stage_c1
if [[ -z "$experiment" ]]; then
    case "$variant" in
        stage_c1) experiment=baseline ;;
        stage_c2) experiment=output-axis ;;
        stage_c3) experiment=fused,batched ;;
    esac
fi
[[ "$tp" == 8 && "$dtype" == bfloat16 ]] || reject "DeepSeek requires TP=8 and DTYPE=bfloat16"
[[ "$image" == "$frozen" ]] || reject "DeepSeek requires the accepted frozen image $frozen"
[[ "$max_model_len" =~ ^[1-9][0-9]{0,2}$ ]] && ((max_model_len <= 161)) ||
    reject "MAX_MODEL_LEN must be 1–161"
[[ "$max_tokens" =~ ^[1-9][0-9]{0,2}$ ]] && ((max_tokens < max_model_len)) ||
    reject "MAX_TOKENS must leave room for the prompt in MAX_MODEL_LEN"
if ((max_model_len > 128)) && [[ "$variant" != stage_c3 || "$mode" == benchmark || "$mode" == probe ]]; then
    reject "Extended context requires the Stage C.3 inference entry point"
fi
case "$variant:$experiment" in
    stage_c1:baseline|stage_c2:output-axis|stage_c3:fused,batched) ;;
    stage_c3:*) reject "Stage C.3 requires DEEPSEEK_EXPERIMENT=fused,batched" ;;
    stage_c2:*) reject "Stage C.2 requires DEEPSEEK_EXPERIMENT=output-axis" ;;
    stage_c1:*) reject "Stage C.1 requires DEEPSEEK_EXPERIMENT=baseline" ;;
esac
if [[ "$mode" == benchmark || "$mode" == probe ]]; then
    [[ "$model" == "$checkpoint" ]] || reject "Frozen stage runners require the sibling DeepSeek-V2-Lite checkpoint"
    [[ "$variant" == stage_c2 || "$variant" == stage_c3 ]] || reject "Benchmark supports Stage C.2 and C.3"
    [[ "$warm" =~ ^[0-9]+$ ]] || reject "WARM_REQUESTS must be a nonnegative integer"
fi

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
docker image inspect "$image" >/dev/null || reject "Unable to inspect the frozen image: $image"

log_root=${DEEPSEEK_LOG_ROOT:-$root/artifacts/logs/deepseek}
mkdir -p "$log_root"
run_dir=$(mktemp -d "$(realpath "$log_root")/$target.XXXXXXXX")
printf 'Evidence: %s\n' "$run_dir"
cnmon > "$run_dir/cnmon_before.log" 2>&1
python3 scripts/deepseek_preflight.py "$run_dir/cnmon_before.log" "$run_dir"
printf '%s\n' "mode=$mode" "image=$image" "model=$model" "tp=$tp" "dtype=$dtype" \
    "variant=$variant" "experiment=$experiment" "max_tokens=$max_tokens" "max_model_len=$max_model_len" \
    "stock_reference=$stock_reference" "capture_logits=$capture_logits" | tee "$run_dir/launch.txt"
if [[ "$mode" == check ]]; then
    echo "DeepSeek preflight passed."
    exit 0
fi

phase=integration/vllm_native_mlu/phase15
if [[ "$mode" == probe ]]; then
    launch=(bash "$phase/stage_c3/run.sh" "$run_dir" probe)
elif [[ "$mode" == benchmark && "$variant" == stage_c3 ]]; then
    launch=(bash "$phase/stage_c3/run.sh" "$run_dir" benchmark --experiment "$experiment" --warm-requests "$warm")
elif [[ "$mode" == benchmark ]]; then
    launch=(bash "$phase/stage_c2/run.sh" "$run_dir" --variant candidate --warm-requests "$warm")
else
    python_path=/project:/project/$phase/tp8:/project/$phase/stage_c2:/project/$phase/stage_c3
    launch=(timeout 900s docker run --rm --pull=never --network=none --cap-drop=ALL
        --security-opt=no-new-privileges --user "$(id -u):$(id -g)" --read-only
        --tmpfs /tmp:rw,exec,mode=1777 --shm-size=1g "${devices[@]}"
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
        -m integration.vllm_native_mlu.phase15.deepseek_infer
        --model-dir /checkpoint --variant "$variant" --tp "$tp" --dtype "$dtype"
        --output-dir /results --max-tokens "$max_tokens" --prompt "$prompt" --experiment "$experiment"
        --max-model-len "$max_model_len" "${extra[@]}")
fi

status=0
child_pid=""

interrupt_handler() {
    echo "Interrupted by user (Ctrl+C), stopping DeepSeek child process..." >&2
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -INT "$child_pid" 2>/dev/null || true
        sleep 1
        kill -TERM "$child_pid" 2>/dev/null || true
    fi
}

trap interrupt_handler INT TERM

"${launch[@]}" > "$run_dir/runtime.log" 2>&1 &
child_pid=$!
wait "$child_pid" || status=$?

trap - INT TERM
printf '%s\n' "$status" > "$run_dir/exit_code.txt"
cnmon > "$run_dir/cnmon_after.log" 2>&1
if ((status)); then
    echo "DeepSeek failed with exit=$status. See $run_dir/runtime.log" >&2
    exit "$status"
fi
if [[ "$mode" == infer ]]; then
    python3 - "$run_dir/generation_outputs.json" "$max_tokens" <<'PYCODE'
import json
import sys
from pathlib import Path

result, = json.loads(Path(sys.argv[1]).read_text())
if len(result["generated_token_ids"]) != int(sys.argv[2]):
    sys.exit("Generation evidence has the wrong token count")
print(result["generated_text"])
PYCODE
fi
echo "DeepSeek $mode completed. Evidence: $run_dir"
