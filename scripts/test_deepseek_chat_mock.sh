#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m unittest discover -s integration/vllm_native_mlu/phase16 -p "test_*.py"
for setting in MAX_MODEL_LEN=4096 MAX_TOKENS=512 VARIANT=stage_c2 DEEPSEEK_EXPERIMENT=baseline; do
    if env "$setting" bash scripts/deepseek_chat.sh </dev/null; then
        echo "Accepted invalid chat setting: $setting" >&2
        exit 1
    fi
done
