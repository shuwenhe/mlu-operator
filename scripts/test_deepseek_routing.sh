#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

assert_contains() {
    local haystack="$1"
    local needle="$2"
    if [[ "$haystack" != *"$needle"* ]]; then
        echo "ASSERTION FAILED: expected [$needle]" >&2
        exit 1
    fi
}

assert_not_contains() {
    local haystack="$1"
    local needle="$2"
    if [[ "$haystack" == *"$needle"* ]]; then
        echo "ASSERTION FAILED: unexpected [$needle]" >&2
        exit 1
    fi
}

dry_deepseek=$(make -n deepseek)
dry_tp8=$(make -n deepseek-tp8)
dry_c2=$(make -n deepseek-c2)
dry_c3=$(make -n deepseek-c3)
dry_bench=$(make -n deepseek-benchmark)
dry_chat=$(make -n deepseek-chat)

assert_contains "$dry_deepseek" 'VARIANT="stage_c3"'
assert_contains "$dry_deepseek" 'DEEPSEEK_EXPERIMENT="fused,batched"'
assert_contains "$dry_tp8" 'VARIANT="stage_c1"'
assert_contains "$dry_tp8" 'DEEPSEEK_EXPERIMENT="baseline"'
assert_contains "$dry_c2" 'VARIANT="stage_c2"'
assert_contains "$dry_c2" 'DEEPSEEK_EXPERIMENT="output-axis"'
assert_contains "$dry_c3" 'VARIANT="stage_c3"'
assert_contains "$dry_c3" 'DEEPSEEK_EXPERIMENT="fused,batched"'
assert_contains "$dry_bench" 'VARIANT="stage_c3"'
assert_contains "$dry_bench" 'DEEPSEEK_EXPERIMENT="fused,batched"'
assert_contains "$dry_chat" 'VARIANT="stage_c3"'
assert_contains "$dry_chat" 'DEEPSEEK_EXPERIMENT="fused,batched"'

if DEEPSEEK_EXPERIMENT=fused,batched VARIANT=stage_c1 scripts/deepseek_infer.sh infer >/dev/null 2>&1; then
    echo "ASSERTION FAILED: stage_c1 accepted fused,batched" >&2
    exit 1
fi

if DEEPSEEK_EXPERIMENT=baseline VARIANT=stage_c2 scripts/deepseek_infer.sh infer >/dev/null 2>&1; then
    echo "ASSERTION FAILED: stage_c2 accepted baseline" >&2
    exit 1
fi

if DEEPSEEK_EXPERIMENT=output-axis VARIANT=stage_c3 scripts/deepseek_infer.sh infer >/dev/null 2>&1; then
    echo "ASSERTION FAILED: stage_c3 accepted output-axis" >&2
    exit 1
fi

# Environment precedence and target isolation.
overridden=$(DEEPSEEK_EXPERIMENT=baseline make -n deepseek-c3)
assert_contains "$overridden" 'DEEPSEEK_EXPERIMENT="fused,batched"'
assert_not_contains "$overridden" 'DEEPSEEK_EXPERIMENT="baseline"'

script_body=$(< scripts/deepseek_infer.sh)
assert_contains "$script_body" '-e DEEPSEEK_EXPERIMENT="$experiment"'
assert_contains "$script_body" '--experiment "$experiment"'

runner_body=$(< integration/vllm_native_mlu/phase15/deepseek_infer.py)
assert_contains "$runner_body" 'requires experiment='

echo "DeepSeek routing static tests: PASS"
