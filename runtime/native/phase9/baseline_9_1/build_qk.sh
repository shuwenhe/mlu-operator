#!/usr/bin/env bash
set -euo pipefail

out=${1:?output directory required}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
src=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"

flags=(--bang-mlu-arch=mtp_592 -O2 -std=c++11 -ffp-contract=off -I"$sdk/include")
"$sdk/bin/cncc" "${flags[@]}" -c "$src/attention_qk.mlu" -o "$out/qk-fp16.o"
"$sdk/bin/cncc" "${flags[@]}" -DQK_BF16 -c "$src/attention_qk.mlu" -o "$out/qk-bf16.o"
g++ -no-pie -std=c++11 -O2 -Wall -Wextra -Werror -I"$sdk/include" \
    "$src/qk_device_test.cpp" "$out/qk-fp16.o" "$out/qk-bf16.o" \
    -L"$sdk/lib64" "-Wl,-rpath,$sdk/lib64" "-Wl,-rpath-link,$sdk/lib64" \
    -lcnrt -o "$out/qk_device_test"
