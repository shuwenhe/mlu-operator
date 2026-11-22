#!/usr/bin/env bash
set -euo pipefail
output=${1:?output executable}
ops=${2:?directory of QK and AV objects}
softmax=${3:?directory of Softmax objects}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
src=$(cd "$(dirname "$0")" && pwd)
g++ -no-pie -std=c++17 -O2 -Wall -Wextra -Werror -I"$sdk/include" "$src/chain_test.cpp" \
    "$ops/qk-fp16.o" "$ops/qk-bf16.o" "$ops/av-fp16.o" "$ops/av-bf16.o" \
    "$softmax/softmax-fp16.o" "$softmax/softmax-bf16.o" \
    -L"$sdk/lib64" "-Wl,-rpath,$sdk/lib64" "-Wl,-rpath-link,$sdk/lib64" -lcnrt -o "$output"
