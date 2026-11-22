#!/usr/bin/env bash
set -euo pipefail
out=${1:?output directory required}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
src=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
flags=(--bang-mlu-arch=mtp_592 -O2 -std=c++11 -ffp-contract=off -I"$sdk/include")
"$sdk/bin/cncc" "${flags[@]}" -c "$src/softmax.mlu" -o "$out/softmax-fp16.o"
"$sdk/bin/cncc" "${flags[@]}" -DSOFTMAX_BF16 -c "$src/softmax.mlu" -o "$out/softmax-bf16.o"
if [[ ${2:-} == --objects-only ]]; then exit 0; fi
g++ -no-pie -std=c++11 -O2 -Wall -Wextra -Werror -I"$sdk/include" "$src/test.cpp" \
    "$out/softmax-fp16.o" "$out/softmax-bf16.o" -L"$sdk/lib64" \
    "-Wl,-rpath,$sdk/lib64" "-Wl,-rpath-link,$sdk/lib64" -lcnrt -o "$out/softmax_test"
