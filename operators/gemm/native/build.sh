#!/usr/bin/env bash
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
out=${1:?provide writable output directory}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
cncc="$sdk/bin/cncc"
mkdir -p "$out"
"$cncc" --version
for dtype in fp16 bf16; do
    flags=(--bang-mlu-arch=mtp_592 -O2 -std=c++11 -ffp-contract=off "-I$sdk/include")
    "$cncc" "${flags[@]}" -c "$src/gemm_$dtype.mlu" -o "$out/gemm_$dtype.o"
    "$cncc" "${flags[@]}" --bang-device-only -S "$src/gemm_$dtype.mlu" -o "$out/gemm_$dtype.s"
    readelf -SW "$out/gemm_$dtype.o" | grep -E 'cn_fatbin|cnFatBinSegment'
    nm -g "$out/gemm_$dtype.o" | grep " T mlu_bang_gemm_$dtype$"
    grep -E '^.arch|cvt\.nram\.f32|mul.*f32|add.*f32' "$out/gemm_$dtype.s" | head -25
done
"${CXX:-g++}" -std=c++11 -O2 -Wall -Wextra -Werror -ffp-contract=off \
    -I"$sdk/include" -I"$src/../../../common/reference" -c "$src/test.cpp" -o "$out/test.o"
"$cncc" "$out/test.o" "$out/gemm_fp16.o" "$out/gemm_bf16.o" \
    -L"$sdk/lib64" -lcnrt -lstdc++ -lm -o "$out/gemm_native_test"
file "$out/gemm_native_test"
