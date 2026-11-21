#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
out=${1:?provide a writable build directory}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
mkdir -p "$out"
"$sdk/bin/cncc" --version
for dtype in fp16 bf16; do
    flags=(--bang-mlu-arch=mtp_592 -O2 -std=c++11 -ffp-contract=off "-I$sdk/include")
    if [[ $dtype == bf16 ]]; then flags+=(-DRMSNORM_BF16); fi
    "$sdk/bin/cncc" "${flags[@]}" -c operators/rmsnorm/native/rmsnorm.mlu -o "$out/$dtype.o"
    "$sdk/bin/cncc" "${flags[@]}" --bang-device-only -S operators/rmsnorm/native/rmsnorm.mlu -o "$out/$dtype.s"
    readelf -SW "$out/$dtype.o" | grep cn_fatbin
    nm -g "$out/$dtype.o" | grep " T bang_rmsnorm_$dtype$"
    grep -E '^\.arch|rsqrt|sqrt|active|cvt\.nram\.rn|add\.gpr\.f32|\.kernel' "$out/$dtype.s"
done
"${CC:-cc}" -std=c11 -O2 -Wall -Wextra -Werror -I"$sdk/include" \
    -Icommon/runtime -Ioperators/rmsnorm/include -c operators/rmsnorm/src/mlu_rmsnorm.c -o "$out/cnnl.o"
"${CXX:-g++}" -no-pie -std=c++11 -O2 -Wall -Wextra -Werror -I"$sdk/include" \
    -Icommon/runtime -Ioperators/rmsnorm/include operators/rmsnorm/native/diagnostics.cpp \
    "$out/fp16.o" "$out/bf16.o" "$out/cnnl.o" -L"$sdk/lib64" \
    -Wl,-rpath-link,"$sdk/lib64" -lcnnl -lcnrt -lm -o "$out/rmsnorm_diagnostics"
file "$out/rmsnorm_diagnostics"
