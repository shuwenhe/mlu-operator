#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
out=${1:?provide a writable build directory}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
mkdir -p "$out"
for op in gemm rmsnorm; do
    "${CC:-cc}" -std=c11 -O2 -Wall -Wextra -Werror -ffp-contract=off \
        -I"$sdk/include" -Icommon/runtime -I"operators/$op/include" \
        -c "operators/$op/src/mlu_$op.c" -o "$out/$op.o"
done
for dtype in fp16 bf16; do
    "$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -O2 -std=c++11 -ffp-contract=off \
        -I"$sdk/include" -c "operators/gemm/native/gemm_$dtype.mlu" -o "$out/bang_$dtype.o"
done
"${CXX:-g++}" -no-pie -std=c++11 -O2 -Wall -Wextra -Werror -ffp-contract=off \
    -I"$sdk/include" -Icommon/runtime -Icommon/reference \
    -Ioperators/gemm/include -Ioperators/gemm/native -Ioperators/rmsnorm/include integration/deepseek/runner.cpp \
    "$out/gemm.o" "$out/rmsnorm.o" "$out/bang_fp16.o" "$out/bang_bf16.o" -L"$sdk/lib64" -Wl,-rpath-link,"$sdk/lib64" \
    -lcnnl -lcnrt -lm -o "$out/deepseek_operators"
