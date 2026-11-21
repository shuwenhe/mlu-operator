#!/usr/bin/env bash
set -euxo pipefail
cd "$(dirname "$0")/../../../.."
out=${1:?build directory}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
mkdir -p "$out"
matrix=operators/gemm/native/matrix
pre=operators/gemm/native/prepacked
flags=(--bang-mlu-arch=mtp_592 -O2 -std=c++11 "-I$sdk/include" "-I$matrix")
"$sdk/bin/cncc" --version
"$sdk/bin/cncc" "${flags[@]}" -c "$matrix/matrix_gemm.mlu" -o "$out/fp16.o"
"$sdk/bin/cncc" "${flags[@]}" -DMATRIX_BF16 -c "$matrix/matrix_gemm.mlu" -o "$out/bf16.o"
"$sdk/bin/cncc" "${flags[@]}" -c "$pre/packed_weight.mlu" -o "$out/packed.o"
g++ -no-pie -std=c++11 -O2 -Wall -Wextra -Werror -ffp-contract=off -I"$sdk/include" -I"$matrix" \
    -Icommon/reference "$pre/test.cpp" "$out/fp16.o" "$out/bf16.o" "$out/packed.o" \
    -L"$sdk/lib64" -Wl,-rpath-link,"$sdk/lib64" -Wl,--wrap=cnrtMalloc -lcnrt -lcnnl -o "$out/prepacked_test"
