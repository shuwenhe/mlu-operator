#!/usr/bin/env bash
set -euxo pipefail
out=$(realpath -m "${1:?output directory}")
cd "$(dirname "$0")/../../.."
mkdir -p "$out"
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
matrix=operators/gemm/native/matrix
packed=operators/gemm/native/prepacked
flags=(--bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off -fPIC -I"$sdk/include" -I"$matrix")
"$sdk/bin/cncc" "${flags[@]}" -c "$matrix/matrix_gemm.mlu" -o "$out/fp16.o"
"$sdk/bin/cncc" "${flags[@]}" -DMATRIX_BF16 -c "$matrix/matrix_gemm.mlu" -o "$out/bf16.o"
"$sdk/bin/cncc" "${flags[@]}" -c "$packed/packed_weight.mlu" -o "$out/packed.o"
"$sdk/bin/cncc" "${flags[@]}" -DPHASE8_BF16 -c runtime/native/phase8/candidate_gemm.mlu -o "$out/projection.o"
"$sdk/bin/cncc" "${flags[@]}" -DMLA_BF16 -c runtime/native/phase10/layout.mlu -o "$out/bf16-layout.o"
g++ -shared -fPIC -std=c++11 -O2 -Wall -Wextra -Werror -ffp-contract=off \
    -I"$packed" -I"$sdk/include" integration/vllm_native_mlu/phase12_4/projection.cpp \
    "$out"/{bf16,fp16,packed,projection,bf16-layout}.o -L"$sdk/lib64" -Wl,-rpath,"$sdk/lib64" \
    -lcnrt -o "$out/libprojection.so"
