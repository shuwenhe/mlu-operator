#!/usr/bin/env bash
set -euxo pipefail
cd "$(dirname "$0")/../../.."
out=$(realpath -m "${1:?build directory}")
base=${2:?Phase 10 objects}
packed=${3:?accepted packed weight objects}
accepted=${4:?original Phase 11 objects}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
mkdir -p "$out"
cp runtime/native/phase11/{model_run,moe_test}.cpp "$out/"
patch --batch --forward -d "$out" -p0 < runtime/native/phase11_5/runner-router.patch
"$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off -I"$sdk/include" \
  -c runtime/native/phase11_5/router.mlu -o "$out/router.o"
flags=(-no-pie -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -I"$sdk/include"
       -Ioperators/gemm/native/prepacked -Iruntime/native/phase11
       -L"$sdk/lib64" "-Wl,-rpath,$sdk/lib64" "-Wl,-rpath-link,$sdk/lib64")
common=(runtime/native/{runtime,workspace}.cpp "$packed"/{fp16,bf16,packed}.o
        "$accepted"/{pointwise,moe}.o "$out/router.o")
g++ "${flags[@]}" "${common[@]}" runtime/native/phase10/mla.cpp runtime/native/phase11/dense.cpp \
  "$out/model_run.cpp" "$base"/{layout,projection,norm,rope,qk,softmax,av}-{fp16,bf16}.o \
  "$base/kv-cast.o" "$accepted/boundaries.o" -lcnrt -o "$out/model_run"
g++ "${flags[@]}" "${common[@]}" "$out/moe_test.cpp" "$base"/{projection,layout,softmax}-bf16.o \
  -lcnrt -o "$out/moe_test"
