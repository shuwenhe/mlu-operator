#!/usr/bin/env bash
set -euxo pipefail
cd "$(dirname "$0")/../../.."
out=${1:?output directory}
base=${2:?Phase 10 build directory}
packed=${3:?accepted packed weight objects}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
mkdir -p "$out"
"$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off -I"$sdk/include" \
  -c runtime/native/phase11/pointwise.mlu -o "$out/pointwise.o"
g++ -no-pie -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -I"$sdk/include" \
  -Ioperators/gemm/native/prepacked runtime/native/{runtime,workspace}.cpp runtime/native/phase10/mla.cpp \
  runtime/native/phase11/{dense,test}.cpp "$packed"/{fp16,bf16,packed}.o \
  "$base"/{layout,projection,norm,rope,qk,softmax,av}-{fp16,bf16}.o "$base/kv-cast.o" "$out/pointwise.o" \
  -L"$sdk/lib64" "-Wl,-rpath,$sdk/lib64" "-Wl,-rpath-link,$sdk/lib64" -lcnrt -o "$out/dense_test"
"$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off -I"$sdk/include" \
  -c runtime/native/phase11/boundaries.mlu -o "$out/boundaries.o"
g++ -no-pie -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -I"$sdk/include" \
  -Ioperators/gemm/native/prepacked runtime/native/{runtime,workspace}.cpp \
  runtime/native/phase11/checkpoint_test.cpp "$packed"/{fp16,bf16,packed}.o \
  "$base"/{projection,norm}-bf16.o "$out/boundaries.o" \
  -L"$sdk/lib64" "-Wl,-rpath,$sdk/lib64" "-Wl,-rpath-link,$sdk/lib64" -lcnrt -o "$out/checkpoint_test"
"$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off -I"$sdk/include" \
  -c runtime/native/phase11/moe.mlu -o "$out/moe.o"
g++ -no-pie -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -I"$sdk/include" \
  -Ioperators/gemm/native/prepacked runtime/native/{runtime,workspace}.cpp \
  runtime/native/phase11/moe_test.cpp "$packed"/{fp16,bf16,packed}.o \
  "$base"/{projection,layout,softmax}-bf16.o "$out/pointwise.o" "$out/moe.o" \
  -L"$sdk/lib64" "-Wl,-rpath,$sdk/lib64" "-Wl,-rpath-link,$sdk/lib64" -lcnrt -o "$out/moe_test"
g++ -no-pie -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -I"$sdk/include" \
  -Ioperators/gemm/native/prepacked runtime/native/{runtime,workspace}.cpp runtime/native/phase10/mla.cpp \
  runtime/native/phase11/{dense,model_run}.cpp "$packed"/{fp16,bf16,packed}.o \
  "$base"/{layout,projection,norm,rope,qk,softmax,av}-{fp16,bf16}.o "$base/kv-cast.o" \
  "$out"/{pointwise,boundaries,moe}.o \
  -L"$sdk/lib64" "-Wl,-rpath,$sdk/lib64" "-Wl,-rpath-link,$sdk/lib64" -lcnrt -o "$out/model_run"
