#!/usr/bin/env bash
set -euxo pipefail
cd "$(dirname "$0")/../.."
out=${1:?output}
accepted=${2:?accepted GEMM object directory}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
mkdir -p "$out"
cxx=(g++ -no-pie -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -I"$sdk/include" -Ioperators/gemm/native/prepacked -Iruntime/native)
libs=("$accepted/fp16.o" "$accepted/bf16.o" "$accepted/packed.o" -L"$sdk/lib64" "-Wl,-rpath-link,$sdk/lib64" -lcnrt)
"${cxx[@]}" runtime/native/runtime.cpp runtime/native/r12_test.cpp "${libs[@]}" -o "$out/r12_test"
"${cxx[@]}" runtime/native/runtime.cpp runtime/native/workspace.cpp runtime/native/r34_test.cpp   "${libs[@]}" -Wl,--wrap=cnrtMalloc -Wl,--wrap=cnrtQueueSync -o "$out/r34_test"
if [[ $# -lt 3 ]]; then exit 0; fi
rms=$3
for d in fp16 bf16; do
  defs=(); if [[ $d == bf16 ]]; then defs+=(-DROPE_BF16); fi
  "$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -O2 -std=c++11 -ffp-contract=off "-I$sdk/include"     "${defs[@]}" -c runtime/native/rope.mlu -o "$out/rope-$d.o"
done
"$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -O2 -std=c++11 "-I$sdk/include"   -c integration/deepseek/phase5/kv_cast.mlu -o "$out/kv-cast.o"
bash runtime/native/phase9/softmax/build.sh "$out" --objects-only
bash runtime/native/phase9/av/build.sh "$out" --objects-only
operator_libs=("$out/av-fp16.o" "$out/av-bf16.o" "$out/softmax-fp16.o" "$out/softmax-bf16.o" "$rms/fp16.o" "$rms/bf16.o" "$out/rope-fp16.o" "$out/rope-bf16.o" "$out/kv-cast.o" -lcnnl)
sources=(runtime/native/runtime.cpp runtime/native/workspace.cpp runtime/native/dispatch.cpp)
"${cxx[@]}" -Ioperators/rmsnorm/native "${sources[@]}" runtime/native/r5_test.cpp   "${libs[@]}" "${operator_libs[@]}" -o "$out/r5_test"
"${cxx[@]}" -Ioperators/rmsnorm/native -Icommon/reference "${sources[@]}" runtime/native/r6_test.cpp   "${libs[@]}" "${operator_libs[@]}" -o "$out/r6_test"
"${cxx[@]}" -Ioperators/rmsnorm/native "${sources[@]}" runtime/native/phase9/softmax/dispatch_test.cpp \
    "${libs[@]}" "${operator_libs[@]}" -o "$out/softmax_dispatch_test"
"${cxx[@]}" -Ioperators/rmsnorm/native "${sources[@]}" runtime/native/phase9/av/dispatch_test.cpp \
    "${libs[@]}" "${operator_libs[@]}" -o "$out/av_dispatch_test"
