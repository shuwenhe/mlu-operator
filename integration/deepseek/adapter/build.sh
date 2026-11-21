#!/usr/bin/env bash
set -euxo pipefail
out=${1:?results directory with accepted runtime objects}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
adapter=integration/deepseek/adapter
g++ -no-pie -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off \
    -I"$sdk/include" -Iruntime/native -Ioperators/gemm/native/prepacked -Ioperators/rmsnorm/native \
    runtime/native/{runtime,workspace,dispatch}.cpp "$adapter"/{adapter,acceptance}.cpp \
    "$out"/prepacked-final-build/{fp16,bf16,packed}.o "$out"/rms-build/{fp16,bf16}.o \
    "$out"/runtime/{rope-fp16,rope-bf16,kv-cast}.o \
    -L"$sdk/lib64" "-Wl,-rpath-link,$sdk/lib64" -lcnrt -lcnnl -o "$out/deepseek_adapter"
