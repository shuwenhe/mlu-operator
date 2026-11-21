#!/usr/bin/env bash
set -euxo pipefail
out=${1:?isolated output directory}
results=${2:?accepted object directory}
sdk=/usr/local/neuware
source=/project/runtime/native/phase8
cd "$out"
for dtype in fp16 bf16; do
 defs=()
 if [[ $dtype == bf16 ]]; then defs=(-DPHASE8_BF16); fi
 mkdir -p "$dtype"
 (
  cd "$dtype"
  "$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off -save-temps=obj \
    -I/project/operators/gemm/native/matrix "${defs[@]}" -c "$source/candidate_gemm.mlu" -o gemm.o
 )
done
cd /project
g++ -no-pie -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off \
    -I"$sdk/include" -Ioperators/gemm/native/prepacked -Iruntime/native -Ioperators/rmsnorm/native -Icommon/reference \
    runtime/native/{runtime,workspace,dispatch,r6_test}.cpp "$source/candidate_wrap.cpp" \
    "$results"/prepacked-final-build/{fp16,bf16,packed}.o "$results"/rms-build/{fp16,bf16}.o \
    "$results"/runtime/{rope-fp16,rope-bf16,kv-cast}.o "$out"/{fp16,bf16}/gemm.o \
    -Wl,--wrap=_ZN3mlu14gemm_prepackedEPKtRKNS_12PackedWeightEPfiP10CNqueue_st \
    -L"$sdk/lib64" -Wl,-rpath-link,"$sdk/lib64" -lcnrt -lcnnl -o "$out/layer0_candidate"
