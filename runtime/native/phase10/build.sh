#!/usr/bin/env bash
set -euxo pipefail
cd "$(dirname "$0")/../../.."
out=${1:?output directory}
accepted=${2:?accepted packed weight object directory}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
mkdir -p "$out"
sdkflags=(--bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off -I"$sdk/include")
native=runtime/native
python3 "$native/phase82/native_norm.py" "$out" --exact-rsqrt
for dtype in fp16 bf16; do
  defs=(); gemm_defs=(); norm_defs=(); rope_defs=()
  if [[ $dtype == bf16 ]]; then
    defs=(-DMLA_BF16); gemm_defs=(-DPHASE8_BF16)
    norm_defs=(-DRMSNORM_BF16); rope_defs=(-DROPE_BF16)
  fi
  "$sdk/bin/cncc" "${sdkflags[@]}" "${defs[@]}" -c "$native/phase10/layout.mlu" -o "$out/layout-$dtype.o"
  "$sdk/bin/cncc" "${sdkflags[@]}" "${gemm_defs[@]}" -Ioperators/gemm/native/matrix \
    -c "$native/phase8/candidate_gemm.mlu" -o "$out/projection-$dtype.o"
  "$sdk/bin/cncc" "${sdkflags[@]}" "${norm_defs[@]}" -Ioperators/rmsnorm/native -I"$native/phase82" \
    -c "$out/native_norm.mlu" -o "$out/norm-$dtype.o"
  "$sdk/bin/cncc" "${sdkflags[@]}" "${rope_defs[@]}" -c "$native/rope.mlu" -o "$out/rope-$dtype.o"
done
"$sdk/bin/cncc" "${sdkflags[@]}" -c integration/deepseek/phase5/kv_cast.mlu -o "$out/kv-cast.o"
bash "$native/phase9/build_qk.sh" "$out"
bash "$native/phase9/softmax/build.sh" "$out" --objects-only
bash "$native/phase9/av/build.sh" "$out" --objects-only
g++ -no-pie -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off -I"$sdk/include" \
    -Ioperators/gemm/native/prepacked "$native"/{runtime,workspace}.cpp "$native"/phase10/{mla,test}.cpp \
    "$accepted"/{fp16,bf16,packed}.o "$out"/{layout,projection,norm,rope,qk,softmax,av}-{fp16,bf16}.o \
    "$out/kv-cast.o" -L"$sdk/lib64" "-Wl,-rpath,$sdk/lib64" "-Wl,-rpath-link,$sdk/lib64" \
    -lcnrt -o "$out/mla_test"
