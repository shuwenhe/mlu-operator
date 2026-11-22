#!/usr/bin/env bash
set -euxo pipefail
out=$(realpath -m "${1:?output directory}")
cd "$(dirname "$0")/../.."
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
mkdir -p "$out"
python3 runtime/native/phase82/native_norm.py "$out" --exact-rsqrt
flags=(--bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off -fPIC -I"$sdk/include")
"$sdk/bin/cncc" "${flags[@]}" -DRMSNORM_BF16 -Ioperators/rmsnorm/native -Iruntime/native/phase82 \
  -c "$out/native_norm.mlu" -o "$out/norm.o"
"$sdk/bin/cncc" "${flags[@]}" -c runtime/native/phase11_5/router.mlu -o "$out/router.o"
attention=runtime/native/phase9
for dtype in bf16 fp16; do
    qk_defs=() softmax_defs=() av_defs=()
    if [[ $dtype == bf16 ]]; then
        qk_defs=(-DQK_BF16) softmax_defs=(-DSOFTMAX_BF16) av_defs=(-DAV_BF16)
    fi
    "$sdk/bin/cncc" "${flags[@]}" "${qk_defs[@]}" -c "$attention/attention_qk.mlu" -o "$out/qk-$dtype.o"
    "$sdk/bin/cncc" "${flags[@]}" "${softmax_defs[@]}" -c "$attention/softmax/softmax.mlu" -o "$out/softmax-$dtype.o"
    "$sdk/bin/cncc" "${flags[@]}" "${av_defs[@]}" -c "$attention/av/av.mlu" -o "$out/av-$dtype.o"
done
g++ -shared "$out"/{norm,router}.o "$out"/{qk,softmax,av}-{bf16,fp16}.o \
    -L"$sdk/lib64" -Wl,-rpath,"$sdk/lib64" -lcnrt -o "$out/libnative_mlu.so"
