#!/usr/bin/env bash
set -euxo pipefail
out=$(realpath -m "${1:?output directory}")
cd "$(dirname "$0")/../../.."
mkdir -p "$out"
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
flags=(--bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off -fPIC -I"$sdk/include")
for name in moe pointwise boundaries; do
    "$sdk/bin/cncc" "${flags[@]}" -c "runtime/native/phase11/$name.mlu" -o "$out/$name.o"
done
"$sdk/bin/cncc" "${flags[@]}" -DROPE_BF16 -c runtime/native/rope.mlu -o "$out/rope.o"
"$sdk/bin/cncc" "${flags[@]}" -c integration/deepseek/phase5/kv_cast.mlu -o "$out/kv-cast.o"
"$sdk/bin/cncc" "${flags[@]}" -c runtime/native/phase10/layout.mlu -o "$out/layout.o"
g++ -shared "$out"/{moe,pointwise,boundaries,rope,kv-cast,layout}.o -L"$sdk/lib64" \
    -Wl,-rpath,"$sdk/lib64" -lcnrt -o "$out/libdecoder.so"
