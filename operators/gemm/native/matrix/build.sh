#!/usr/bin/env bash
set -euxo pipefail
cd "$(dirname "$0")/../../../.."
out=${1:?output directory}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
mkdir -p "$out"
"$sdk/bin/cncc" --version
for d in fp16 bf16; do
 flags=(--bang-mlu-arch=mtp_592 -O2 -std=c++11 -ffp-contract=off "-I$sdk/include")
 if [[ $d == bf16 ]]; then flags+=(-DMATRIX_BF16); fi
 "$sdk/bin/cncc" "${flags[@]}" -c operators/gemm/native/matrix/matrix_gemm.mlu -o "$out/$d.o"
 "$sdk/bin/cncc" "${flags[@]}" --bang-device-only -S operators/gemm/native/matrix/matrix_gemm.mlu -o "$out/$d.s"
 grep -E '^\.arch|conv\.nram\.f32' "$out/$d.s"
 readelf -SW "$out/$d.o" | grep cn_fatbin
done
g++ -no-pie -std=c++11 -O2 -Wall -Wextra -Werror -ffp-contract=off -I"$sdk/include" -Icommon/reference \
 operators/gemm/native/matrix/benchmark.cpp "$out/fp16.o" "$out/bf16.o" -L"$sdk/lib64" \
 -Wl,-rpath-link,"$sdk/lib64" -lcnrt -lcnnl -o "$out/benchmark"
