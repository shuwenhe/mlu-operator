#!/usr/bin/env bash
set -euo pipefail
sdk=/usr/local/neuware
if [[ "${1:-all}" != deferred ]]; then
    "$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off -fPIC \
        -I"$sdk/include" -Ioperators/gemm/native/matrix -Ioperators/gemm/native/prepacked \
        -c integration/vllm_native_mlu/phase13_3_3/ordered.mlu -o /results/ordered.o
    g++ -shared /results/ordered.o -L"$sdk/lib64" -Wl,-rpath,"$sdk/lib64" -lcnrt -o /results/libordered.so
fi

"$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -std=c++11 -O2 -ffp-contract=off -fPIC \
    -I"$sdk/include" -Ioperators/gemm/native/matrix \
    -c integration/vllm_native_mlu/phase13_3_3/fold_only.mlu -o /results/fold_only.o
g++ -shared /results/fold_only.o -L"$sdk/lib64" -Wl,-rpath,"$sdk/lib64" -lcnrt -o /results/libdeferred.so
