#!/usr/bin/env bash
set -euo pipefail
sdk=/usr/local/neuware
sed 's/constexpr int heads = 16;/constexpr int heads = 8;/'   integration/vllm_native_mlu/phase12/paged_kv.mlu > /results/paged_eight.mlu
"$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -std=c++11 -O2 -fPIC     -c /results/paged_eight.mlu -o /results/paged_eight.o
g++ -shared /results/paged_eight.o -L"$sdk/lib64" -Wl,-rpath,"$sdk/lib64" -lcnrt -o /results/libpaged_eight.so
if [[ "${1:-}" == --benchmark ]]; then
    shift
    export NEURX_TP2_BENCHMARK=1
    unset NEURX_TP2_DUMP NEURX_TP2_TRACE
    exec /torch/venv3/pytorch_infer/bin/python3 -m integration.vllm_native_mlu.phase14_1.benchmark --tp 2 "$@"
fi
exec /torch/venv3/pytorch_infer/bin/python3 -m integration.vllm_native_mlu.phase14_1.engine "$@"
