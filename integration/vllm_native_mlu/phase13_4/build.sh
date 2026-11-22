#!/usr/bin/env bash
set -euo pipefail
sdk=/usr/local/neuware
sed 's/constexpr int heads = 16;/constexpr int heads = 2;/' \
  integration/vllm_native_mlu/phase12/paged_kv.mlu > /results/paged_two.mlu
"$sdk/bin/cncc" --bang-mlu-arch=mtp_592 -std=c++11 -O2 -fPIC \
  -c /results/paged_two.mlu -o /results/paged_two.o
g++ -shared /results/paged_two.o -L"$sdk/lib64" -Wl,-rpath,"$sdk/lib64" -lcnrt -o /results/libpaged_two.so
