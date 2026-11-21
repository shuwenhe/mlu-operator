#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-cpu}
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
out=artifacts/build
mkdir -p "$out"
cc_bin=${CC:-cc}
flags=(-std=c11 -O2 -Wall -Wextra -Werror -ffp-contract=off)
"$cc_bin" "${flags[@]}" -Icommon/reference operators/gemm/tests/reference_test.c -lm -o "$out/gemm_reference"
case "$mode" in
    cpu) exit 0 ;;
    objects|native) ;;
    *) printf 'usage: %s [cpu|objects|native]\n' "$0" >&2; exit 2 ;;
esac
"$cc_bin" "${flags[@]}" -I"$sdk/include" -c common/runtime/cnrt_probe.c -o "$out/cnrt_probe.o"
if [[ $mode == native ]]; then
    "$cc_bin" "$out/cnrt_probe.o" -L"$sdk/lib64" -Wl,-rpath-link,"$sdk/lib64" -lcnrt -o "$out/cnrt_probe"
fi
for op in gemm rmsnorm; do
    inc=(-Icommon/runtime -Icommon/reference "-Ioperators/$op/include")
    "$cc_bin" "${flags[@]}" "${inc[@]}" -I"$sdk/include" -c \
        "operators/$op/src/mlu_$op.c" -o "$out/${op}_operator.o"
    "$cc_bin" "${flags[@]}" "${inc[@]}" -c \
        "operators/$op/tests/native_test.c" -o "$out/${op}_native_test.o"
    if [[ $mode == native ]]; then
        "$cc_bin" "$out/${op}_operator.o" "$out/${op}_native_test.o" \
            -L"$sdk/lib64" -Wl,-rpath-link,"$sdk/lib64" -lcnnl -lcnrt -lm -o "$out/${op}_native_test"
    fi
done
