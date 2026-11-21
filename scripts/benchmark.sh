#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --authorized || ! ${2:-} =~ ^[0-7]$ || $# != 2 ]]; then
    printf 'usage after explicit approval and exclusive card reservation: %s --authorized <CNRT ordinal>\n' "$0" >&2
    exit 2
fi
sdk=${NEUWARE_ROOT:-/usr/local/neuware}
export LD_LIBRARY_PATH="$sdk/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
artifacts/build/gemm_native_test "$2" | tee artifacts/performance/gemm.csv
artifacts/build/rmsnorm_native_test "$2" | tee artifacts/accuracy/rmsnorm.csv
