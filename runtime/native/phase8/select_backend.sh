#!/usr/bin/env bash
set -euo pipefail

backend=${1:?native_default, native_avx2_compatible, or cnnl_reference}
results=${2:?SDK results mount}
candidate=${3:?path to isolated candidate build}
shift 3

case "$backend" in
 native_default) binary="$results/runtime/phase7_r6" ;;
 native_avx2_compatible) binary="$candidate/layer0_candidate" ;;
 cnnl_reference)
  echo "backend=cnnl_reference scope=adapter_projection" >&2
  exec "$results/deepseek_adapter" "$@" cnnl_reference
  ;;
 *) echo "Unknown experimental backend: $backend" >&2; exit 2 ;;
esac
echo "backend=$backend scope=layer0_reference_mla" >&2
exec "$binary" "$@"
