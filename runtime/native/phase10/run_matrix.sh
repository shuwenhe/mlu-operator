#!/usr/bin/env bash
set -euo pipefail
binary=${1:?MLA test executable}
fixtures=${2:?checkpoint fixtures}
out=${3:?output directory}
pci=${4:?authorized PCI}
mkdir -p "$out"
status=0
run_case() {
  local dtype=$1 length=$2 requests=$3 queues=$4
  local prefix="$out/$dtype-${length/:/_}-$requests-$queues"
  local rc=0
  "$binary" "$pci" "$fixtures" "$dtype" "$length" "$requests" "$queues" "$prefix" > "$prefix.log" 2>&1 || rc=$?
  printf '%s %s requests=%s queues=%s exit=%s\n' "$dtype" "$length" "$requests" "$queues" "$rc"
  if ((rc)); then status=$rc; fi
}
for dtype in bf16 fp16; do
  for length in 1 7 32 128; do
    run_case "$dtype" "$length" 1 1
    run_case "$dtype" "$length" 2 1
    run_case "$dtype" "$length" 2 2
  done
  run_case "$dtype" 7:128 2 1
  run_case "$dtype" 7:128 2 2
done
exit "$status"
