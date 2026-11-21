#!/usr/bin/env bash
set -euxo pipefail
dir=${1:?experiment directory}
pci=${2:?authorized PCI}
dtype=${3:?dtype}
n=${4:?prefill}
trace=$(mktemp -d "$dir/experiment-$dtype-$n.XXXXXX")
args=(--authorized-reference-mla "$pci" /results/runtime-full-fixtures)
if [[ $dtype != bf16 || $n != 2 ]]; then args+=("$dtype" "$n"); fi
rc=0
LAYER0_TRACE_DIR="$trace" "$dir/layer0_experiment" "${args[@]}" || rc=$?
echo "phase82_case $dtype $n exit=$rc"
test "$rc" = 1 || test "$rc" = 0
test -s "$trace/1_${n}_final.bin"
echo "$dtype $n $trace" >> "$dir/trace_manifest"
