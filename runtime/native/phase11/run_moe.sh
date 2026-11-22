#!/usr/bin/env bash
set -euo pipefail
py=${1:?PyTorch interpreter}
checkpoint=${2:?checkpoint}
baseline=${3:?reference input directory}
binary=${4:?native MoE harness}
output=${5:?results directory}
pci=${6:?MLU PCI}
mkdir -p "$output"
status=0
for layer in $(seq 1 26); do
  path="$output/layer-$layer"
  rc=0
  {
    "$py" runtime/native/phase11/moe_fixture.py "$checkpoint" "$baseline" "$path" "$layer" &&
      "$binary" "$pci" "$checkpoint" "$path" "$layer"
  } > "$path.log" 2>&1 || rc=$?
  printf 'MoE layer %d exit=%d\n' "$layer" "$rc"
  if ((rc)); then status=$rc; fi
done
exit "$status"
