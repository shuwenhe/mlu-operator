#!/usr/bin/env bash
set -euxo pipefail
out=${1:?result directory}
checkpoint=${2:?checkpoint path}
phase10=${3:?Phase 10 artifact directory}
phase11=${4:?Phase 11 artifact directory}
packed=${5:?accepted packed weights}
captures=${6:?router captures}
py=${7:?pinned Python}
pci=${8:?MLU PCI}
bash runtime/native/phase11_5/build_candidate.sh "$out/bin" "$phase10/profile-bin" "$packed" "$phase11/model-bin"
mkdir -p "$out/one" "$out/multi" "$out/isolated"
b="$out/bin/model_run"
rope="$phase10/fixtures/bf16"
"$b" "$pci" "$checkpoint" "$phase11/model" "$rope" "$phase11/model/one.bin" 1 "$out/one" > "$out/one.log"
"$b" "$pci" "$checkpoint" "$phase11/model" "$rope" "$phase11/model/prompt.bin" 32 "$out/multi" > "$out/generation1.log"
"$b" "$pci" "$checkpoint" "$phase11/model" "$rope" "$phase11/model/prompt.bin" 32 > "$out/generation2.log"
failures=0
for layer in $(seq 1 26); do
    rc=0
    "$out/bin/moe_test" "$pci" "$checkpoint" "$phase11/moe-matrix/layer-$layer" "$layer" \
        > "$out/isolated/moe-$layer.log" || rc=$?
    echo "isolated layer $layer exit=$rc"
    failures=$((failures + (rc != 0)))
done
for case in "2 17" "26 1"; do
    read -r layer rows <<< "$case"
    rc=0
    "$out/bin/moe_test" "$pci" "$checkpoint" "$captures/diag-$layer" "$layer" "$rows" \
        > "$out/isolated/diag-$layer.log" || rc=$?
    echo "checkpoint diagnostic layer=$layer rows=$rows exit=$rc"
    failures=$((failures + (rc != 0)))
done
"$py" runtime/native/phase11_5/compare_candidate.py "$(dirname "$out")" "$phase11" "$(basename "$out")"
exit "$((failures > 0))"
