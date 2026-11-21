#!/usr/bin/env bash
set -euxo pipefail
dir=${1:?device trace parent in SDK}
for dtype in fp16 bf16; do
 for n in 1 2 7 32 128; do
  if [[ $dtype == fp16 && $n == 2 ]]; then continue; fi
  trace=
  for attempt in $(seq 1 240); do
   if [[ -f "$dir/trace_manifest" ]]; then
    trace=$(awk -v d="$dtype" -v n="$n" '$1==d&&$2==n {p=$3} END{print p}' "$dir/trace_manifest")
   fi
   if [[ -n $trace && -s "$trace/1_${n}_final.bin" ]]; then break; fi
   sleep 2
  done
  test -n "$trace"
  ref=/results/p0-torch-$dtype-$n
  if [[ $n == 2 ]]; then ref=/results/p0-torch; fi
  /torch/venv3/pytorch_infer/bin/python3 runtime/native/phase82/regression.py \
    "$trace" /results/runtime-full-fixtures "$ref" "$trace" "$dtype" "$n"
 done
done
