# Phase 13.3.3 — Ordered O projection

The isolated `ordered.mlu` kernel reuses the frozen BANG C BF16 product, lane updates, packing permutation and fold. Each rank keeps its Phase 13.2 row-sharded packed BF16 weight and handles one contiguous, 128-aligned K region. The first rank initializes state, intermediate ranks load and update it, and the final rank folds once. CNCL broadcasts the unfurled FP32 state after each preceding rank and broadcasts the FP32 result from the last rank. The BF16 conversion and residual addition remain after that result. The production Row path and TP=1 model guard remain frozen.

The probe reports each rank's existing Row FP32 partial and CNCL SUM as a control diagnostic, captures the compatible FP32 and BF16 values, and replays the frozen O boundary using the exact Phase 13.3.2 saved AV inputs. That control SUM is independent from the ordered output path. The latter uses multiple collectives and serial rank work because a single folded-output SUM cannot carry the required recurrence.

## Reproduction

From the repository root on the MLU host, use a fresh evidence target for any additional execution.

```bash
set -euo pipefail
d=$(mktemp -d "$PWD/artifacts/logs/phase13_3_3/replay.XXXXXX")
export NEURX_1333_EVIDENCE_DIR="$d"
image=sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756
src=integration/vllm_native_mlu/phase13_3_3
python3 "$src/test_rounding.py"
docker run --rm --pull=never --network=none --read-only \
  --tmpfs /tmp:rw,exec,mode=1777 --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/project,readonly" \
  --mount "type=bind,src=$d,dst=/results" -w /project \
  --entrypoint /bin/bash "$image" "$src/build.sh"
for tp in 1 2 4 8; do
  bash "$src/run_probe.sh" "$tp" test > "$d/tp$tp.log" 2>&1
done
python3 "$src/report.py" "$d"
```

The runner uses the saved `artifacts/build/phase13_3_2/boundary_*.pt` records and reads the original checkpoint and frozen Phase 12.5.4 reference files in place. The independent review compared the recurrence/fold text to the frozen source, verified K tile alignment and checked flow ordering. The final tests use integer views for bitwise stability and record mismatches before collective cleanup.

Set and export `NEURX_1333_EVIDENCE_DIR` to the new evidence directory when repeating these commands. Build into the same directory. The initial ordered run is kept alongside supplementary `verified_tp*.log` traces from the final bitwise and raw-tensor checks. The exporter prefers those supplemental traces when present.

For the optional strategy comparisons, use `--output-axis --no-capture` or `--deferred --no-capture` on the same runner. Store their stdout/stderr in separate subdirectories and export each directory independently with `report.py`. The numerical gate is enforced in TP order for each strategy. A deferred run that returns nonzero is evidence that independent lane states cannot reproduce the frozen path. Preserve its log and stop that sweep at the failing TP.

The frozen-path regression commands are `python3 integration/vllm_native_mlu/phase13_3_2/test_mapping.py` and `bash integration/vllm_native_mlu/phase13_3_2/run_probe.sh 1 test`. Running the latter with TP=2 is the native red control. The repository GEMM CPU regression used the compile flags from `scripts/test.sh` with a Phase 13.3.3 executable and CSV destination.
