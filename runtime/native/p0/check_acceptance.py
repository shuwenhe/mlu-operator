"""Fail the unchanged Layer 0 tolerance gate using the checkpoint comparison CSV."""
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = [row for row in csv.DictReader(stream) if row["comparison"] == "original_source"]

required_runs = {("bf16", 2)}
required_runs.update((dtype, width) for dtype in ("fp16", "bf16") for width in (1, 7, 32, 128))
expected_samples = {
    (dtype, width, request, position)
    for dtype, width in required_runs
    for request in (0, 1)
    for position in (0, width)
}
actual_samples = {
    (row["run_dtype"], int(row["prefill"]), int(row["request"]), int(row["position"]))
    for row in rows if row["stage"] == "final"
}
if actual_samples != expected_samples:
    raise SystemExit(f"FAIL: incomplete or unexpected run coverage: {expected_samples ^ actual_samples}")
failed_rows = [row for row in rows if int(row["mismatches"]) or int(row["nonfinite"])]
print(f"{'FAIL' if failed_rows else 'PASS'}: {len(failed_rows)}/{len(rows)} checkpoint stage comparisons failed")
sys.exit(bool(failed_rows))
