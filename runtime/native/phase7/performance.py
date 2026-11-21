"""Extract measured trace-run totals and identify unmeasured performance scopes."""
import csv
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text()
out = csv.writer(sys.stdout)
out.writerow(("dtype", "prefill", "metric", "value", "unit", "status", "scope"))

cases = re.split(r"(?m)^case_dtype=(bf16|fp16) prefill=(1|7|32|128) decode=1$", log)
for i in range(1, len(cases), 3):
    dtype, length, body = cases[i:i + 3]
    case = rf"phase7_case {dtype} {length} exit=1"
    if case not in body:
        continue
    timing = re.search(r"(?m)^real\s+(\d+)m([0-9.]+)s", body)
    usage = re.search(r"packed_weight_bytes=(\d+) cache_bytes=(\d+) peak_managed_bytes=(\d+) remaining=(\d+)", body)
    if not timing or not usage:
        raise SystemExit(f"missing hardware telemetry for {dtype}/{length}")
    seconds = int(timing[1]) * 60 + float(timing[2])
    out.writerow((dtype, length, "trace_process_wall", f"{seconds:.3f}", "s", "MEASURED",
                  "includes host oracles, traces, transfers and device execution"))
    for name, value in zip(("persistent_packed_weights", "kv_cache", "peak_managed", "remaining_managed"),
                           usage.groups()):
        out.writerow((dtype, length, name, value, "bytes", "MEASURED",
                      "runtime-tracked; peak managed includes cache and excludes registry"))
    for name in ("weight_loading", "weight_packing", "initialization", "prefill_latency",
                 "decode_latency", "kernel_latency", "transfer_inclusive_latency",
                 "peak_physical_memory", "workspace_only_peak"):
        out.writerow((dtype, length, name, "", "", "BLOCKED",
                      "no isolated measurement in trace run; Layer 0 numerical acceptance failed"))
