"""Merge Phase 8.1 reports without treating the historical oracle as PyTorch."""
import csv
import re
import sys
from pathlib import Path

stage, logs = map(Path, sys.argv[1:])
runs = [(d, m) for d, m, _ in map(str.split, (stage / "candidate/trace_manifest").read_text().splitlines())]
assert len(runs) == len(set(runs)) == 9


def combine(target, sources, extra=()):
    fields = None
    with target.open("w") as destination:
        writer = csv.writer(destination)
        for path, tags in sources:
            with path.open(newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader)
                if fields is None:
                    fields = header
                    writer.writerow([*extra, *fields])
                assert fields == header, path
                writer.writerows([*tags, *row] for row in reader)


replays = stage / "phase8_1"
combine(logs / "phase8_1_projection_accuracy.csv",
        [(logs / "phase7_projection_accuracy.csv", ("native_default", "phase7_device", ""))] +
        [(replays / f"{d}-{m}" / name, ("native_avx2_compatible", "phase8_device", ""))
         for d, m in runs for name in ("accuracy.csv", "widened.csv")],
        ("backend", "device_trace", "kernel_latency_us"))
combine(logs / "phase8_1_layer0_accuracy.csv",
        [(replays / f"{d}-{m}" / "layer.csv", ()) for d, m in runs])

body = (logs / "phase8_1_hardware.log").read_text()
cases = re.split(r"(?m)^case_dtype=(fp16|bf16) prefill=(\d+) decode=1$", body)
with (logs / "phase8_1_performance.csv").open("w") as stream:
    out = csv.writer(stream)
    out.writerow(("backend", "dtype", "prefill", "metric", "value", "unit", "status", "scope"))
    for dtype, length, case in zip(cases[1::3], cases[2::3], cases[3::3]):
        wall = re.search(r"(?m)^real\s+(\d+)m([\d.]+)s", case)
        assert wall and f"candidate_case {dtype} {length} exit=1" in case
        out.writerow(("native_avx2_compatible", dtype, length, "trace_process_wall",
                      int(wall[1]) * 60 + float(wall[2]), "s", "MEASURED",
                      "two slots, two repetitions, host reference, transfers and trace I/O"))
        usage = re.search(r"packed_weight_bytes=(\d+) cache_bytes=(\d+) peak_managed_bytes=(\d+) remaining=(\d+)", case)
        assert usage
        for name, value in zip(("packed_weights", "kv_cache", "peak_managed", "remaining"), usage.groups()):
            out.writerow(("native_avx2_compatible", dtype, length, name, value, "bytes", "MEASURED",
                          "runtime accounting; peak managed excludes packed registry"))
    for backend in ("native_default", "native_avx2_compatible", "cnnl_reference"):
        for metric in ("kernel_latency", "prefill_latency", "decode_latency", "throughput",
                       "weight_packing", "transfer_inclusive_latency", "peak_physical_memory"):
            out.writerow((backend, "", "", metric, "", "", "BLOCKED", "Phase E requires numerical acceptance"))
