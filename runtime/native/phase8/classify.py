"""Classify preserved Phase 7 gates and prepare exact operands for device experiments."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from projections import PROJECTIONS, ordered

STAGES = {"output_projection": "o", "output_projection_fp32": "o"}


def classify(logs, output):
    fields = ("suite", "dtype", "prefill", "request", "phase", "stage", "projection",
              "reference", "m", "n", "k", "row", "column", "failed_elements", "nonfinite")
    summary = Counter()
    with (output / "failure_classification.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for suite in ("layer0", "projection"):
            with (logs / f"phase7_{suite}_accuracy.csv").open() as source:
                for row in csv.DictReader(source):
                    count, nonfinite = int(row["mismatches"]), int(row["nonfinite"])
                    if count == 0 and nonfinite == 0:
                        continue
                    stage = row["stage"]
                    reference = row.get("reference", row.get("comparison"))
                    projection = STAGES.get(stage, stage)
                    width = PROJECTIONS[projection][1] if projection in PROJECTIONS else ""
                    if not width:
                        projection = ""
                    m, n = ((int(row["m"]), int(row["n"])) if suite == "projection"
                            else tuple(map(int, row["shape"].split("x")[-2:])))
                    first = divmod(int(row["first_mismatch"]), n) if row["first_mismatch"] else ("", "")
                    writer.writerow((suite, row.get("dtype", row.get("run_dtype")), row["prefill"],
                                     row["request"], "prefill" if row["position"] == "0" else "decode",
                                     stage, projection, reference, m, n, row.get("k", width),
                                     *first, count, nonfinite))
                    if suite == "layer0" and reference == "original_source":
                        summary[stage] += 1
    with (logs / "phase7_performance.csv").open() as source:
        timing = Counter(row["status"] for row in csv.DictReader(source))
    print("original_source_failed_stage_rows", json.dumps(summary, sort_keys=True), sum(summary.values()))
    print("phase7_performance_status", dict(timing))


def low_words(values, dtype):
    f = values.astype("<f4", copy=False)
    if dtype == "fp16":
        return f.astype("<f2").view("<u2")
    u = f.view("<u4")
    return ((u + 0x7fff + ((u >> 16) & 1)) >> 16).astype("<u2")


def widen(words, dtype):
    if dtype == "fp16":
        return words.view("<f2").astype("<f4")
    return (words.astype("<u4") << 16).view("<f4")


def main():
    logs, root, replay, output = map(Path, sys.argv[1:5])
    all_matched = sys.argv[5:] == ["--all-matched"]
    output.mkdir()
    classify(logs, output)
    fixture = root / "runtime-full-fixtures"
    traces = {(dtype, int(m)): root / Path(path).name
              for dtype, m, path in (line.split() for line in
                                     (root / "phase7_trace_manifest").read_text().splitlines())}
    weights = {}
    inexact_products = 0
    with (output / "projection_accuracy.csv").open("w") as detail, \
         (output / "samples.csv").open("w") as samples, (output / "operands.bin").open("wb") as operands:
        rows, sample_rows = csv.writer(detail), csv.writer(samples)
        rows.writerow(("dtype", "prefill", "request", "phase", "projection", "m", "n", "k", "row",
                       "column", "reference", "native_fp32", "reference_value", "native_bits",
                       "reference_bits", "ulp", "lower_neighbor", "upper_neighbor", "midpoint",
                       "native_minus_midpoint", "outside_tolerance", "experiment_sample"))
        sample_rows.writerow(("sample", "dtype", "prefill", "request", "phase", "projection", "m",
                              "n", "k", "row", "column", "native_fp32", "native_bits", "torch_bits", "fp64"))
        sample_id = 0
        for (dtype, length), trace in sorted(traces.items()):
            reference = root / f"p0-torch-{dtype}-{length}"
            captured = replay / f"{dtype}-{length}"
            for stage, (_, k, n) in PROJECTIONS.items():
                if (dtype, stage) not in weights:
                    bf16 = np.fromfile(fixture / f"{stage}.bin", "<u2")
                    weights[dtype, stage] = widen(low_words(widen(bf16, "bf16"), dtype), dtype).reshape(n, k)
                for request in range(2):
                    for position in (0, length):
                        phase = "prefill" if position == 0 else "decode"
                        prefix = f"{request}_{position}_{stage}"
                        native = np.fromfile(trace / f"{prefix}_fp32.bin", "<f4")
                        actual = low_words(native, dtype)
                        inputs = np.fromfile(captured / f"{prefix}_input.u16", "<u2").reshape(-1, k)
                        m = inputs.shape[0]
                        fp64 = np.fromfile(captured / f"{prefix}_fp64.f64", "<f8")
                        matched = np.fromfile(captured / f"{prefix}_matched_torch.u16", "<u2")
                        alias = "output_projection" if stage == "o" else stage
                        original = low_words(np.fromfile(reference / f"{request}_{position}_{alias}.bin", "<f4"), dtype)
                        for label, expected in (("matched_pytorch", matched), ("original_pytorch", original)):
                            assert expected.shape == actual.shape == (m * n,)
                            if all_matched and label != "matched_pytorch":
                                continue
                            mismatch = np.flatnonzero(actual != expected)
                            if not mismatch.size:
                                continue
                            got, want = widen(actual[mismatch], dtype), widen(expected[mismatch], dtype)
                            outside = np.abs(got.astype(np.float64) - want) > 1e-5 + 1e-5 * np.abs(want)
                            selection = int(np.flatnonzero(outside)[0]) if outside.any() else 0
                            distance = np.abs(ordered(actual[mismatch]) - ordered(expected[mismatch]))
                            # Neighbor of PyTorch's result in the direction of the native rounded value.
                            target = ordered(expected[mismatch])
                            neighbor = target + np.sign(ordered(actual[mismatch]) - target)
                            neighbor_words = np.where(neighbor < 32768, 65536 - neighbor, neighbor - 32768).astype("<u2")
                            adjacent = widen(neighbor_words, dtype)
                            lower, upper = np.minimum(want, adjacent), np.maximum(want, adjacent)
                            mid = (lower.astype(np.float64) + upper) / 2
                            for i, j in enumerate(mismatch):
                                r, c = divmod(int(j), n)
                                experiment = ""
                                if label == "matched_pytorch" and (all_matched or i == selection):
                                    experiment = sample_id
                                    sample_rows.writerow((sample_id, dtype, length, request, phase, stage, m,
                                                          n, k, r, c, float(native[j]), int(actual[j]),
                                                          int(expected[j]), float(fp64[j])))
                                    np.array([k], dtype="<i4").tofile(operands)
                                    a = widen(inputs[r], dtype)
                                    b = weights[dtype, stage][c]
                                    inexact_products += int(np.count_nonzero(a * b != a.astype(np.float64) * b))
                                    a.tofile(operands)
                                    b.tofile(operands)
                                    sample_id += 1
                                rows.writerow((dtype, length, request, phase, stage, m, n, k, r, c, label,
                                               float(native[j]), float(want[i]), f"{actual[j]:04x}",
                                               f"{expected[j]:04x}", int(distance[i]), float(lower[i]),
                                               float(upper[i]), float(mid[i]), float(native[j] - mid[i]),
                                               int(outside[i]), experiment))
    print("experiment_samples", sample_id, "inexact_fp32_operand_products", inexact_products)


if __name__ == "__main__":
    main()
