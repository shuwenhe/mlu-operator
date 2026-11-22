"""Export ordered compatibility records and slowest-rank summaries."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{"kind": ')]


def write(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def p95(items):
    return sorted(items)[math.ceil(0.95 * len(items)) - 1]


def main(root):
    accuracy, performance, summaries, issues = [], [], [], []
    for tp in (1, 2, 4, 8):
        path = root / f"tp{tp}.log"
        verified = root / f"verified_tp{tp}.log"
        if verified.exists():
            path = verified
        if not path.exists():
            issues.append((tp, "NOT RUN"))
            continue
        rows = records(path)
        checks = [r for r in rows if r["kind"] == "accuracy"]
        samples = [r for r in rows if r["kind"] == "performance"]
        accuracy.extend(checks)
        performance.extend(samples)
        ids = set(range(tp))
        if len(checks) != 64*tp or len(samples) != 100*tp:
            issues.append((tp, "record coverage"))
            continue
        for kind in ("result", "cleanup", "rank", "midpoint"):
            if {r["rank"] for r in rows if r["kind"] == kind} != ids:
                issues.append((tp, kind))
        if any(r["mismatches"] for r in rows if r["kind"] == "result"):
            issues.append((tp, "hardware result"))
        for check in checks:
            if check["tolerance_mismatch"] or check["nan"] or check["inf"]:
                issues.append((tp, check["boundary"]))
            if check["boundary"] != "legacy_fp32_cncl_sum_diagnostic" and check["bit_mismatch"]:
                issues.append((tp, "bitwise " + check["boundary"]))
        for repeat in range(100):
            if {r["rank"] for r in samples if r["repeat"] == repeat} != ids:
                issues.append((tp, "repetition coverage"))
        metrics = ["local_gemm_fold_ms", "communication_ms", "cast_ms", "o_ms", "chain_ms"]
        if all("fold_only_ms" in r for r in samples):
            metrics.append("fold_only_ms")
        for metric in metrics:
            peaks = [max(r[metric] for r in samples if r["repeat"] == i) for i in range(100)]
            summaries.append(dict(tp=tp, metric=metric, p50_ms=statistics.median(peaks),
                                  p95_ms=p95(peaks), scratch_bytes_per_rank=max(r["scratch_bytes"] for r in samples),
                                  allocator_peak_max=max(r["peak_allocator"] for r in samples)))
    chain_baseline = next((r["p50_ms"] for r in summaries if r["tp"] == 1 and r["metric"] == "chain_ms"), None)
    for row in summaries:
        is_chain = row["metric"] == "chain_ms"
        row["chain_tokens_per_second"] = 17000 / row["p50_ms"] if is_chain else ""
        row["chain_speedup"] = chain_baseline / row["p50_ms"] if is_chain and chain_baseline else ""
        row["chain_efficiency"] = row["chain_speedup"] / row["tp"] if row["chain_speedup"] else ""
    write(root / "accuracy.csv", accuracy)
    write(root / "performance.csv", performance)
    write(root / "performance_summary.csv", summaries)
    for issue in sorted(set(issues)):
        print("Gate", *issue)
    print(f"{len(accuracy)} accuracy rows and {len(performance)} timing rows")
    return bool(issues)


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
