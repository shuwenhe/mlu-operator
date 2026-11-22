"""Validate hardware evidence and produce Phase 13.2 acceptance CSVs."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean
import struct


CASES = {
    "small": (128, 512, 256),
    "medium": (512, 2048, 1024),
    "large": (2048, 4096, 4096),
    "projection": (256, 7168, 4096),
}
DTYPES = ("bfloat16", "float32")
TPS = (1, 2, 4, 8)
MODES = ("column", "row")


def read_log(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.startswith('{"kind": ')]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(samples, mode, case, dtype, tp):
    global_latency_key = "global_end_to_end_ms" if mode == "column" else "end_to_end_ms"
    stats = {
        "mode": mode, "case": case, "dtype": dtype, "tp": tp,
        "repetitions_per_rank": len(samples) // tp,
        "worst_rank_mean_gemm_ms": 0, "worst_rank_mean_communication_ms": 0,
        "max_load_ms": max(row["load_ms"] for row in samples),
        "peak_bytes_per_card_max": max(row["peak_bytes"] for row in samples),
    }
    by_iteration = {}
    for sample in samples:
        by_iteration.setdefault(sample["iteration"], []).append(sample)

    def mean_max(key):
        return mean(max(row[key] for row in group) for group in by_iteration.values())

    stats["worst_rank_mean_gemm_ms"] = mean_max("gemm_ms")
    stats["worst_rank_mean_communication_ms"] = mean_max("communication_ms")
    stats["end_to_end_local_ms"] = mean_max("end_to_end_ms")
    if all(global_latency_key in row for row in samples):
        stats["end_to_end_global_ms"] = mean_max(global_latency_key)
    else:
        stats["end_to_end_global_ms"] = stats["end_to_end_local_ms"] + stats["worst_rank_mean_communication_ms"]
    m, k, n = CASES[case]
    stats["throughput_tflops"] = 2 * m * n * k / stats["end_to_end_global_ms"] / 1e9
    stats["communication_overhead_fraction"] = (
        stats["worst_rank_mean_communication_ms"] / stats["end_to_end_global_ms"] if tp > 1 else 0
    )
    return stats


def valid_weight(row, mkn, mode, tp):
    _, k, n = mkn
    rank = row["rank"]
    if mode == "column":
        width = n // (2 * tp)
        return (row["original_shape"], row["local_shape"], row["axis"], row["spans"]) == (
            [n, k], [2 * width, k], 0, [[rank * width, width], [n // 2 + rank * width, width]],
        )
    width = k // tp
    return (row["original_shape"], row["local_shape"], row["axis"], row["spans"]) == (
        [n, k], [n, width], 1, [[rank * width, width]],
    )


def valid_precision_bias(row):
    if row["mode"] == "column":
        start, width = row["rank"], 1
    else:
        start, width = 0, 8
    values = [(i + 1) / 16 for i in range(start, start + width)]
    expected_hash = hashlib.sha256(struct.pack(f"<{width}f", *values)).hexdigest()
    return (row["original_shape"], row["local_shape"], row["spans"], row["sha256"]) == (
        [8], [width], [[start, width]], expected_hash,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path)
    args = parser.parse_args()
    path = args.logs
    accuracy, weights, performance, samples = [], [], [], []
    missing = []
    checkpoints = read_log(path / "phase13_2_checkpoint_tp8.log")
    checkpoint_rows = [row for row in checkpoints if row["kind"] == "weight"]
    if (
        len(checkpoint_rows) != 48
        or len({(row["rank"], row["parameter"]) for row in checkpoint_rows}) != 48
        or {row["rank"] for row in checkpoint_rows} != set(range(8))
        or {row["rank"] for row in checkpoints if row["kind"] == "cleanup"} != set(range(8))
    ):
        missing.append("checkpoint/TP8")
    verification = path / "phase13_2_checkpoint_device_verify.log"
    if not verification.exists() or "CHECKPOINT_MLU_SHA256 PASS 48/48" not in verification.read_text():
        missing.append("checkpoint/reconstruction")
    for record in checkpoints:
        if record["kind"] == "weight":
            weights.append(record)
    for mode, tp in itertools.product(MODES, TPS):
        records = read_log(path / f"phase13_2_{mode}_tp{tp}.log")
        if not records and tp == 8:
            records = read_log(path / f"phase13_2_{mode}_smoke.log")
        rank_records = [row for row in records if row["kind"] == "rank"]
        if len(rank_records) != tp or {row["rank"] for row in rank_records} != set(range(tp)):
            missing.append(f"{mode}/TP{tp}/rank binding")
        assert all(row["backend"] == "cncl" and row["device"] == f'mlu:{row["rank"]}'
                   and row["communicator"] == ("MLUCommunicator" if tp > 1 else "NoneType")
                   for row in rank_records)
        clean = {row["rank"] for row in records if row["kind"] == "cleanup"}
        if clean != set(range(tp)):
            missing.append(f"{mode}/TP{tp}/cleanup")
        for case, dtype in itertools.product(CASES, DTYPES):
            chosen = [row for row in records if row.get("case") == case and row.get("dtype") == dtype]
            checks = [row for row in chosen if row["kind"] == "accuracy"]
            local_weights = [row for row in chosen if row["kind"] == "weight"]
            perf = [row for row in chosen if row["kind"] == "performance"]
            check_names = ("local", "gather") if mode == "column" else ("local", "replicated_input", "sharded_input")
            ranking = set(range(tp))
            expected_checks = set(itertools.product(ranking, check_names))
            if (
                {(r["rank"], r["check"]) for r in checks} != expected_checks
                or len(checks) != len(expected_checks)
                or len(local_weights) != tp or {row["rank"] for row in local_weights} != ranking
                or not all(valid_weight(row, CASES[case], mode, tp) for row in local_weights)
                or {row["rank"] for row in chosen if row["kind"] == "negative"} != ranking
                or len(perf) < 100 * tp
            ):
                missing.append(f"{mode}/TP{tp}/{case}/{dtype}")
            assert all(row["tolerance_mismatches"] == 0 for row in checks)
            accuracy.extend(checks)
            weights.extend(local_weights)
            # Earlier smoke runs predate indexed timing samples.
            if perf and all("iteration" in row for row in perf):
                assert len(perf) == len({(row["rank"], row["iteration"]) for row in perf})
                samples.extend(perf)
                if len(perf) >= 100 * tp:
                    assert {(row["rank"], row["iteration"]) for row in perf} == set(itertools.product(ranking, range(len(perf) // tp)))
                    performance.append(aggregate(perf, mode, case, dtype, tp))

    baseline = {(row["mode"], row["case"], row["dtype"]): row["end_to_end_global_ms"]
                for row in performance if row["tp"] == 1}
    for row in performance:
        latency = baseline.get((row["mode"], row["case"], row["dtype"]))
        row["speedup"] = latency / row["end_to_end_global_ms"] if latency else None
        row["parallel_efficiency"] = row["speedup"] / row["tp"] if latency else None
    write_csv(path / "phase13_2_performance.csv", performance)
    for mode in MODES:
        write_csv(path / f"phase13_2_{mode}_parallel.csv", [row for row in samples if row["mode"] == mode])
    for tp in (1, 8):
        wide = read_log(path / f"phase13_2_wide_checkpoint_tp{tp}.log")
        checks = [row for row in wide if row["kind"] == "accuracy"]
        ranks = set(range(tp))
        if (
            {(row["rank"], row["check"]) for row in checks}
            != set(itertools.product(ranks, ("local", "gather")))
            or any(row["tolerance_mismatches"] for row in checks)
            or any({row["rank"] for row in wide if row["kind"] == kind} != ranks
                   for kind in ("wide_checkpoint", "cleanup"))
        ):
            missing.append(f"checkpoint/packed/TP{tp}")
        accuracy.extend(checks)
    write_csv(path / "phase13_2_accuracy.csv", accuracy)
    precision = read_log(path / "phase13_2_precision_tp8.log")
    if any({r["rank"] for r in precision if r["kind"] == kind} != set(range(8))
           for kind in ("rank", "precision", "cleanup")):
        missing.append("FP32/precision/TP8")
    biases = [row for row in precision if row["kind"] == "weight"]
    if len(biases) != 16 or {(row["rank"], row["mode"]) for row in biases} != set(itertools.product(range(8), MODES)):
        missing.append("bias/ownership/TP8")
    if not all(valid_precision_bias(row) for row in biases):
        missing.append("bias/content/TP8")
    weights.extend(biases)
    write_csv(path / "phase13_2_weight_shards.csv", weights)
    print(f"{len(accuracy)} numerical rows, {len(weights)} shard rows, {len(performance)} complete timing configurations")
    if missing:
        print("Pending evidence:", ", ".join(missing))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
