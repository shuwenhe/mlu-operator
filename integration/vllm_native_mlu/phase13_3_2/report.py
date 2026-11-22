"""Export verified rank records and slowest-rank performance summaries."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import statistics


def records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.startswith('{"kind": ')]


def write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def p95(values):
    return sorted(values)[math.ceil(.95 * len(values)) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    reference = records(directory / "reference.log")
    accuracy = [r for r in reference if r["kind"] == "accuracy"]
    perf, weights, summary, issues = [], [], [], []
    if len(accuracy) != 80 or any(row["tolerance_mismatch"] for row in accuracy):
        issues.append("reference accuracy")
    for tp in (1,2,4,8):
        rows = records(directory / f"tp{tp}.log")
        benchmark = records(directory / f"timing_tp{tp}.log") or rows
        checks = [r for r in rows if r["kind"] == "accuracy"]
        accuracy.extend(checks)
        loaded = [r for r in rows if r["kind"] == "weight"]
        weights.extend(loaded)
        if len(loaded) != 5*tp:
            issues.append(f"TP={tp} checkpoint identities")
        replicas = collections.defaultdict(set)
        for r in checks:
            if r["boundary"] in ("q_head_order", "kv_a_latent", "kv_a_shared_rotary", "kv_a_norm",
                                 "kv_b_head_order", "o_cncl_fp32", "o_cast_bf16", "residual_bf16"):
                replicas[r["boundary"],r["length"],r["step"]].add(r["result_sha256"])
        if len(replicas)!=8*16 or any(len(v)!=1 for v in replicas.values()):
            issues.append(f"TP={tp} output consistency")
        if any(r["nan"] or r["inf"] or r["tolerance_mismatch"] for r in checks):
            issues.append(f"TP={tp} nonfinite or tolerance failure")
        ranks = [r for r in rows if r["kind"] == "rank"]
        finished = {r["rank"] for r in rows if r["kind"] == "result"}
        clean = {r["rank"] for r in rows if r["kind"] == "cleanup"}
        if (
            len(checks) != 224*tp or len(ranks) != tp or finished != set(range(tp))
            or clean != set(range(tp))
            or any(r["backend"]!="cncl" or r["device"]!=f'mlu:{r["rank"]}'
                   or r["communicator"]!=("MLUCommunicator" if tp>1 else "NoneType") for r in ranks)
        ):
            issues.append(f"TP={tp} accuracy/binding/cleanup")
        loads = [r["host_ms"] for r in benchmark if r["kind"]=="load"]
        if len(loads)==tp:
            summary.append(dict(tp=tp, op="load", p50_host_ms=statistics.median(loads),
                                p95_host_ms=p95(loads), slowest_sample_ms=max(loads)))
        else:
            issues.append(f"TP={tp} weight loading")
        for kind in ("performance", "load"):
            perf.extend(r for r in benchmark if r["kind"]==kind)
        grouped = collections.defaultdict(list)
        stability = {(r["rank"], r["repeat"]) for r in benchmark if r["kind"]=="stability"}
        if stability != {(rank, repeat) for rank in range(tp) for repeat in range(100)}:
            issues.append(f"TP={tp} stability")
        for row in benchmark:
            if row["kind"]=="performance":
                grouped[row["op"]].append(row)
        for op in ("q","kv_a","kv_norm","kv_b","o_gemm","allreduce","chain"):
            samples = grouped[op]
            if len(samples) != 100*tp:
                issues.append(f"TP={tp} {op} timing")
                continue
            repeats = collections.defaultdict(list)
            for r in samples:
                repeats[r["repeat"]].append(r)
            host = [max(r["host_ms"] for r in a) for a in repeats.values()]
            device = [max(r["stream_ms"] for r in a) for a in repeats.values()]
            peak = {rank:max(r["allocator_peak"] for r in samples if r["rank"]==rank) for rank in range(tp)}
            native = {rank:max(r.get("native_bytes",0) for r in samples if r["rank"]==rank) for rank in range(tp)}
            record = dict(tp=tp, op=op, p50_host_ms=statistics.median(host),
                          p95_host_ms=p95(host), p50_stream_ms=statistics.median(device),
                          p95_stream_ms=p95(device), slowest_sample_ms=max(host),
                          per_rank_peak_bytes=json.dumps(peak), per_rank_native_bytes=json.dumps(native),
                          aggregate_tracked_bytes=sum(peak.values())+sum(native.values()))
            kernel = [max(r["kernel_ms"] for r in a) for a in repeats.values()
                      if all(r.get("kernel_ms") is not None for r in a)]
            if kernel:
                record.update(kernel_p50_ms=statistics.median(kernel), kernel_p95_ms=p95(kernel))
            elif op not in ("chain","allreduce"):
                issues.append(f"TP={tp} {op} direct kernel timing")
            if op=="chain":
                record["tokens_per_second"]=17000/record["p50_host_ms"]
            summary.append(record)
    baseline = {r["op"]:r["p50_host_ms"] for r in summary if r["tp"]==1}
    for row in summary:
        if row["op"] in baseline:
            row["speedup"]=baseline[row["op"]]/row["p50_host_ms"]
            row["parallel_efficiency"]=row["speedup"]/row["tp"]
    write(directory/"accuracy.csv",accuracy)
    write(directory/"performance.csv",perf)
    write(directory/"weight_shards.csv",weights)
    write(directory/"performance_summary.csv",summary)
    print(f"{len(accuracy)} accuracy rows, {len(weights)} device weight records, {len(perf)} timing rows")
    for issue in issues:
        print("Incomplete:",issue)
    raise SystemExit(bool(issues))


if __name__=="__main__":
    main()
