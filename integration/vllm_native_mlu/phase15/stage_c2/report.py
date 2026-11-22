"""Summarize repeated measurements and validate all TP8 logit pairs."""
import argparse
from array import array
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

RANKS = range(8)
TOKENS = 32
LOGIT_BYTES = 102400 * 4
TIMINGS = ("ttft_ms", "tpot_mean_ms", "tpot_p95_ms", "decode_tokens_s", "end_to_end_tokens_s",
           "engine_init_ms", "submission_ms")


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark(root, variants):
    measurements = []
    summaries = []
    for variant in variants:
        for run in (1, 2, 3):
            directory = root / variant / str(run)
            rows = read_csv(directory / "performance.csv")
            workers = read_csv(directory / "lifecycle.csv")
            if len(rows) != 3 or any(row["generated_tokens"] != "32" for row in rows):
                raise ValueError(f"Incomplete token run {directory}")
            if len(workers) != 8 or any(row["exit_code"] != "0" or row["alive"] != "False" for row in workers):
                raise ValueError(f"Unclean shutdown in {directory}")
            memory = read_csv(directory / "memory_per_card.csv")
            for row in rows:
                phase = f"request_{row['request']}"
                per_card = {int(card["card"]): int(card["used_bytes"]) for card in memory if card["phase"] == phase}
                if set(per_card) != set(RANKS):
                    raise ValueError(f"Missing device memory in {directory}")
                row.update(run=run, **{f"mlu_{rank}_bytes": per_card[rank] for rank in RANKS})
            measurements.extend(rows)
        for phase in ("cold", "warm"):
            rows = [row for row in measurements if row["variant"] == variant and row["phase"] == phase]
            for metric in TIMINGS:
                values = [float(row[metric]) for row in rows]
                summaries.append(dict(variant=variant, phase=phase, metric=metric, samples=len(values),
                                      median=median(values), minimum=min(values), maximum=max(values)))
    write_csv(root / "benchmark_requests.csv", measurements)
    write_csv(root / "benchmark_summary.csv", summaries)
    memory_summary = []
    for variant in variants:
        for phase in ("cold", "warm"):
            rows = [r for r in measurements if r["variant"] == variant and r["phase"] == phase]
            for rank in RANKS:
                values = [r[f"mlu_{rank}_bytes"] for r in rows]
                memory_summary.append(dict(variant=variant, phase=phase, rank=rank,
                                           median_bytes=median(values), min_bytes=min(values), max_bytes=max(values)))
    write_csv(root / "memory_summary.csv", memory_summary)


def exclusive(row):
    time = row["milliseconds"]

    def component(*names):
        return sum(time.get(name, 0) for name in names)

    cncl = {key: ms for key, ms in time.items() if key.startswith("cncl_")}
    moe_comm = sum(ms for key, ms in cncl.items() if key.endswith("_moe"))
    o_comm = sum(ms for key, ms in cncl.items() if key.endswith("_o"))
    routing = component("moe_router_projection", "moe_router_softmax", "phase11_route")
    experts = component("routed_expert_gemm", "shared_expert_gemm")
    result = {
        "cncl_wait_inclusive": sum(cncl.values()),
        "mla_output_non_cncl": component("o_total") - o_comm,
        "mla_projection_gemm": component("mla_projection_gemm"),
        "mla_attention_rope": component("qk", "softmax", "av", "rope"),
        "kv_cache": component("cache_read", "cache_write"),
        "moe_routing_kernels": routing,
        "moe_expert_gemm_including_pack": experts,
        "moe_other": component("moe_total") - routing - experts - moe_comm,
        "dense_gemm": component("dense_gemm"),
        "logits_gemm": component("logits_gemm"),
    }
    total = component("worker_forward_total", "worker_logits_total")
    result["worker_other"] = total - sum(result.values())
    return result


def profile(directory):
    groups = {}
    for rank in RANKS:
        path = directory / f"profile_rank{rank}.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            groups.setdefault(row["step"], {})[rank] = row
    if set(groups) != set(range(64)) or any(set(rows) != set(RANKS) for rows in groups.values()):
        raise ValueError(f"Missing profile steps or ranks in {directory}")
    components, collective_rows, details = [], [], []
    schedule = {int(row["step"]): float(row["milliseconds"])
                for row in read_csv(directory / "host_schedule.csv")}
    if set(schedule) != set(groups):
        raise ValueError(f"Host and worker steps differ in {directory}")
    for step, ranks in sorted(groups.items()):
        def total(row):
            return row["milliseconds"]["worker_forward_total"] + row["milliseconds"]["worker_logits_total"]

        critical = max(ranks.values(), key=total)
        phase = ("cold" if step < TOKENS else "warm") + ("_prefill" if step % TOKENS == 0 else "_decode")
        components.extend(dict(step=step, phase=phase, rank=critical["rank"], component=name, ms=ms)
                          for name, ms in exclusive(critical).items())
        components.extend([
            dict(step=step, phase=phase, rank=critical["rank"], component="worker_wall", ms=total(critical)),
            dict(step=step, phase=phase, rank=critical["rank"], component="weight_pack_subset",
                 ms=critical["milliseconds"].get("weight_pack", 0)),
            dict(step=step, phase=phase, rank="host", component="host_scheduler", ms=schedule[step]),
        ])
        for rank, row in ranks.items():
            for name, payload in row["payload_bytes"].items():
                collective_rows.append(dict(step=step, phase=phase, rank=rank, collective=name,
                                            calls=row["calls"][name], operand_bytes=payload,
                                            ms=row["milliseconds"][name]))
            for name, ms in row["milliseconds"].items():
                details.append(dict(step=step, phase=phase, rank=rank, category=name, ms=ms))
    write_csv(directory / "components.csv", components)
    write_csv(directory / "operator_details.csv", details)
    write_csv(directory / "collectives.csv", collective_rows)
    by_component = defaultdict(list)
    for row in components:
        by_component[row["phase"], row["component"]].append(row["ms"])
    summary = [dict(phase=phase, component=component, mean_ms=mean(values),
                    min_ms=min(values), max_ms=max(values))
               for (phase, component), values in sorted(by_component.items())]
    write_csv(directory / "component_summary.csv", summary)
    by_collective = defaultdict(list)
    for row in collective_rows:
        by_collective[row["phase"], row["rank"], row["collective"]].append(row)
    cncl_summary = [
        dict(phase=phase, rank=rank, collective=kind, steps=len(rows),
             mean_ms=mean(row["ms"] for row in rows),
             mean_calls=mean(row["calls"] for row in rows),
             mean_operand_bytes=mean(row["operand_bytes"] for row in rows))
        for (phase, rank, kind), rows in sorted(by_collective.items())
    ]
    write_csv(directory / "collective_summary.csv", cncl_summary)
    for row in summary:
        print(f"{row['phase']:13} {row['component']:32} {row['mean_ms']:9.3f} ms")


def compare_logits(reference, candidate, report):
    results = []
    for rank in RANKS:
        with (reference / f"logits_rank{rank}.bin").open("rb") as ref, (
                candidate / f"logits_rank{rank}.bin").open("rb") as actual:
            for step in range(64):
                r, a = ref.read(LOGIT_BYTES), actual.read(LOGIT_BYTES)
                if len(r) != LOGIT_BYTES or len(a) != LOGIT_BYTES:
                    raise ValueError(f"Missing logits: rank {rank} step {step}")
                worst, mismatches = 0.0, 0
                if r != a:
                    expected, observed = array("f"), array("f")
                    expected.frombytes(r)
                    observed.frombytes(a)
                    for want, got in zip(expected, observed):
                        error = abs(got - want)
                        worst = max(worst, error)
                        mismatches += (not math.isfinite(want) or not math.isfinite(got)
                                       or error > 1e-5 + 1e-5 * abs(want))
                results.append(dict(rank=rank, step=step, bitwise_equal=r == a,
                                    max_abs=worst, tolerance_mismatches=mismatches))
            if ref.read(1) or actual.read(1):
                raise ValueError("Unexpected trailing logits")
    write_csv(report, results)
    if any(row["tolerance_mismatches"] for row in results):
        raise ValueError("Stage C.2 logit tolerance exceeded")
    print(f"Validated {len(results)} rank/step pairs of 102400 logits")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--compare-logits", action="store_true")
    args = parser.parse_args()
    if args.compare_logits:
        compare_logits(args.root / "baseline_profile", args.root / "candidate_profile", args.root / "logit_parity.csv")
    if args.profile:
        for directory in args.profile:
            profile(args.root / directory)
    else:
        benchmark(args.root, ("baseline",) if args.baseline_only else ("baseline", "candidate"))


if __name__ == "__main__":
    main()
