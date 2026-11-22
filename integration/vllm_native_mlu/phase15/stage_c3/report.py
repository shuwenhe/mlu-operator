"""Build Stage C.3 evidence without changing earlier stage artifacts."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median

from runpy import run_path

C2 = run_path(str(Path(__file__).resolve().parents[1] / "stage_c2/report.py"))
read_csv, write_csv = C2["read_csv"], C2["write_csv"]
RANKS = set(range(8))


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def probe(root, directory):
    groups = defaultdict(list)
    workload = []
    for rank in RANKS:
        rows = json_rows(root / directory / f"rank{rank}.jsonl")
        assert rows[-1]["lifecycle"] == "cleanup"
        assert rows[0]["gate"] == "rank_device" and rows[0]["device"] == f"mlu:{rank}"
        assert rows[0]["backend"] == "cncl"
        gates = [row for row in rows if row.get("gate") == "numerical"]
        assert len(gates) == 6
        workload.extend(gates)
        for row in rows:
            if "variant" in row:
                key = tuple(row[k] for k in ("variant", "rows", "layer", "sample"))
                groups[key].append(row["milliseconds"])
    assert groups and all(len(values) == 8 for values in groups.values())
    merged = defaultdict(list)
    for (variant, rows, _, _), values in groups.items():
        merged[variant, rows].append(max(values))
    rows = [dict(variant=variant, rows=count, samples=len(values),
                 median_ms=median(values), mean_ms=mean(values))
            for (variant, count), values in sorted(merged.items())]
    assert all(row["samples"] == 30 for row in rows)
    write_csv(root / f"{directory}_alternatives.csv", rows)
    write_csv(root / f"{directory}_workload.csv", workload)
    for row in rows:
        print(row)


def profiles(root, frozen):
    selected = root / "candidate_profile"
    C2["compare_logits"](frozen / "candidate_profile", selected, root / "logit_parity.csv")
    workload = []
    collective_rows = []
    gemm_rows = []
    for variant, directory in (("stage_c2", frozen / "candidate_profile"), ("stage_c3", selected)):
        for rank in RANKS:
            events = json_rows(directory / f"profile_rank{rank}.jsonl")
            assert {row["step"] for row in events} == set(range(64))
            if variant == "stage_c3":
                routed = json_rows(directory / f"moe_workload_rank{rank}.jsonl")
                assert len(routed) == 64 * 26
                workload.extend(routed)
            for phase, predicate in (("cold_prefill", lambda i: i == 0),
                                     ("cold_decode", lambda i: 0 < i < 32),
                                     ("warm_prefill", lambda i: i == 32),
                                     ("warm_decode", lambda i: 32 < i < 64)):
                rows = [row for row in events if predicate(row["step"])]
                names = {name for row in rows for name in row["milliseconds"]}
                for name in sorted(names):
                    if "cncl" in name and name.endswith("_moe"):
                        collective_rows.append(dict(variant=variant, phase=phase, rank=rank, collective=name,
                                                    mean_ms=mean(row["milliseconds"].get(name, 0) for row in rows),
                                                    mean_calls=mean(row["calls"].get(name, 0) for row in rows),
                                                    mean_operand_bytes=mean(row["payload_bytes"].get(name, 0) for row in rows)))
                    if "expert" in name and "gemm" in name:
                        gemm_rows.append(dict(variant=variant, phase=phase, rank=rank, projection=name,
                                              mean_ms=mean(row["milliseconds"].get(name, 0) for row in rows),
                                              mean_calls=mean(row["calls"].get(name, 0) for row in rows)))
    write_csv(root / "moe_communication_profile.csv", collective_rows)
    write_csv(root / "expert_gemm_profile.csv", gemm_rows)
    if workload:
        write_csv(root / "rank_workload.csv", workload)
    return len(workload)


def benchmarks(root):
    data = []
    for variant in ("baseline", "candidate"):
        for run in (1, 2, 3):
            folder = root / variant / str(run)
            benchmark = read_csv(folder / "performance.csv")
            assert [row["phase"] for row in benchmark] == ["cold", "warm", "warm"]
            assert all(row["generated_tokens"] == "32" for row in benchmark)
            lifecycle = read_csv(folder / "lifecycle.csv")
            assert len(lifecycle) == 8 and all(r["exit_code"] == "0" and r["alive"] == "False" for r in lifecycle)
            memory = read_csv(folder / "memory_per_card.csv")
            for row in benchmark:
                cards = {int(card["card"]): int(card["used_bytes"]) for card in memory
                         if card["phase"] == f"request_{row['request']}"}
                assert set(cards) == RANKS
                row.update(variant=variant, run=run, **{f"mlu_{rank}_bytes": cards[rank] for rank in sorted(RANKS)})
                data.append(row)
    write_csv(root / "benchmark.csv", data)
    metrics = ("ttft_ms", "tpot_mean_ms", "tpot_p95_ms", "decode_tokens_s", "end_to_end_tokens_s", "engine_init_ms")
    results = []
    for variant in ("baseline", "candidate"):
        for phase in ("cold", "warm"):
            rows = [row for row in data if row["variant"] == variant and row["phase"] == phase]
            for metric in metrics:
                values = [float(row[metric]) for row in rows]
                results.append(dict(variant=variant, phase=phase, metric=metric, samples=len(values),
                                    median=median(values), minimum=min(values), maximum=max(values)))
    write_csv(root / "benchmark_summary.csv", results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("mode", choices=("probe", "overlap_probe", "profiles", "benchmarks"))
    args = parser.parse_args()
    if args.mode == "profiles":
        profiles(args.root, args.root.parent / "stage_c2")
    elif args.mode == "benchmarks":
        benchmarks(args.root)
    else:
        probe(args.root, args.mode)


if __name__ == "__main__":
    main()
