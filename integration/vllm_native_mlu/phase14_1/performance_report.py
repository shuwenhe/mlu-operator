"""Export slowest-rank performance for actual engine requests."""
import csv
import math
from pathlib import Path
import statistics
import sys

from inspect_run import records


def main(root):
    rows=[]
    summaries=[]
    passed=True
    for tp in (1,2):
        found=[r for r in records(root/f"performance_tp{tp}.log") if r["phase14_1"]=="benchmark"]
        if len(found)!=100 or {r["repeat"] for r in found}!=set(range(100)):
            passed=False
        for r in found:
            flat=dict(r)
            steps=flat.pop("ranks")
            if steps:
                flat["peak_allocator_bytes"]=max(x["peak_allocator_bytes"] for x in steps)
                flat["peak_native_pack_bytes"]=max(x["native_pack_bytes"] for x in steps)
                for i,stage in enumerate(("prefill","decode")):
                    names=set().union(*(x["steps"][i]["stages"] for x in steps))
                    for metric in names:
                        flat[f"{stage}_{metric.removesuffix('_ms')}_ms"]=max(x["steps"][i]["stages"].get(metric,0) for x in steps)
                    flat[f"{stage}_slowest_forward_ms"]=max(x["steps"][i]["ms"] for x in steps)
            rows.append(flat)
        tp_rows = [r for r in rows if r["tp"] == tp]
        metrics = {"ttft_ms", "prefill_ms", "tpot_ms", "wall_ms", "tokens_per_second"}
        metrics.update(key for r in tp_rows for key in r
                       if key.endswith(("_ms", "_bytes")))
        for key in sorted(metrics):
            values=sorted(r[key] for r in tp_rows if key in r)
            if values:
                summaries.append(dict(tp=tp,metric=key,p50=statistics.median(values),
                                      p95=values[math.ceil(.95*len(values))-1],repetitions=len(values)))
    for name,items in (("performance.csv",rows),("performance_summary.csv",summaries)):
        with (root/name).open("w",newline="") as f:
            fields=list(dict.fromkeys(k for r in items for k in r)) or ["tp","metric","status"]
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows(items)
    print(f"{len(rows)} timed requests")
    return not passed


if __name__=="__main__":
    raise SystemExit(main(Path(sys.argv[1])))
