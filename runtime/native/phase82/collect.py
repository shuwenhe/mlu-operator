"""Merge only completed unique Layer 0 fixtures and assert coverage."""
import collections
import csv
import sys
from pathlib import Path

root, logs = map(Path,sys.argv[1:])
paths = {(d,int(m)):root/Path(path).name for d,m,path in
         (line.split() for line in (root/"trace_manifest").read_text().splitlines())}
cases={(d,m) for d in ("fp16","bf16") for m in (1,7,32,128)}|{("bf16",2)}
assert set(paths)==cases
for source,label,count in (("layer","layer0_accuracy",64),("projections","projection_regression",28)):
    table=[]
    for key,path in sorted(paths.items()):
        with (path/f"{source}.csv").open() as f:
            r=csv.DictReader(f)
            rows=list(r)
            assert len(rows)==count,(key,path,len(rows))
            table.extend(rows)
    with (logs/f"phase8_2_{label}.csv").open("w") as f:
        w=csv.DictWriter(f,list(table[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    counts=collections.Counter()
    for row in table:
        counts["failed_rows"]+=int(int(row["mismatches"])>0)
        counts["nonfinite"]+=int(row["nonfinite"])
        counts["bit_differences"]+=int(row["bit_differences"])
    print(label,len(table),dict(counts))
    if source=="layer":
        for d,m in sorted(cases):
            rows=[r for r in table if r["run_dtype"]==d and int(r["prefill"])==m]
            failed=[r for r in rows if int(r["mismatches"])]
            print(d,m,sum(int(r["mismatches"]) for r in rows if r["stage"]=="final"),
                  [(r["stage"],r["request"],r["position"],r["first_mismatch"]) for r in failed[:4]])
