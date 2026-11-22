"""Extract real engine milestones and compare frozen logit bytes."""
import array
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def records(path):
    if not path.exists():
        return []
    output = []
    for line in path.read_text().splitlines():
        start = line.find('{"phase14_1"')
        if start >= 0:
            output.append(json.JSONDecoder().raw_decode(line[start:])[0])
    return output


def compare(file, expected, rank, step):
    a,b = array.array("f"),array.array("f")
    a.frombytes(file.read_bytes())
    b.frombytes(expected.read_bytes())
    if len(a)!=102400 or len(a)!=len(b):
        raise ValueError("Unexpected logit boundary")
    actual_bits, target_bits = array.array("i"),array.array("i")
    actual_bits.frombytes(file.read_bytes())
    target_bits.frombytes(expected.read_bytes())

    def order(i):
        return -(1<<31)-i if i<0 else i

    errors = [abs(x-y) for x,y in zip(a,b)]
    bad = [i for i,(x,y,e) in enumerate(zip(a,b,errors))
           if not math.isfinite(x) or not math.isfinite(y) or e>1e-5+1e-5*abs(y)]
    return dict(rank=rank,device=f"mlu:{rank}",step=step,boundary="logits",
                max_abs=max(errors), max_rel=max(e/max(abs(y),1e-30) for e,y in zip(errors,b)),
                mean_abs=sum(errors)/len(errors),
                max_ulp=max(abs(order(i)-order(j)) for i,j in zip(actual_bits,target_bits)),
                bitwise_mismatch=sum(i!=j for i,j in zip(actual_bits,target_bits)),
                tolerance_mismatch=len(bad),first_mismatch=bad[0] if bad else "",
                nan=sum(math.isnan(x) for x in a),inf=sum(math.isinf(x) for x in a),
                fixture_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),
                output_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                greedy=max(range(len(a)),key=a.__getitem__),reference_greedy=max(range(len(b)),key=b.__getitem__))


def main(root, frozen, raw):
    entries = records(raw)
    for name, kinds in {
        "tp2_model_load.log": {"registered","constructed","shard","loaded"},
        "tp2_kv_lifecycle.log": {"cache_bound","forward"},
        "process_group_cleanup.log": {"engine_shutdown"},
    }.items():
        with (root/name).open("w") as out:
            rows = [r for r in entries if r["phase14_1"] in kinds]
            if not rows:
                print("NOT RUN or no successful records; see raw TP=2 worker log",file=out)
            for row in rows:
                print(json.dumps(row),file=out)
    forwards = [r for r in entries if r["phase14_1"]=="forward"]
    for tag, select in (("prefill",lambda r:all(s[2]-s[1]==s[3] for s in r["schedule"])),
                        ("decode",lambda r:any(s[2]-s[1]<s[3] for s in r["schedule"]))):
        data = [r for r in forwards if select(r)]
        (root/f"tp2_{tag}.log").write_text("\n".join(map(json.dumps,data))+"\n" if data else "NOT RUN or no successful records\n")
    native = [r for r in entries if r["phase14_1"]=="operator"]
    (root/"tp2_native_invocations.json").write_text(json.dumps(native,indent=2)+"\n")
    rows=[]
    for rank in (0,1):
        for step in (0,1):
            file=raw.parent/"tp2_dump"/f"rank{rank}_logits{step}.bin"
            if file.exists():
                rows.append(compare(file,frozen/f"{step}-logits.bin",rank,step))
    fields = list(rows[0]) if rows else ["rank","device","step","boundary","status"]
    with (root/"tp2_accuracy.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Worker model construction records:",sum(r["phase14_1"]=="constructed" for r in entries))
    print("Model load records:",sum(r["phase14_1"]=="loaded" for r in entries))
    print("Native execution records:",len(native))
    print("Logit comparisons:",len(rows),"tolerance mismatches:",sum(r["tolerance_mismatch"] for r in rows))
    print("Generation:",[r for r in entries if r["phase14_1"]=="generate"])


if __name__=="__main__":
    main(*map(Path,sys.argv[1:4]))
