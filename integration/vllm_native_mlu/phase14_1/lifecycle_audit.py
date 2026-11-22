"""Validate scheduler, restart and fault events from real engine traces."""
import json
from pathlib import Path
import sys

from inspect_run import records


def chronological(path):
    for line in path.read_text().splitlines():
        for marker in ('{"backend": "vendor"', '{"phase14_1"'):
            index=line.find(marker)
            if index>=0:
                yield json.JSONDecoder().raw_decode(line[index:])[0]
                break


def ids(value):
    if isinstance(value,int):
        return {value}
    if isinstance(value,list):
        return set().union(*(ids(v) for v in value))
    return set()


def main(root):
    expansive=records(root/"tp2_expanded/worker.log")
    tokens=[r for r in expansive if r["phase14_1"]=="generate"]
    beams=[r for r in expansive if r["phase14_1"]=="logits_parity"]
    assert len(tokens)==12 and all(r["tokens"]==r["expected"] for r in tokens)
    assert len(beams)==96 and all(r["tolerance_mismatch"]==0 for r in beams)

    path=root/"cross-page/worker.log"
    forwards={}
    released,active,reuse=set(),{},0
    for r in chronological(path):
        if r.get("phase14_1")=="forward":
            forwards.setdefault(r["step"],{})[r["rank"]]=r["schedule"]
        if r.get("event")=="cache_free":
            released.update(active.pop(r["request"],set()))
        if r.get("event")=="schedule":
            for new in r["new"]:
                blocks=ids(new["blocks"])
                reuse+=bool(released & blocks)
                active[new["request"]]=blocks
    assert all(set(pair)=={0,1} and pair[0]==pair[1] for pair in forwards.values())
    boundary=forwards[16][0][0]
    assert boundary[3]==17 and boundary[6]==[16] and len(boundary[4])==2
    assert boundary[5]==[16*boundary[4][1]]
    mixed=[i for i,pair in forwards.items() if len(pair[0])==2
           and sorted(s[2]-s[1] for s in pair[0])[0]==1
           and sorted(s[2]-s[1] for s in pair[0])[1]>1]
    assert mixed and reuse
    print(json.dumps(dict(expanded_requests=len(tokens),logit_compares=len(beams),
                          page_cross_position=16,page_cross_slot=boundary[5][0],
                          mixed_scheduler_steps=mixed,reuse_events=reuse)))
    suite = root / "fault-suite" / "worker.log"
    if suite.exists():
        fault_records = records(suite)
        assert all(r["phase14_1"] != "forward" for r in fault_records)
        for kind in ("invalid-block", "invalid-slot", "divergent", "missing-metadata"):
            rejected = [r for r in fault_records
                        if r["phase14_1"] == "negative_rejected" and r["fault"] == kind]
            assert len(rejected) == 1 and rejected[0]["aborted_requests"] == 1
            print(kind, "COHERENT REJECTION AND ABORT PASS")
        print("Fault-engine shutdown", "PASS" if any(
            r["phase14_1"] == "engine_shutdown" for r in fault_records) else "BLOCKED")
    for phase in ("tp2_expanded", "cross-page"):
        assert "all workers exited gracefully" in (root / phase / "worker.log").read_text(), phase
    print("Observed graceful normal worker exits passed")


if __name__=="__main__":
    main(Path(sys.argv[1]))
