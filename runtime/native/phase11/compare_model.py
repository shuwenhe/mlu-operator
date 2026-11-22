"""Compare native and original checkpoint BF16 tensors at every decoder boundary."""
import csv
from pathlib import Path
import sys
import numpy as np


def main(actual, ref):
    writer = csv.writer(sys.stdout,lineterminator="\n")
    writer.writerow(("step","stage","elements","mismatches","bit_differences","nonfinite",
                     "max_abs","max_rel","first_mismatch"))
    if {p.name for p in actual.glob("*.bin")} != {p.name for p in ref.glob("*.bin")}:
        raise ValueError("Mismatched checkpoint trace file set")
    failures = 0
    for path in sorted(actual.glob("*.bin")):
        fp32 = path.name.endswith("-logits.bin")
        dtype = "<f4" if fp32 else "<u2"
        a,b = (np.fromfile(p,dtype=dtype) for p in (path, ref/path.name))
        if a.shape != b.shape:
            raise ValueError(f"Different trace shape for {path.name}")
        aw,bw = (a,b) if fp32 else ((a.astype(np.uint32)<<16).view(np.float32),
                                     (b.astype(np.uint32)<<16).view(np.float32))
        error = np.abs(aw.astype(np.float64)-bw)
        invalid = ~np.isfinite(aw)|~np.isfinite(bw)
        bad = invalid|(error>1e-5+1e-5*np.abs(bw))
        first = np.flatnonzero(bad)
        stage = path.stem.split("-",1)
        writer.writerow((*stage,len(a),int(bad.sum()),int((a!=b).sum()),int(invalid.sum()),
                         error.max(initial=0),(error/np.maximum(np.abs(bw),1e-30)).max(initial=0),
                         int(first[0]) if first.size else ""))
        failures += int(bad.sum())
    print(f"trace_tolerance_mismatches={failures}",file=sys.stderr)
    return int(failures!=0)


if __name__=="__main__":
    raise SystemExit(main(*map(Path,sys.argv[1:])))
