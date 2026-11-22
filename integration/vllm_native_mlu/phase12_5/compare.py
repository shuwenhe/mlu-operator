"""Compare per-request native worker tensor dumps against independent traces."""
import csv
from pathlib import Path
import sys

import numpy as np


def compare(native, reference, writer):
    count = 0
    length = (native / "prompt.ids").stat().st_size // 4
    for expected in sorted(reference.glob("*.bin")):
        path = native / expected.name
        if not path.exists():
            writer.writerow((native.name, length, expected.name, "", "", "", "", "", "", "MISSING"))
            count += 1
            continue
        logits = path.name.endswith("-logits.bin")
        dtype = "<f4" if logits else "<u2"
        a, b = (np.fromfile(p, dtype=dtype) for p in (path, expected))
        if a.shape != b.shape:
            raise ValueError(f"{native.name}/{path.name} has {a.shape} vs {b.shape}")
        def widen(x):
            if logits:
                return x
            return (x.astype(np.uint32) << 16).view(np.float32)
        av, bv = widen(a), widen(b)
        err = np.abs(av.astype(np.float64) - bv)
        bad = ~(np.isfinite(av) & np.isfinite(bv)) | (err > 1e-5 + 1e-5 * np.abs(bv))
        changes = np.flatnonzero(bad)
        mismatch = int(bad.sum())
        count += mismatch
        writer.writerow((native.name, length, path.name, a.size, int((a != b).sum()), mismatch,
                         err.max(initial=0), (err / np.maximum(np.abs(bv), 1e-30)).max(initial=0),
                         int(changes[0]) if changes.size else "", "FAIL" if mismatch else "PASS"))
    return count


def main(base, trace, output, prefix):
    failures = 0
    with output.open("w", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(("request", "prompt_length", "tensor", "elements", "bit_mismatches",
                         "tolerance_mismatches", "max_abs", "max_rel", "first_index", "status"))
        for native in sorted(trace.glob("request*")):
            prompt = native / "prompt.ids"
            if not prompt.is_file():
                continue
            length = prompt.stat().st_size // 4
            if prompt.read_bytes() != (base / "prompts" / f"{length}.bin").read_bytes():
                raise ValueError(f"{native.name} prompt bits differ")
            reference = base / f"{prefix}{length}"
            if not (reference / "0-logits.bin").exists():
                raise ValueError(f"Reference trace missing for length {length}")
            failures += compare(native, reference, writer)
    print(f"{prefix} tolerance failures={failures}")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main(*(Path(s) for s in sys.argv[1:4]), sys.argv[4]))
