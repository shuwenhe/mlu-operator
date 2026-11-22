"""Compare complete candidate traces and both greedy runs with the pinned reference."""

import re
from pathlib import Path
import sys

import numpy as np

from orders import compare

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase11"))
from compare_model import main as compare_model


def gold_tokens(log):
    return [(int(step), int(token)) for step, token in
            re.findall(r"^generated (\d+) (\d+) ", log.read_text(), re.M)]


def main(root, accepted, run_dir):
    import csv
    from contextlib import redirect_stdout

    candidate = root / run_dir
    failed = False
    for name, reference in (("one", accepted / "ref-one"), ("multi", root / "reference-full")):
        with (candidate / f"{name}-all.csv").open("w") as output, redirect_stdout(output):
            status = compare_model(candidate / name, reference)
        failed |= bool(status)
        print(f"candidate_{name}_trace_exit={status}")
    with (candidate / "logits-accuracy.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        for name, ref, steps in (("one", accepted / "ref-one", 1),
                                 ("multi", root / "reference-full", 32)):
            for step in range(steps):
                filename = f"{step}-logits.bin"
                actual, expected = [np.fromfile(path / filename, dtype="<f4")[None, :]
                                    for path in (candidate / name, ref)]
                if actual.shape != (1, 102400) or expected.shape != actual.shape:
                    raise ValueError(f"Invalid vocabulary shape for {name}/{filename}")
                compare(writer, f"{name}-step{step}", "candidate_vocabulary_logits", actual, expected)
    expected = gold_tokens(root / "reference-full.log")
    for run in (1, 2):
        actual = gold_tokens(candidate / f"generation{run}.log")
        passed = len(expected) == 32 and actual == expected
        failed |= not passed
        print(f"candidate_generation{run}_agrees={passed} tokens={len(actual)}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]))
