"""Sample the MKL path for small multi-token router projections."""

import ctypes
import os
from pathlib import Path
import sys

import numpy as np
import torch


def main(root):
    torch.set_num_threads(2)
    capture = root / "layer2-m17"

    def operand(filename):
        bits = np.fromfile(capture / filename, dtype="<i2")
        return torch.from_numpy(bits).view(torch.bfloat16).float().reshape(-1, 2048)

    x, weight = operand("input.bin"), operand("weights.bin")
    sampler = ctypes.CDLL(str(root / "sampler.so"))
    for rows in (2, 4, 7):
        a = x[:rows]
        path = root / f"skinny{rows}"
        torch.nn.functional.linear(a, weight).numpy().tofile(path.with_suffix(".bin"))
        assert sampler.begin_samples() == 0
        for _ in range(10000):
            torch.nn.functional.linear(a, weight)
        sampler.end_samples(os.fsencode(path.with_suffix(".pcs")))
    (root / "skinny-maps").write_text(Path("/proc/self/maps").read_text())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
