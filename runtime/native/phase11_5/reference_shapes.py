"""Validate the sampled CPU MKL accumulators as the token dimension varies."""

from pathlib import Path
import sys

import numpy as np
import torch

from orders import gemv_prefix4, panel112


def main(root):
    torch.set_num_threads(2)
    x = np.fromfile(root / "input.bin", dtype="<u2").reshape(-1, 2048)
    w = np.fromfile(root / "weights.bin", dtype="<u2").reshape(64, 2048)

    def widen(bits):
        return (bits.astype(np.uint32) << 16).view(np.float32)

    a = widen(x)[np.arange(128) % len(x)]
    b = widen(w)
    p = a[:, None, :] * b[None, :, :]
    gemv, _ = gemv_prefix4(p)
    gemm, _ = panel112(p)
    for rows in range(1, 129):
        ref = torch.nn.functional.linear(torch.from_numpy(a[:rows]), torch.from_numpy(b)).numpy()
        good_m = (ref.view(np.uint32) == gemm[:rows].view(np.uint32)).all(axis=1)
        good_v = (ref.view(np.uint32) == gemv[:rows].view(np.uint32)).all(axis=1)
        pattern = "".join("M" if m else "V" if v else "?" for m, v in zip(good_m, good_v))
        print(f"{rows:3d} {pattern}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
