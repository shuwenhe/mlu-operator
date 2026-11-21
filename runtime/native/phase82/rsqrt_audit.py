"""Compare checkpoint reciprocal sqrt with binary32 sqrt plus division."""
from pathlib import Path
import sys
import numpy as np
import torch

p = Path(sys.argv[1])
bad = total = 0
for line in (p / "trace_manifest").read_text().splitlines():
    dtype, _, trace = line.split()
    low = torch.bfloat16 if dtype == "bf16" else torch.float16
    for f in sorted(Path(trace).glob("*_residual.bin")):
        x = torch.from_numpy(np.fromfile(f,np.float32)).reshape(-1,2048).to(low).float()
        eps = x.square().mean(-1) + 1.e-6
        want = eps.rsqrt().numpy()
        got = np.float32(1.) / np.sqrt(eps.numpy())
        bad += np.count_nonzero(want != got)
        total += len(want)
print("checkpoint rsqrt versus FP32 sqrt then FP32 division", bad, "/", total)
