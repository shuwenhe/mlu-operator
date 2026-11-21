"""Verify exact-rsqrt device output on the remaining Phase 8.2 checkpoint chunk."""
import sys
from pathlib import Path
import numpy as np
import torch

mode, trace, fixture, case = sys.argv[1:]
trace,fixture,case=map(Path,(trace,fixture,case))
x=torch.from_numpy(np.fromfile(trace/"1_0_residual.bin","<f4")).reshape(128,2048).half()
g=torch.from_file(str(fixture/"post_gamma.bin"),size=2048,dtype=torch.int16).view(torch.bfloat16).half()
if mode=="prep":
    x.view(torch.int16).numpy().tofile(str(case)+".in.u16")
    g.view(torch.int16).numpy().tofile(str(case)+".gamma.u16")
else:
    got=torch.from_numpy(np.fromfile(str(case)+".native.u16","<i2")).view(torch.float16).reshape_as(x)
    ref=((x.float()*(x.float().square().mean(-1,keepdim=True)+1e-6).rsqrt()).half()*g)
    bits=(ref.view(torch.int16)!=got.view(torch.int16)).sum().item()
    bad=((ref.float()-got.float()).abs()>1e-5+1e-5*ref.float().abs()).sum().item()
    print("target post norm", "mismatches", bad, "bit differences", bits, "elements", x.numel())
    assert bad==0
