"""Extract the exact BF16 router operands and sample the pinned FP32 CPU path."""
import ctypes
import os
from pathlib import Path
import sys

import numpy as np
import torch


def main(checkpoint, captures, output):
    sampler = ctypes.CDLL(str(output / "sampler.so"))
    torch.set_num_threads(2)
    for layer, m in ((26,1),(2,1),(2,17)):
        fixture = captures / f"diag-{layer}"
        entry = next(line.split() for line in (fixture/"weights.txt").read_text().splitlines()
                     if line.startswith("gate.weight "))
        _, shard, offset, n, k = entry
        n, k = int(n), int(k)
        full_rows=17 if layer==2 else 1
        source = np.fromfile(fixture/f"input-{full_rows}.bin",dtype="<u2").reshape(-1,k)[:m]
        weights = np.memmap(checkpoint/shard,mode="r",offset=int(offset),dtype="<u2",
                            shape=(n,k)).copy()
        path = output/f"layer{layer}-m{m}"
        path.mkdir(exist_ok=True)
        source.tofile(path/"input.bin")
        weights.tofile(path/"weights.bin")
        a,b = (torch.from_numpy(x.copy()).view(torch.bfloat16).float()
               for x in (source,weights))
        project=lambda:torch.nn.functional.linear(a,b)
        result=project()
        if m==full_rows:
            old=np.fromfile(fixture/f"logits-{m}.bin",dtype="<f4").reshape(m,n)
            assert np.array_equal(old,result.numpy())
        result.numpy().tofile(path/"reference.bin")
        (path/"maps").write_text(Path("/proc/self/maps").read_text())
        flags=os.environ.pop("MKL_VERBOSE",None)
        # The verbose descriptor is recorded separately to avoid logging every sampled call.
        assert sampler.begin_samples()==0
        for _ in range(8000):
            project()
        print(f"layer={layer} rows={m} samples={sampler.end_samples(os.fsencode(path/'pcs'))}",flush=True)
        if flags is not None:
            os.environ["MKL_VERBOSE"]=flags


if __name__=="__main__":
    main(*map(Path,sys.argv[1:]))
