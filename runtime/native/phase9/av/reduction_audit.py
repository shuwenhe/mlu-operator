"""Matched-input CPU reduction diagnostic for the AVX2-order AV experiment."""
from pathlib import Path
import sys

import numpy as np
import torch

from replay_av import replay


def fold(v):
    a, b, c, d = v[0]+v[4], v[1]+v[5], v[2]+v[6], v[3]+v[7]
    return (a + c) + (b + d)


def avx2(p, v):
    length = p.shape[-1]
    shape = (*p.shape[:-1], v.shape[-1])
    accum = np.zeros((64, *shape), dtype=np.float32)
    tail = np.zeros((8, *shape), dtype=np.float32)
    full, vector_end = length // 64 * 64, length // 16 * 16
    for key in range(vector_end):
        products = p[..., key, None] * v[:, None, key, :]
        if key < full:
            accum[key % 64] += products
        else:
            tail[key % 8] += products
    a = accum
    lanes = ((a[:8]+a[32:40])+(a[16:24]+a[48:56]))+((a[24:32]+a[56:64])+(a[8:16]+a[40:48]))
    result = fold(lanes) + fold(tail)
    for key in range(vector_end, length):
        result += p[..., key, None] * v[:, None, key, :]
    return torch.from_numpy(result)


def main():
    root, binary = map(Path, sys.argv[1:])
    torch.set_num_threads(2)
    for path in sorted(root.glob("fp16-*/av_*.npz")):
        with np.load(path) as f:
            p, v, reference = (torch.from_numpy(f[key]).view(torch.float16)
                                for key in ("probs", "v", "out"))
        cpu = avx2(p.float().numpy(), v.float().numpy())
        raw, low = replay(binary, "fp16", p, v)
        indices = torch.nonzero(low != reference)
        if indices.numel():
            index = tuple(indices[0].tolist())
            print(path.parent.name, path.name, f"raw_diff={(cpu != raw).sum().item()}",
                  f"cpu_cast_diff={(cpu.half() != reference).sum().item()}",
                  f"device_cast_diff={(raw.half() != low).sum().item()}", index,
                  f"device_raw={raw[index].item():.12g} cpu_raw={cpu[index].item():.12g}",
                  f"ref={reference[index].item():.12g} device={low[index].item():.12g}",
                  flush=True)


if __name__ == "__main__":
    main()
