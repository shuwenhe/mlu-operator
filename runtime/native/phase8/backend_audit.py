"""Observe the pinned CPU PyTorch low-precision Linear calls on exact checkpoint inputs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

def main():
    p = argparse.ArgumentParser()
    p.add_argument("fixture", type=Path)
    p.add_argument("replay", type=Path)
    args = p.parse_args()
    print(torch.__version__, torch.__config__.show(), flush=True)
    print(torch.__config__.parallel_info(), flush=True)
    for threads in (2, 4):
        torch.set_num_threads(threads)
        for tag, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            raw = np.fromfile(args.fixture / "gate.bin", "<i2")
            weight = torch.from_numpy(raw).view(torch.bfloat16).reshape(10944, 2048).to(dtype)
            for length in (1, 7, 32, 128):
                prefix = args.replay / f"{tag}-{length}" / "0_0_gate"
                words = np.fromfile(str(prefix) + "_input.u16", "<i2")
                a = torch.from_numpy(words).view(dtype).reshape(length, 2048)
                cached = np.fromfile(str(prefix) + "_matched_torch.u16", "<i2")
                print("CALL", tag, length, threads, a.device, a.dtype, a.stride(), weight.stride(),
                      flush=True)
                out = F.linear(a, weight)
                baseline = (out.view(torch.int16).numpy().ravel() != cached).sum()
                with torch.backends.mkldnn.flags(enabled=False):
                    plain = F.linear(a, weight)
                changed = (out.view(torch.int16) != plain.view(torch.int16)).sum().item()
                print("RESULT", out.dtype, "vs_phase7_bits", baseline,
                      "mkldnn_disabled_bit_changes", changed, flush=True)

if __name__ == "__main__":
    main()
