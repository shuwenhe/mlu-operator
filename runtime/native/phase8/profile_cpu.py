"""Resolve sampled CPU instruction pointers from the pinned reference library."""
import bisect
import collections
import ctypes
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

root, replay, output = map(Path, sys.argv[1:])
sampler = ctypes.CDLL(str(output / "pc_sampler.so"))
output.mkdir(exist_ok=True)
torch.set_num_threads(2)
raw = np.fromfile(root / "gate.bin", "<i2")
for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
    w = torch.from_numpy(raw).view(torch.bfloat16).reshape(10944, 2048).to(dtype)
    for length in (1, 7, 32, 128):
        x = np.fromfile(replay / f"{name}-{length}" / "0_0_gate_input.u16", "<i2")
        x = torch.from_numpy(x).view(dtype).reshape(length, 2048)
        torch.nn.functional.linear(x, w)
        path = output / f"{name}-{length}.pc"
        assert sampler.begin_samples() == 0
        end = time.monotonic() + 0.35
        while time.monotonic() < end:
            torch.nn.functional.linear(x, w)
        print(name, length, sampler.end_samples(str(path).encode()), flush=True)

maps = Path("/proc/self/maps").read_text()
(output / "maps").write_text(maps)
intervals = []
for line in maps.splitlines():
    cols = line.split(maxsplit=5)
    if len(cols) == 6 and "x" in cols[1] and cols[5].startswith("/"):
        start, end = (int(value, 16) for value in cols[0].split("-"))
        intervals.append((start, end, int(cols[2], 16), cols[5]))
cached = {}
for sample in sorted(output.glob("*.pc")):
    hits = collections.Counter()
    for line in sample.read_text().splitlines():
        pc = int(line, 16)
        for lo, hi, offset, lib in intervals:
            if not lo <= pc < hi:
                continue
            if lib not in cached:
                symbols = subprocess.run(["nm", "-nCS", "--defined-only", lib],
                                         capture_output=True, text=True, check=False).stdout
                ranges = []
                for symbol in symbols.splitlines():
                    parts = symbol.split(maxsplit=3)
                    if len(parts) == 4 and parts[2] in {"t", "T", "w", "W"}:
                        ranges.append((int(parts[0], 16), int(parts[1], 16), parts[3]))
                cached[lib] = ranges, [item[0] for item in ranges]
            segments, offsets = cached[lib]
            address = pc - lo + offset
            index = bisect.bisect_right(offsets, address) - 1
            label = Path(lib).name
            if index >= 0 and address < segments[index][0] + segments[index][1]:
                label += " " + segments[index][2]
            hits[label] += 1
            break
    print(sample.stem, hits.most_common(6), flush=True)
