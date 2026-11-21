"""Record widened PyTorch FP32 output separately from low-dtype linear parity."""
import csv
import hashlib
import sys
from pathlib import Path

import torch
from torch.nn.functional import linear

from projections import PROJECTIONS, compare, fp32_file, word_array

fixtures, trace, output = map(Path, sys.argv[1:4])
dtype, length = sys.argv[4], int(sys.argv[5])
low = torch.float16 if dtype == "fp16" else torch.bfloat16
torch.set_num_threads(4)
with (output / "accuracy.csv").open() as stream:
    fields = next(csv.reader(stream))
writer = csv.writer(sys.stdout)
writer.writerow(fields)
for stage, (input_name, k, n) in PROJECTIONS.items():
    weight = torch.from_file(str(fixtures / f"{stage}.bin"), size=n*k, dtype=torch.int16)
    weight = weight.view(torch.bfloat16).reshape(n, k).to(low)
    weight_sha = hashlib.sha256(word_array(weight)).hexdigest()
    weight32 = weight.float()
    for request in range(2):
        for position in (0, length):
            prefix = f"{request}_{position}_"
            x = fp32_file(trace / f"{prefix}{input_name}.bin", k).to(low)
            activation_sha = hashlib.sha256(word_array(x)).hexdigest()
            fp32 = linear(x.float(), weight32)
            fp32.numpy().tofile(output / f"{prefix}{stage}_widened_torch.f32")
            native = fp32_file(trace / f"{prefix}{stage}_fp32.bin", n)
            writer.writerow((dtype, length, request, position, stage, "matched_widened_pytorch_fp32",
                             x.shape[0], k, n, k, k, n, weight_sha, activation_sha,
                             *compare(fp32, native, low)))
