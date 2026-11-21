"""Replay the checkpoint RMSNorm on the candidate's exact saved norm inputs."""
import ast
import csv
import sys
from pathlib import Path

import torch

from projections import compare, fp32_file

model, weights, traces = map(Path, sys.argv[1:])
source = ast.parse(model.read_text())
definition, = (n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2RMSNorm")
scope = {"torch": torch, "nn": torch.nn}
exec(compile(ast.Module(body=[definition], type_ignores=[]), str(model), "exec"), scope)
torch.set_num_threads(2)
out = csv.writer(sys.stdout)
out.writerow(("dtype", "prefill", "request", "position", "stage", "elements", "mismatches",
              "nonfinite", "max_abs", "max_rel", "first_mismatch", "max_ulp", "bit_differences",
              "first_bit_difference", "torch_bits", "device_bits", "first_mismatch_ulp"))
for line in (traces / "trace_manifest").read_text().splitlines():
    dtype, length, path = line.split()
    low = torch.float16 if dtype == "fp16" else torch.bfloat16
    norm = scope["DeepseekV2RMSNorm"](2048).eval()
    with torch.no_grad():
        gamma = torch.from_file(str(weights / "post_gamma.bin"), size=2048, dtype=torch.int16)
        norm.weight.copy_(gamma.view(torch.bfloat16))
        norm.to(low)
        for request in range(2):
            for position in (0, int(length)):
                prefix = f"{request}_{position}_"
                activation = fp32_file(Path(path) / f"{prefix}residual.bin", 2048)
                result = fp32_file(Path(path) / f"{prefix}post_norm.bin", 2048)
                expected = norm(activation.to(low)).float()
                out.writerow((dtype, length, request, position, "post_norm", *compare(expected, result, low)))
