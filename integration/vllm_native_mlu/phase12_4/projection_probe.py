"""Run the accepted projection arithmetic on original checkpoint operands."""

import csv
import json
from pathlib import Path
import sys

from safetensors import safe_open
import torch
import torch_mlu

from .projection import PackedProjection
from runtime.native.phase9.softmax.acceptance import measure


def main(library, checkpoint, fixture, report):
    checkpoint = Path(checkpoint)
    weights = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    torch.set_num_threads(2)
    assert torch_mlu.mlu.get_device_name(0) == "MLU590-M9"
    stream = torch_mlu.mlu.Stream(device=0)
    print(f"native projection probe device=MLU590-M9 queue={stream.mlu_stream:#x}", flush=True)
    x = torch.from_file(fixture, dtype=torch.int16, size=2048).view(torch.bfloat16)[None]
    failures = 0
    with torch_mlu.mlu.stream(stream), open(report, "w", newline="") as output:
        writer = csv.writer(output)
        for layer in (0, 26):
            for suffix in ("q_proj", "kv_a_proj_with_mqa"):
                name = f"model.layers.{layer}.self_attn.{suffix}.weight"
                with safe_open(checkpoint / weights[name], framework="pt", device="cpu") as file:
                    weight = file.get_tensor(name)
                reference = torch.nn.functional.linear(x.float(), weight.float())
                with PackedProjection(library, weight.mlu()) as operation:
                    actual = operation(x.mlu()).cpu()
                bits, bad, absolute, relative, ulp, _, _, _ = measure(reference, actual)
                first = (actual.view(torch.int32) != reference.view(torch.int32)).flatten().nonzero()
                index = int(first[0, 0]) if first.numel() else None
                writer.writerow((name, actual.numel(), bits, bad, absolute, relative, ulp,
                                 index if index is not None else "", "FAIL" if bad else "PASS",
                                 reference.flatten()[index].item() if index is not None else "",
                                 actual.flatten()[index].item() if index is not None else ""))
                failures += bad
    print(f"projection_tolerance_mismatches={failures}", flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
