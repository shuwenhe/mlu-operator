"""Compare every router length supported by the standalone model on card 0."""

import csv
import ctypes
from pathlib import Path
import sys

import numpy as np
import torch
import torch_mlu

from orders import compare


def main(library, fixture, output):
    torch.set_num_threads(2)
    launch = ctypes.CDLL(library).phase115_router_bf16
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]
    launch.restype = ctypes.c_int

    def operand(filename):
        bits = np.fromfile(fixture / filename, dtype="<i2")
        return torch.from_numpy(bits).view(torch.bfloat16).reshape(-1, 2048)

    x, w = operand("input.bin"), operand("weights.bin")
    x = x.repeat((8, 1))[:128].contiguous()
    a, b = x.float(), w.float()
    stream = torch_mlu.mlu.Stream(device=0)
    assert torch_mlu.mlu.get_device_name(0) == "MLU590-M9"
    print(f"card=0 name={torch_mlu.mlu.get_device_name(0)} queue={stream.mlu_stream:#x}")
    with torch_mlu.mlu.stream(stream), output.open("w", newline="") as report:
        xx, ww = x.mlu(), w.mlu()
        yy = torch.empty((128, 64), dtype=torch.float32, device=xx.device)
        writer = csv.writer(report)
        mismatches = 0
        for rows in range(1, 129):
            ref = torch.nn.functional.linear(a[:rows], b).numpy()
            status = launch(stream.mlu_stream, xx.data_ptr(), ww.data_ptr(), yy.data_ptr(), rows)
            if status:
                raise RuntimeError(f"Router launch failed for rows={rows}, status={status}")
            stream.synchronize()
            actual = yy[:rows].cpu().numpy()
            compare(writer, f"layer2-repeated-m{rows}", "router_all_supported_shapes", actual, ref)
            mismatches += np.count_nonzero(actual.view(np.uint32) != ref.view(np.uint32))
    print(f"shape_sweep_bit_mismatches={mismatches}")
    return int(mismatches != 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])))
