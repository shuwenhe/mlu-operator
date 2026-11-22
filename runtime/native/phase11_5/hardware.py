"""Record router multiplication and accumulation traces on MLU card 0."""

import ctypes
import csv
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import torch_mlu

from orders import compare, gemv_prefix4, panel112, partials


def main(library, root):
    launch = ctypes.CDLL(library).phase115_router_probe
    launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 2
    launch.restype = ctypes.c_int
    stream = torch_mlu.mlu.Stream(device=0)
    print(f"device={torch_mlu.mlu.get_device_name(0)} queue={stream.mlu_stream:#x}", flush=True)
    assert torch_mlu.mlu.get_device_name(0) == "MLU590-M9"
    with (root / "hardware-accuracy.csv").open("w", newline="") as report:
        writer = csv.writer(report)
        for case in sorted(root.glob("layer*-m*")):
            a = np.fromfile(case / "input.bin", dtype="<u2").reshape(-1, 2048)
            w = np.fromfile(case / "weights.bin", dtype="<u2").reshape(64, 2048)
            ref = np.fromfile(case / "reference.bin", dtype="<f4").reshape(-1, 64)
            xa, wa = [(v.astype(np.uint32) << 16).view(np.float32) for v in (a, w)]
            expected_products = xa[:, None, :] * wa[None, :, :]
            with torch_mlu.mlu.stream(stream):
                x, weight = [torch.from_numpy(v.view(np.int16)).view(torch.bfloat16).mlu() for v in (a,w)]
                scores = torch.empty(ref.shape, device=x.device, dtype=torch.float32)
                products = torch.empty(expected_products.shape, device=x.device, dtype=torch.float32)
                stream.synchronize()
                for mode in (1, 4, 5, 6, 64):
                    lanes = {5: 4, 6: 19}.get(mode, mode)
                    sums = torch.empty((*ref.shape, lanes), device=x.device, dtype=torch.float32)
                    pointers = [t.data_ptr() for t in (x, weight, scores, products, sums)]
                    start = perf_counter()
                    status = launch(stream.mlu_stream, *pointers, len(a), mode)
                    if status:
                        raise RuntimeError(f"CNRT launch returned {status}")
                    stream.synchronize()
                    elapsed = (perf_counter() - start) * 1000
                    actual, p, s = [t.cpu().numpy() for t in (scores, products, sums)]
                    if mode == 5:
                        _, expected_sums = gemv_prefix4(expected_products)
                    elif mode == 6:
                        split = len(a) // 4 * 4
                        expected_sums = np.zeros_like(s)
                        _, expected_sums[:split] = panel112(expected_products[:split])
                        _, expected_sums[split:, :, :4] = gemv_prefix4(expected_products[split:])
                    else:
                        expected_sums = partials(expected_products, lanes)
                    product_bad = np.count_nonzero(p.view(np.uint32) != expected_products.view(np.uint32))
                    sums_bad = np.count_nonzero(s.view(np.uint32) != expected_sums.view(np.uint32))
                    print(f"{case.name} mode={mode} elapsed_ms={elapsed:.6f} "
                          f"product_bit_mismatch={product_bad} partial_bit_mismatch={sums_bad}", flush=True)
                    np.savez(case / f"hardware-mode{mode}.npz", scores=actual, products=p, partials=s)
                    compare(writer, case.name, f"mlu_mode{mode}", actual, ref)
                    if product_bad or sums_bad:
                        raise RuntimeError("MLU arithmetic differs from the explicitly ordered host control")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
