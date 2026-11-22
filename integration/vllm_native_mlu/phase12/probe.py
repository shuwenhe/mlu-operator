"""Probe physical remapping, request isolation, append, padding and page reuse."""

import sys

import torch
import torch_mlu

from .paged_kv import PagedKV


def main(library):
    assert torch_mlu.mlu.get_device_name(0) == "MLU590-M9"
    stream = torch_mlu.mlu.Stream(device=0)
    print(f"card=0 device=MLU590-M9 queue={stream.mlu_stream:#x}", flush=True)
    rng = torch.Generator().manual_seed(12)
    with torch_mlu.mlu.stream(stream):
        for dtype in (torch.bfloat16, torch.float16):
            page_k = torch.zeros((48, 16, 192), device="mlu:0", dtype=dtype)
            page_v = torch.zeros((48, 16, 128), device="mlu:0", dtype=dtype)
            cache = PagedKV(library, page_k, page_v, 8)
            a = tuple(torch.randn(16, 10, w, generator=rng).to(dtype) for w in (192, 128))
            b = tuple(torch.randn(16, 5, w, generator=rng).to(dtype) for w in (192, 128))
            c = tuple(torch.randn(16, 1, w, generator=rng).to(dtype) for w in (192, 128))

            def mapping(items):
                return torch.tensor(items, dtype=torch.int32, device="mlu:0")

            def check(blocks, length, expected):
                for actual, want in zip(cache.read(mapping(blocks), length), expected):
                    assert torch.equal(actual.cpu().view(torch.int16), want.view(torch.int16))

            cache.write(*(x.mlu() for x in a), mapping([*range(32, 40), 8, 9]))
            cache.write(*(x.mlu() for x in b), mapping(range(16, 21)))
            check([4, 1], 10, a)
            check([2], 5, b)
            cache.write(*(x.mlu() for x in c), mapping([10]))
            check([4, 1], 11, [torch.cat((x, y), dim=1) for x, y in zip(a, c)])
            cache.write(*(x.mlu() for x in c), mapping([-1]))
            check([2], 5, b)
            # The harness releases page 4 and reassigns it to a new owner here.
            cache.write(*(x.mlu() for x in c), mapping([32]))
            check([4], 1, c)
            check([2], 5, b)
            print(f"dtype={dtype} remap=PASS append=PASS isolation=PASS padding=PASS reuse=PASS",
                  flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
