"""Check the installed spec, storage aliasing, and native page movement on card 0."""

import csv
import sys

import torch
import torch_mlu

from .cache_spec import per_layer_specs, reshape_kv_cache
from ..phase12.paged_kv import PagedKV
from vllm.v1.kv_cache_interface import FullAttentionSpec


def main(library, report):
    device = torch.device("mlu:0")
    assert torch_mlu.mlu.get_device_name(0) == "MLU590-M9"
    stream = torch_mlu.mlu.Stream(device=device)
    print(f"cache_spec_probe device={device} name=MLU590-M9 queue={stream.mlu_stream:#x}", flush=True)
    with torch_mlu.mlu.stream(stream), open(report, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("case", "elements", "bit_mismatches", "tolerance_mismatches",
                         "max_abs", "max_rel", "max_ulp", "first_index", "status"))
        names = [f"model.layers.{i}.self_attn.native_mla" for i in range(27)]
        specs = per_layer_specs(names, 16)
        combined = FullAttentionSpec.merge(list(specs.values()))
        assert combined.page_size_bytes == 16 * 10240
        for name, spec in specs.items():
            slots = 3 * spec.block_size
            raw = torch.empty(3 * spec.page_size_bytes, device=device, dtype=torch.int8)
            keys, values = reshape_kv_cache(raw, spec, 3, 16, "auto")
            assert keys.stride() == (3072, 192, 1) and values.stride() == (2048, 128, 1)
            assert keys.data_ptr() == raw.data_ptr()
            assert values.data_ptr() == raw.data_ptr() + slots * 16 * 192 * 2
            cache = PagedKV(library, keys, values, 16)
            rng = torch.Generator().manual_seed(124)
            reference = tuple(torch.randn(16, 17, width, generator=rng).bfloat16()
                              for width in (192, 128))
            mapping = torch.tensor([*range(32, 48), 0], dtype=torch.int32, device=device)
            cache.write(*(x.to(device) for x in reference), mapping)
            readback = cache.read(torch.tensor([2, 0], dtype=torch.int32, device=device), 17)
            for stage, got, expected in zip(("key", "value"), readback, reference, strict=True):
                differences = (got.cpu().view(torch.int16) != expected.view(torch.int16)).flatten()
                count = int(differences.sum())
                first = int(differences.nonzero()[0]) if count else ""
                if count:
                    writer.writerow((f"{name}.{stage}", got.numel(), count, "", "", "", "", first, "FAIL"))
                    raise AssertionError(f"{name} {stage} cache readback mismatch at {first}")
                writer.writerow((f"{name}.{stage}", got.numel(), 0, 0, 0, 0, 0, "", "PASS"))
        print(f"cache_spec_probe PASS layers={len(specs)} block_size=16 page_bytes={combined.page_size_bytes}")


if __name__ == "__main__":
    main(*sys.argv[1:])
