"""Check the adapter on checkpoint QKV with a deterministic metadata fixture."""

import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch_mlu

from ..attention import NativeMaterializedAttention
from ..phase12.paged_kv import PagedKV
from .scheduled_attention import ScheduledAttention, validate_batch
from runtime.native.phase9.softmax.acceptance import measure
from vllm.config import VllmConfig, set_current_vllm_config


def main(bridge, paged, qk_root, av_root, report):
    assert torch_mlu.mlu.get_device_name(0) == "MLU590-M9"
    stream = torch_mlu.mlu.Stream(device=0)
    print(f"fixture metadata only, device=MLU590-M9 queue={stream.mlu_stream:#x}", flush=True)
    with set_current_vllm_config(VllmConfig()):
        native = NativeMaterializedAttention(bridge)
    failed = 0
    with torch_mlu.mlu.stream(stream), open(report, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "backend", "elements", "bit_mismatches", "tolerance_mismatches",
                         "max_abs", "max_rel", "max_ulp", "nan", "inf", "unexpected_nonfinite"))
        for tag, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            def allocate(width):
                return torch.empty((48, 16, width), device="mlu:0", dtype=dtype)
            def integer(values):
                return torch.tensor(values, device="mlu:0", dtype=torch.int32)
            adapter = ScheduledAttention(PagedKV(paged, allocate(192), allocate(128), 4), native)
            for step in (0, 1):
                blocks, positions, slots, ends, lengths = [], [], [], [0], []
                batch_q, batch_k, batch_v, refs = [], [], [], []
                scale = None
                for req in (0, 1):
                    case = req * 2 + step
                    with np.load(Path(qk_root) / f"{tag}-7" / f"qk_{case}.npz") as qk:
                        q, k = (torch.from_numpy(qk[f]).view(dtype) for f in ("q", "k"))
                        pos = qk["positions"].tolist()
                        current_scale = float(qk["scale"])
                    with np.load(Path(av_root) / f"{tag}-7" / f"av_{case}.npz") as av:
                        v, expected = (torch.from_numpy(av[f]).view(dtype) for f in ("v", "out"))
                    if scale is not None and scale != current_scale:
                        raise ValueError("Fixture scales disagree")
                    scale = current_scale
                    block = ([5, 2, 9, 11] if req == 0 else [7, 1, 3, 4])
                    if len(pos) == k.shape[1]:
                        # The frozen test's logical prefix is already present in q/k/v.
                        assert pos == list(range(len(pos)))
                    blocks.append(block)
                    positions.extend(pos)
                    slots.extend(block[p // 4] * 4 + p % 4 for p in pos)
                    ends.append(ends[-1] + len(pos))
                    lengths.append(k.shape[1])
                    batch_q.append(q.transpose(0, 1))
                    batch_k.append(k[:, pos].transpose(0, 1))
                    batch_v.append(v[:, pos].transpose(0, 1))
                    refs.append(expected.transpose(0, 1))
                meta = SimpleNamespace(query_start_loc=integer(ends), seq_lens=integer(lengths),
                                       block_table_tensor=integer(blocks), slot_mapping=integer(slots))
                output = adapter(meta, ["fixture-0", "fixture-1"], integer(positions),
                                 *(torch.cat(t).mlu() for t in (batch_q, batch_k, batch_v)),
                                 scale).cpu()
                for request, (start, end) in enumerate(zip(ends, ends[1:])):
                    actual = output[start:end].contiguous()
                    stats = measure(refs[request].contiguous(), actual)
                    writer.writerow((f"{tag}-request{request}-step{step}", "native_fixture",
                                     actual.numel(), *stats))
                    failed += stats[1]
                meta.slot_mapping = integer([-1, *slots[1:]])
                try:
                    validate_batch(meta, ["fixture-0", "fixture-1"], integer(positions), 4, 12)
                except ValueError:
                    print(f"{tag} step={step} invalid_slot_rejected", flush=True)
                else:
                    raise AssertionError("Invalid live slot accepted")
    print(f"fixture_tolerance_mismatches={failed}", flush=True)
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
