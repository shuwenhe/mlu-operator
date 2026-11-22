"""Exercise the native attention chain through vLLM CustomOp on checkpoint tensors."""

import csv
from pathlib import Path
import sys

import numpy as np
import torch
import torch_mlu

from integration.vllm_native_mlu.attention import NativeMaterializedAttention
from runtime.native.phase9.softmax.acceptance import measure
from vllm.config import VllmConfig, set_current_vllm_config


def main(library, report, qk_dir, av_dir):
    torch.set_num_threads(2)
    with set_current_vllm_config(VllmConfig()):
        op = NativeMaterializedAttention(library)
    streams = [torch_mlu.mlu.Stream(device=0) for _ in range(2)]
    assert torch_mlu.mlu.get_device_name(0) == "MLU590-M9"
    print(f"device=MLU590-M9 queues={[s.mlu_stream for s in streams]}", flush=True)
    failures = 0
    with open(report, "w", newline="") as output:
        writer = csv.writer(output)
        for tag, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            for size in (1, 7, 32, 128):
                for case in range(4):
                    folder = f"{tag}-{size}"
                    with np.load(Path(qk_dir) / folder / f"qk_{case}.npz") as qk:
                        q, k = (torch.from_numpy(qk[key]).view(dtype) for key in ("q", "k"))
                        pos = qk["positions"].astype(np.int32)
                        offset = int(pos[0])
                        scale = float(qk["scale"])
                        logits = torch.from_numpy(qk["masked"]).view(dtype).float()
                    with np.load(Path(av_dir) / folder / f"av_{case}.npz") as av:
                        v, p, out = (torch.from_numpy(av[key]).view(dtype) for key in ("v", "probs", "out"))
                    references = (logits, p, p.float() @ v.float(), out)
                    with torch_mlu.mlu.stream(streams[case // 2]):
                        results = [t.cpu() for t in op(q.mlu(), k.mlu(), v.mlu(),
                                                       torch.from_numpy(pos - offset).mlu(),
                                                       offset, k.shape[1], scale)]
                    for stage, actual, expected in zip(("qk", "softmax", "av_fp32", "av_low"),
                                                       results, references):
                        bits, bad, absolute, relative, ulp, _, _, invalid = measure(expected, actual)
                        failures += bad
                        writer.writerow([f"attention_{folder}_slot{case // 2}_step{case % 2}_{stage}",
                                         actual.numel(), bits, bad, absolute, relative, ulp,
                                         invalid, "FAIL" if bad else "PASS"])
    print(f"attention_tolerance_mismatches={failures}", flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
