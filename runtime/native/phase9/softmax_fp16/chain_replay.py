"""Run native QK, Softmax and AV on the same queue with device-resident handoffs."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "softmax"))
from acceptance import measure


def run(binary, name, mode, scale, inputs, unmasked, samples, perf, accuracy, variant):
    slots = len(inputs)
    q, k, _, positions, *_ = inputs[0]
    qlen, keys = q.shape[1], k.shape[1]
    offset = int(positions[0])
    args = [str(binary), name, str(qlen), str(keys), str(offset),
            str(scale), str(slots), str(samples), str(int(unmasked))]
    payload = b"".join(q.tobytes() + k.tobytes() + v.tobytes() +
                       (pos - offset).astype(np.uint32).tobytes()
                       for q, k, v, pos, *_ in inputs)
    result = subprocess.run(args, input=payload, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    for line in result.stderr.decode().splitlines():
        tag, slot, sample, qk, softmax, av, chain, wall, mem = line.split(",")
        assert tag == "TIMING"
        perf.writerow((variant, name, mode, qlen, keys, slots, slot, sample, qk, softmax,
                       av, chain, wall, mem, qlen * slots * 1e6 / float(wall)))
    dtype = torch.bfloat16 if name == "bf16" else torch.float16
    count, out_count = 16 * qlen * keys, 16 * qlen * 128
    bytes_per_slot = 6 * (count + out_count)
    if len(result.stdout) != slots * bytes_per_slot:
        raise ValueError("Malformed chain result")
    result_bytes = bytearray(result.stdout)
    failures = 0
    for slot, (_, _, v, _, ref_qk, ref_p, ref_av, request) in enumerate(inputs):
        base = slot * bytes_per_slot
        logits = torch.frombuffer(result_bytes, dtype=torch.float32, count=count, offset=base)
        probabilities = torch.frombuffer(result_bytes, dtype=dtype, count=count, offset=base+4*count)
        raw_av = torch.frombuffer(result_bytes, dtype=torch.float32, count=out_count, offset=base+6*count)
        av = torch.frombuffer(result_bytes, dtype=dtype, count=out_count,
                              offset=base+6*count+4*out_count)
        vals = torch.from_numpy(v).view(dtype)
        raw_ref = ref_p.float() @ vals.float()
        for stage, ref, device in (("qk", ref_qk, logits), ("softmax", ref_p, probabilities),
                                   ("av_fp32", raw_ref, raw_av), ("av_low", ref_av, av)):
            stats = measure(ref.flatten(), device)
            accuracy.writerow((variant, name, mode, qlen, keys, slots, request, stage, *stats))
            failures += stats[1]
    return failures


def main():
    qk_dir, av_dir, old, new, output = map(Path, sys.argv[1:])
    torch.set_num_threads(2)
    with (output / "performance.csv").open("w") as pf, (output / "chain.csv").open("w") as af:
        perf = csv.writer(pf, lineterminator="\n")
        perf.writerow(("variant", "dtype", "mode", "queries", "keys", "queues", "slot", "sample",
                       "qk_us", "softmax_us", "av_us", "chain_us", "wall_us", "buffer_bytes_per_slot",
                       "query_tokens_per_second"))
        accuracy = csv.writer(af, lineterminator="\n")
        accuracy.writerow(("variant", "dtype", "mode", "queries", "keys", "queues", "request",
                           "stage", "bit_differences", "mismatches", "max_abs", "max_rel",
                           "max_ulp", "nan", "inf", "unexpected_nonfinite"))
        outcomes = {}
        for variant, binary in (("baseline", old), ("corrected", new)):
            failures = 0
            for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
                for length in (1, 7, 32, 128):
                    folder = f"{name}-{length}"
                    for step, mode in enumerate(("prefill", "decode")):
                        inputs = []
                        for req in range(2):
                            case = req * 2 + step
                            with np.load(qk_dir / folder / f"qk_{case}.npz") as qk:
                                q, k, pos, scale, masked = (qk[key] for key in
                                                            ("q","k","positions","scale","masked"))
                            with np.load(av_dir / folder / f"av_{case}.npz") as fixture:
                                v, p, av = (fixture[key] for key in ("v","probs","out"))
                            inputs.append((q,k,v,pos,torch.from_numpy(masked).view(dtype).float(),
                                           torch.from_numpy(p).view(dtype),
                                           torch.from_numpy(av).view(dtype),req))
                        for item in inputs:
                            failures += run(binary, name, mode, float(scale), [item], False,
                                            3, perf, accuracy, variant)
                        failures += run(binary, name, mode, float(scale), inputs, False,
                                        3, perf, accuracy, variant)
                    if variant == "corrected":
                        # Eight noncausal synthetic runs, complementing the documented causal AV suite.
                        q = np.zeros((16,length,192), dtype=np.int16)
                        pos = np.arange(length, dtype=np.uint32)
                        vals = torch.linspace(-0.5,0.5,length*128).view(1,length,128).expand(16,-1,-1)
                        vals = vals.to(dtype).contiguous()
                        logits = torch.zeros((16,length,length))
                        p = torch.softmax(logits, -1).to(dtype)
                        inputs = [(q,q,vals.view(torch.int16).numpy(),pos,logits,p,p @ vals,0)]
                        failures += run(binary, name, "unmasked", 1.0, inputs,
                                        True, 3, perf, accuracy, variant)
            outcomes[variant] = failures
    print(json.dumps(outcomes), flush=True)
    return int(outcomes["corrected"] != 0 or outcomes["baseline"] != 4)


if __name__ == "__main__":
    raise SystemExit(main())
