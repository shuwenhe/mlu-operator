"""Replay exact checkpoint Q/K bits and report each numerical boundary."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def report(case, stage, expected, actual, writer):
    ref, got = (x.detach().contiguous().numpy() for x in (expected, actual))
    low_stage = stage in {"cast1_reconstructed", "cast1_device", "scaled", "masked"}
    fp16 = case[1] == "fp16" and low_stage
    raw_type = np.int16 if fp16 else np.int32
    if expected.dtype == torch.bfloat16:
        raise TypeError("Widen BF16 before reporting")
    er, ar = ((x.astype(np.float16) if fp16 else x).view(raw_type).astype(np.int64) for x in (ref, got))
    signed = 1 << (np.dtype(raw_type).itemsize * 8 - 1)
    ulp = np.abs(np.where(er < 0, -signed - er, er)
                 - np.where(ar < 0, -signed - ar, ar))
    keep = np.isfinite(ref) & np.isfinite(got)
    abs_error = np.abs(ref[keep].astype(np.float64) - got[keep])
    magnitude = np.abs(ref[keep].astype(np.float64))
    delta = np.zeros(ref.shape, dtype=bool)
    delta[keep] = abs_error > 1e-5 + 1e-5 * magnitude
    bad_mask = np.isneginf(ref) != np.isneginf(got) if stage == "masked" else np.zeros(ref.shape, bool)
    unexpected = ~keep & ~(np.isneginf(ref) & np.isneginf(got) & (stage == "masked"))
    delta |= unexpected
    bf16 = case[1] == "bf16" and stage in {"cast1_reconstructed", "cast1_device", "scaled", "masked"}
    if bf16:
        er, ar, ulp, signed = er >> 16, ar >> 16, ulp >> 16, 1 << 15
    first = np.argwhere(delta)
    index = tuple(first[0]) if first.size else (() if not delta.ndim and delta else None)
    writer.writerow((*case, stage, "x".join(map(str, ref.shape)), ref.size,
                     int(delta.sum()), int((er != ar).sum()), int(unexpected.sum()),
                     int(bad_mask.sum()), abs_error.max(initial=0),
                     (abs_error / np.maximum(magnitude, 1e-30)).max(initial=0),
                     ulp[keep].max(initial=0), "" if index is None else "/".join(map(str, index)),
                     "" if index is None else hex(int(er[index]) & (2 * signed - 1)),
                     "" if index is None else hex(int(ar[index]) & (2 * signed - 1))))


def main():
    root, experiment = Path(sys.argv[1]), Path(sys.argv[2])
    torch.set_num_threads(2)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("strategy", "dtype", "prefill", "request", "pass", "stage",
                     "shape", "elements", "mismatches", "bit_differences",
                     "unexpected_nonfinite", "mask_failures", "max_abs", "max_rel",
                     "max_ulp", "first_mismatch", "expected_bits", "actual_bits"))
    for folder in sorted(root.glob("*-*")):
        name, length = folder.name.split("-")
        if name not in ("bf16", "fp16"):
            continue
        dtype = torch.bfloat16 if name == "bf16" else torch.float16
        for path in sorted(folder.glob("qk_*.npz")):
            index = int(path.stem.split("_")[1])
            with np.load(path) as fixture:
                q, k, first, scaled, masked = (
                    torch.from_numpy(fixture[key]).view(dtype)
                    for key in ("q", "k", "qk", "scaled", "masked"))
                position = fixture["positions"]
                scale = float(fixture["scale"])
            seq, nk = q.shape[1], k.shape[1]
            size = first.numel()
            base = int(position[0])
            case = (experiment.name, name, length, index // 2, "decode" if base else "prefill")
            inputs = (q.view(torch.int16).numpy().tobytes() + k.view(torch.int16).numpy().tobytes()
                      + (position - base).astype(np.uint32).tobytes())
            flags = ["0", name, str(seq), str(nk), str(base), str(scale)]

            def run(binary):
                result = subprocess.run([str(binary), *flags], input=inputs,
                                        stdout=subprocess.PIPE, check=True).stdout
                if len(result) != 10 * size:
                    raise ValueError(f"Malformed device output for {case}")
                payload = bytearray(result)
                return tuple(torch.frombuffer(payload, dtype=kind, count=size, offset=offset)
                             .reshape(first.shape) for kind, offset in
                             ((torch.float32, 0), (dtype, 4 * size), (torch.float32, 6 * size)))

            raw, low, mask = run(experiment / "qk_device_test")
            dot64 = q.double() @ k.double().transpose(-1, -2)
            dot32 = q.float() @ k.float().transpose(-1, -2)
            # The CPU low-precision matmul does not expose its internal FP32 accumulator.
            report(case, "raw_vs_fp64_rounded", dot64.float(), raw, writer)
            report(case, "raw_vs_fp32_matmul", dot32, raw, writer)
            report(case, "cast1_reconstructed", first.float(), raw.to(dtype).float(), writer)
            scale_ref = torch.tensor(scale, dtype=torch.float32)
            report(case, "scale_fp32", scale_ref, torch.tensor(np.float32(scale)), writer)
            report(case, "scale_product_reconstructed", first.float() * scale_ref,
                   raw.to(dtype).float() * scale_ref, writer)
            if len(sys.argv) > 3:
                product, diagnostic_low, device_first = run(Path(sys.argv[3]))
                report(case, "cast1_device", first.float(), device_first, writer)
                report(case, "scale_product_device", first.float() * scale_ref, product, writer)
                report(case, "diagnostic_matches_production", low.float(), diagnostic_low.float(), writer)
            report(case, "scaled", scaled.float(), low.float(), writer)
            report(case, "masked", masked.float(), mask, writer)


if __name__ == "__main__":
    main()
