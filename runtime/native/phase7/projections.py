"""Compare device projections on identical native input bits and checkpoint weights."""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECTIONS = {
    "q": ("input_norm", 2048, 3072),
    "kv_a": ("input_norm", 2048, 576),
    "kv_b": ("kv_norm", 512, 4096),
    "o": ("attention", 2048, 2048),
    "gate": ("post_norm", 2048, 10944),
    "up": ("post_norm", 2048, 10944),
    "down": ("swiglu", 10944, 2048),
}


def fp32_file(path, width):
    return torch.from_numpy(np.fromfile(path, dtype="<f4")).reshape(-1, width)


def word_array(value):
    return value.contiguous().view(torch.int16).numpy().view(np.uint16)


def ordered(words):
    magnitude = (words & 32767).astype(np.int32)
    return np.where(words & 32768, 32768 - magnitude, 32768 + magnitude)


def compare(expected, actual, low):
    reference, device = expected.flatten().numpy(), actual.flatten().numpy()
    finite = np.isfinite(reference) & np.isfinite(device)
    delta = np.abs(reference.astype(np.float64) - device)
    good = delta[finite]
    wrong = np.flatnonzero(~finite | (delta > 1e-5 + 1e-5 * np.abs(reference)))
    e = word_array(expected.to(low)).flatten()
    a = word_array(actual.to(low)).flatten()
    ulp = np.abs(ordered(e) - ordered(a))
    bit_diff = np.flatnonzero(e != a)
    first = int(wrong[0]) if len(wrong) else None
    first_bit = int(bit_diff[0]) if len(bit_diff) else None
    return (len(reference), len(wrong), int((~finite).sum()), float(good.max(initial=0)),
            float((good / np.maximum(np.abs(reference[finite]), 1e-6)).max(initial=0)),
            first, int(ulp.max(initial=0)), len(bit_diff), first_bit,
            f"{int(e[first_bit]):04x}" if first_bit is not None else "",
            f"{int(a[first_bit]):04x}" if first_bit is not None else "",
            int(ulp[first]) if first is not None else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("dtype", choices=("bf16", "fp16"))
    parser.add_argument("prefill", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    low = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    args.output.mkdir()
    with (args.output / "accuracy.csv").open("w") as stream:
        out = csv.writer(stream)
        out.writerow(("dtype", "prefill", "request", "position", "stage", "reference",
                      "m", "k", "n", "a_stride", "checkpoint_stride", "logical_rhs_stride",
                      "weight_sha256", "activation_sha256", "elements", "mismatches",
                      "nonfinite", "max_abs", "max_rel", "first_mismatch", "max_low_ulp",
                      "bit_differences", "first_bit_difference", "expected_first_bits",
                      "native_first_bits", "first_mismatch_low_ulp"))
        for stage, (source, k, n) in PROJECTIONS.items():
            weights = torch.from_numpy(np.fromfile(args.fixture / f"{stage}.bin", dtype="<i2"))
            weights = weights.view(torch.bfloat16).reshape(n, k).to(low)
            words = word_array(weights)
            weight_sha = hashlib.sha256(words).hexdigest()
            weight64 = weights.double()
            for request in range(2):
                for position in (0, args.prefill):
                    prefix = f"{request}_{position}_"
                    x = fp32_file(args.trace / f"{prefix}{source}.bin", k).to(low)
                    word_array(x).tofile(args.output / f"{prefix}{stage}_input.u16")
                    activation_sha = hashlib.sha256(word_array(x)).hexdigest()
                    native = fp32_file(args.trace / f"{prefix}{stage}_fp32.bin", n)
                    historical = fp32_file(args.trace / f"{prefix}historical_{stage}_fp32.bin", n)
                    name = "output_projection" if stage == "o" else stage
                    original = fp32_file(args.reference / f"{prefix}{name}.bin", n)
                    pytorch = F.linear(x, weights)
                    exact = x.double() @ weight64.T
                    word_array(pytorch).tofile(args.output / f"{prefix}{stage}_matched_torch.u16")
                    exact.numpy().tofile(args.output / f"{prefix}{stage}_fp64.f64")
                    for label, expected, actual in (
                        ("matched_pytorch_low", pytorch.float(), native.to(low).float()),
                        ("original_pytorch_low", original, native.to(low).float()),
                        ("matched_fp64_vs_native_fp32", exact, native),
                        ("matched_fp64_rounded", exact.to(low).float(), native.to(low).float()),
                        ("historical_cpp_fp32", historical, native),
                    ):
                        out.writerow((args.dtype, args.prefill, request, position, stage, label,
                                      x.shape[0], k, n, k, k, n, weight_sha, activation_sha,
                                      *compare(expected, actual, low)))
        for request in range(2):
            for position in (0, args.prefill):
                prefix = f"{request}_{position}_"
                gate = fp32_file(args.trace / f"{prefix}gate.bin", 10944).to(low)
                up = fp32_file(args.trace / f"{prefix}up.bin", 10944).to(low)
                silu = fp32_file(args.trace / f"{prefix}silu.bin", 10944).to(low)
                for stage, expected, actual in (
                    ("silu", F.silu(gate), silu),
                    ("multiply", silu * up, fp32_file(args.trace / f"{prefix}swiglu.bin", 10944)),
                ):
                    out.writerow((args.dtype, args.prefill, request, position, stage,
                                  "matched_pytorch_low", gate.shape[0], 10944, 10944, 10944,
                                  "", "", "", "", *compare(expected.float(), actual.float(), low)))


if __name__ == "__main__":
    main()
