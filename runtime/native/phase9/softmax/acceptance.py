"""Compare native Softmax with eager PyTorch on native-QK score bits."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def native(binary, args, data):
    return bytearray(subprocess.run([str(binary), *map(str, args)], input=data,
                                    stdout=subprocess.PIPE, check=True).stdout)


def measure(ref, actual):
    if ref.dtype == torch.float32:
        bit_dtype = torch.int32
    else:
        bit_dtype = torch.int16
    rb, ab = (t.contiguous().view(bit_dtype).long() for t in (ref, actual))
    sign = 1 << (ref.element_size() * 8 - 1)
    ordered = lambda x: torch.where(x < 0, -sign - x, x)
    finite = ref.isfinite() & actual.isfinite()
    err = (ref[finite].double() - actual[finite].double()).abs()
    mag = ref[finite].double().abs()
    invalid = ((ref.isnan() != actual.isnan()) | (ref.isinf() != actual.isinf())
               | (ref.isinf() & actual.isinf() & (ref != actual)))
    max_or_zero = lambda x: x.max().item() if x.numel() else 0
    return (int(((rb != ab) & ~(ref.isnan() & actual.isnan())).sum()),
            int((err > 1e-5 + 1e-5 * mag).sum() + invalid.sum()),
            max_or_zero(err), max_or_zero(err / mag.clamp_min(1e-30)),
            max_or_zero((ordered(rb[finite]) - ordered(ab[finite])).abs()),
            int(actual.isnan().sum()), int(actual.isinf().sum()), int(invalid.sum()))


def evaluate(binary, writer, case, dtype, scores, mask=None, valid=None):
    rows, cols = scores.shape
    valid = cols if valid is None else valid
    mask_rows = 0 if mask is None else mask.shape[0]
    data = scores.contiguous().numpy().tobytes()
    if mask_rows:
        data += mask.contiguous().view(torch.int16).numpy().tobytes()
    result = native(binary, [case[0], rows, cols, valid, mask_rows], data)
    if len(result) != 8 * rows + 6 * rows * cols:
        raise ValueError(f"Malformed payload for {case}")
    maxes = torch.frombuffer(result, dtype=torch.float32, count=rows)
    denom = torch.frombuffer(result, dtype=torch.float32, count=rows, offset=4 * rows)
    probs = torch.frombuffer(result, dtype=torch.float32, count=rows * cols,
                             offset=8 * rows).view_as(scores)
    low = torch.frombuffer(result, dtype=dtype, count=rows * cols,
                           offset=8 * rows + 4 * rows * cols).view_as(scores)
    ref_scores = scores.clone()
    if mask_rows:
        ref_scores = (ref_scores.to(dtype) + mask.repeat(rows // mask_rows, 1)).float()
    ref_scores[:, valid:] = -torch.inf
    expected = torch.softmax(ref_scores, dim=-1, dtype=torch.float32)
    max_ref = ref_scores.amax(-1)
    denom_ref = (ref_scores - max_ref[:, None]).exp().sum(-1)
    failures = 0
    for stage, ref, actual in (("maximum", max_ref, maxes), ("denominator_diagnostic", denom_ref, denom),
                               ("fp32", expected, probs), ("low", expected.to(dtype), low)):
        stats = measure(ref, actual)
        sums = actual.float().sum(-1) if stage in {"fp32", "low"} else torch.empty(0)
        sums = sums[sums.isfinite()]
        row_error = (sums - 1).abs().max().item() if sums.numel() else 0
        writer.writerow((*case, rows, cols, valid, stage, ref.numel(), *stats, row_error))
        failures += stats[1] if stage != "denominator_diagnostic" else 0
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("capture_root", "qk", "softmax"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(f"PyTorch {torch.__version__} CPU capability {torch.backends.cpu.get_cpu_capability()}",
          file=sys.stderr)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("dtype", "prefill", "request", "pass", "rows", "cols", "valid", "stage",
                     "elements", "bit_mismatches", "tolerance_mismatches", "max_abs", "max_rel",
                     "max_ulp", "nan", "inf", "unexpected_nonfinite", "max_row_sum_error"))
    fixtures = 0
    failures = 0
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        for length in (1, 7, 32, 128):
            for path in sorted((args.capture_root / f"{name}-{length}").glob("qk_*.npz")):
                index = int(path.stem.split("_")[1])
                with np.load(path) as f:
                    q, k, positions = f["q"], f["k"], f["positions"]
                    first = int(positions[0])
                    scale, expected_bits = float(f["scale"]), f["masked"]
                seq, keys = q.shape[1], k.shape[1]
                result = native(args.qk, [0, name, seq, keys, first, scale],
                                q.tobytes() + k.tobytes() +
                                (positions - first).astype(np.uint32).tobytes())
                scores = torch.frombuffer(result, dtype=torch.float32, count=16 * seq * keys,
                                          offset=6 * 16 * seq * keys).view(16 * seq, keys)
                expected_scores = torch.from_numpy(expected_bits).view(dtype).float().view_as(scores)
                if not torch.equal(scores.view(torch.int32), expected_scores.view(torch.int32)):
                    raise ValueError(f"Frozen QK replay differs from checkpoint in {path}")
                case = (name, length, index // 2, "decode" if first else "prefill")
                failures += evaluate(args.softmax, writer, case, dtype, scores)
                fixtures += 1

        # Broadcast additive bias, an all-masked row, a causal -inf, and a padding tail.
        scores = torch.zeros((4, 131), dtype=torch.float32)
        scores[:, 0] = 2.5
        scores[:, 9] = -torch.inf
        mask = torch.zeros(2, 131, dtype=dtype)
        mask[0, 1] = -0.875
        mask[0, 2] = -torch.inf
        mask[1] = -torch.inf
        failures += evaluate(args.softmax, writer, (name, 0, 0, "mask_unit"),
                             dtype, scores, mask, valid=129)
        failures += evaluate(args.softmax, writer, (name, 0, 0, "empty_unit"),
                             dtype, scores, valid=0)
    print(f"checkpoint_fixtures={fixtures} acceptance_mismatches={failures}", file=sys.stderr)
    return int(fixtures != 32 or failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
