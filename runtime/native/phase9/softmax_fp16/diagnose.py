"""Attribute differences to exponentiation, reduction, normalization or casting."""
import csv
import ctypes
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "softmax"))
from acceptance import native


def denom(exp):
    cols = exp.shape[-1]
    if cols < 8:
        total = exp[:, 0].clone()
        for col in range(1, cols):
            total += exp[:, col]
        return total
    sums = exp[:, :8].clone()
    for col in range(8, cols, 8):
        tail = exp[:, col:col + 8]
        sums[:, :tail.shape[1]] += tail
    sums = sums[:, :4] + sums[:, 4:]
    return (sums[:, 0] + sums[:, 2]) + (sums[:, 1] + sums[:, 3])


def softmax_stages(binary, name, dtype, logits):
    rows, cols = logits.shape
    count = rows * cols
    buf = native(binary, [name, rows, cols, cols, 0], logits.numpy().tobytes())
    return (
        torch.frombuffer(buf, dtype=torch.float32, count=rows),
        torch.frombuffer(buf, dtype=torch.float32, count=rows, offset=4 * rows),
        torch.frombuffer(buf, dtype=torch.float32, count=count, offset=8 * rows).view_as(logits),
        torch.frombuffer(buf, dtype=dtype, count=count, offset=8 * rows + 4 * count).view_as(logits),
    )


def metrics(ref, value):
    valid = ref.isfinite() & value.isfinite()
    difference = (value[valid].double() - ref[valid].double()).abs()
    rel = difference / ref[valid].double().abs().clamp_min(1e-30)
    kind = torch.int32 if ref.element_size() == 4 else torch.int16
    bits = lambda x: x.contiguous().view(kind).long()
    sign = 1 << (ref.element_size() * 8 - 1)
    ordered = lambda b: torch.where(b < 0, -sign - b, b)
    histogram = torch.unique((ordered(bits(ref)[valid]) - ordered(bits(value)[valid])).abs(),
                             return_counts=True)
    ulp, counts = (x.tolist() for x in histogram)
    row_sum = (value.float().sum(-1) - 1).abs()
    nonzero = value[valid & (value != 0)]
    return (float(difference.max()), float(difference.mean()), float(rel.max()),
            float(difference.square().mean().sqrt()), max(ulp),
            " ".join(f"{d}:{n}" for d, n in zip(ulp, counts)), int(value.isnan().sum()),
            int(value.isinf().sum()), float(row_sum.max()), float(value[valid].max()),
            float(nonzero.min()) if nonzero.numel() else 0, int((value == 0).sum()),
            int((difference > 1e-5 + 1e-5 * ref[valid].double().abs()).sum()))


def main():
    root, old, diagnostic, new, reference = map(Path, sys.argv[1:])
    pinned_exp = ctypes.CDLL(str(reference)).phase94_pinned_exp
    pinned_exp.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    pinned_exp.restype = None
    torch.set_num_threads(2)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("dtype", "prefill", "request", "mode", "rows", "cols", "stage", "variant",
                     "max_abs", "mean_abs", "max_rel", "rms", "max_ulp", "ulp_histogram",
                     "nan", "inf", "max_row_sum_error", "max_prob", "min_nonzero", "zero_count",
                     "tolerance_mismatches"))
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        for length in (1, 7, 32, 128):
            for index in range(4):
                with np.load(root / f"{name}-{length}" / f"qk_{index}.npz") as capture:
                    logits = torch.from_numpy(capture["masked"]).view(dtype).float().flatten(0, 1)
                rows, cols = logits.shape
                case = (name, length, index // 2, "decode" if index % 2 else "prefill", rows, cols)
                ref = torch.softmax(logits, -1)
                for variant, binary in (("before", old), ("after", new)):
                    _, _, probs, low = softmax_stages(binary, name, dtype, logits)
                    writer.writerow((*case, "fp32", variant, *metrics(ref, probs)))
                    writer.writerow((*case, "low", variant, *metrics(ref.to(dtype), low)))
                if name != "fp16" or length != 128 or index >= 2:
                    continue
                max_native, sum_native, exp_native, _ = softmax_stages(diagnostic, name, dtype, logits)
                _, _, prob_native, low_native = softmax_stages(old, name, dtype, logits)
                maximum = logits.amax(-1)
                shifted = (logits - maximum[:, None]).contiguous()
                exp_ref = torch.empty_like(shifted)
                pinned_exp(shifted.data_ptr(), exp_ref.data_ptr(), shifted.numel())
                sum_ref = denom(exp_ref)
                fp32_replay = exp_ref * sum_ref.reciprocal()[:, None]
                print(f"{case} max_equal={torch.equal(max_native, maximum)} "
                      f"cpu_exp_replay_bit_diffs={(fp32_replay != ref).sum().item()} "
                      f"native_sum_reconstructed_equal={torch.equal(denom(exp_native), sum_native)}",
                      file=sys.stderr)
                for row, col in torch.nonzero(low_native != ref.half()).tolist():
                    numerator, total = exp_native[row, col], sum_native[row]
                    reciprocal = total.reciprocal()
                    value = prob_native[row, col]
                    fmt = lambda t: f"{t.item():.12g}/0x{t.view(torch.int32).item() & 0xffffffff:08x}"
                    print(f"{case} row={row} key={col} logit={logits[row,col].item()} "
                          f"max={maximum[row].item()} fp32_ref={fmt(ref[row,col])} "
                          f"fp32_mlu={fmt(value)} exponent_ref={fmt(exp_ref[row,col])} "
                          f"exponent_mlu={fmt(numerator)} denominator_ref={fmt(sum_ref[row])} "
                          f"denominator_mlu={fmt(total)} reciprocal_cpu={fmt(reciprocal)} "
                          f"replayed_mlu_norm={fmt(numerator * reciprocal)} "
                          f"expected_half={ref[row,col].half().item():.12g} "
                          f"actual_half={low_native[row,col].item():.12g}", file=sys.stderr)


if __name__ == "__main__":
    main()
