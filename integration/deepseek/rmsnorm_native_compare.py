"""Compare native diagnostic dumps with independently computed CPU values."""
import argparse
import csv
import math
import struct
import sys
from pathlib import Path

from rmsnorm_parity import f32, read_norm, references, round_low, unpack


def expected_stages(values, gamma, epsilon, bf16):
    normalized = []
    width = len(gamma)
    for start in range(0, len(values), width):
        row = values[start:start + width]
        squares = 0.0
        for value in row:
            squares = f32(squares + f32(value * value))
        inv = f32(1.0 / math.sqrt(f32(f32(squares / width) + epsilon)))
        normalized.extend(f32(value * inv) for value in row)
    rounded = [round_low(value, bf16) for value in normalized]
    low_gamma = [round_low(value, bf16) for value in gamma]
    final = [round_low(f32(value * low_gamma[i % width]), bf16)
             for i, value in enumerate(rounded)]
    cnnl_contract = [f32(value * gamma[i % width])
                     for i, value in enumerate(normalized)]
    return normalized, rounded, final, cnnl_contract


def decode(data, count, bf16):
    if len(data) != 12 * count:
        raise ValueError("invalid diagnostic dump size")
    normalized = struct.unpack_from(f"<{count}f", data)
    rounded = [unpack(bits, bf16) for bits in struct.unpack_from(f"<{count}H", data, 4*count)]
    final = [unpack(bits, bf16) for bits in struct.unpack_from(f"<{count}H", data, 6*count)]
    cnnl = struct.unpack_from(f"<{count}f", data, 8*count)
    return normalized, rounded, final, cnnl


def compare(expected, actual, exact):
    errors, nan, inf, failures = [], 0, 0, 0
    for want, got in zip(expected, actual):
        if not math.isfinite(got):
            nan += math.isnan(got)
            inf += math.isinf(got)
            continue
        error = abs(want - got)
        errors.append((error, error / max(abs(want), 1e-6)))
        if exact:
            failures += struct.pack("<f", want) != struct.pack("<f", got)
        else:
            failures += error > 1e-5 + 1e-5 * abs(want)
    return (max((item[0] for item in errors), default=math.nan),
            max((item[1] for item in errors), default=math.nan),
            sum(item[0] for item in errors)/len(errors) if errors else math.nan,
            nan, inf, failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-self-check", action="store_true")
    mode.add_argument("--actual-dir", type=Path)
    args = parser.parse_args()
    failures, elements, cases = 0, 0, 0
    writer = csv.writer(sys.stdout)
    if args.actual_dir:
        writer.writerow(("fixture", "dtype", "stage", "elements", "max_abs", "max_rel",
                         "mean_abs", "nan", "inf", "mismatches"))
    for path in sorted(args.fixtures.glob("*.fixture")):
        fixture = read_norm(path)
        if fixture is None:
            continue
        values, gamma, epsilon, bf16 = fixture
        expected = expected_stages(*fixture)
        if args.cpu_self_check:
            model, _, _ = references(values, [round_low(g, bf16) for g in gamma], epsilon, bf16)
            assert compare(model, expected[2], True)[-3:] == (0, 0, 0)
        else:
            actual = decode((args.actual_dir / (path.name + ".actual")).read_bytes(), len(values), bf16)
            pairs = [
                ("bang_normalized", expected[0], actual[0], False, True),
                ("bang_pre_gamma", expected[1], actual[1], True, True),
                ("bang_final", expected[2], actual[2], True, True),
                ("cnnl_fp32_contract", expected[3], actual[3], False, True),
                ("cnnl_vs_deepseek", expected[2], actual[3], True, False),
            ]
            for stage, want, got, exact, gate in pairs:
                result = compare(want, got, exact)
                writer.writerow((path.stem, "bf16" if bf16 else "fp16", stage, len(values), *result))
                if gate:
                    failures += sum(result[-3:])
        elements += len(values)
        cases += 1
    if cases != 10:
        raise ValueError(f"expected 10 RMSNorm fixtures, got {cases}")
    if args.cpu_self_check:
        print(f"CPU stage reference: {cases} fixtures, {elements} final outputs agree")
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
