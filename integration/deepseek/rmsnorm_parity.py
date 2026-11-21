"""CPU-only DeepSeek RMSNorm operation-order audit using existing fixtures."""
import argparse
import ast
import csv
import math
import struct
import sys
from pathlib import Path


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def unpack(bits, bf16):
    return (struct.unpack("<f", struct.pack("<I", bits << 16))[0] if bf16
            else struct.unpack("<e", struct.pack("<H", bits))[0])


def round_low(value, bf16):
    if not bf16:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    bits, = struct.unpack("<I", struct.pack("<f", value))
    if bits & 0x7F800000 == 0x7F800000:
        return value
    return unpack((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16, True)


def references(values, gamma, epsilon, bf16):
    """Gamma is assumed to be stored in the same dtype as activations."""
    width = len(gamma)
    model, unrounded, late_cast = [], [], []
    for start in range(0, len(values), width):
        row = values[start:start + width]
        squares = 0.0
        for value in row:
            squares = f32(squares + f32(value * value))
        mean = f32(squares / width)
        inv = f32(1.0 / math.sqrt(f32(mean + epsilon)))
        for value, weight in zip(row, gamma):
            normalized = f32(value * inv)
            product = f32(normalized * weight)
            model.append(round_low(f32(round_low(normalized, bf16) * weight), bf16))
            unrounded.append(product)
            late_cast.append(round_low(product, bf16))
    return model, unrounded, late_cast


def read_norm(path):
    raw = path.read_bytes()
    magic, op, dtype, rows, width, k, epsilon = struct.unpack_from("<8s5If", raw)
    if magic != b"DSV2OP01" or dtype not in (0, 1):
        raise ValueError(path)
    if op != 1:
        return None
    if not rows or not width or width != k or len(raw) != 32 + 2*rows*width + 4*width:
        raise ValueError(path)
    bf16 = bool(dtype)
    words = struct.unpack_from(f"<{rows*width}H", raw, 32)
    gamma = struct.unpack_from(f"<{width}f", raw, 32 + 2*rows*width)
    return [unpack(word, bf16) for word in words], gamma, epsilon, bf16


def metrics(expected, actual):
    errors = [abs(a-b) for a, b in zip(expected, actual)]
    if len(expected) != len(actual) or not errors or not all(map(math.isfinite, errors)):
        raise ValueError("invalid comparison")
    return max(errors), max(e/max(abs(a), 1e-6) for a, e in zip(expected, errors)), sum(errors)/len(errors), sum(e != 0 for e in errors)


def torch_check(source):
    import torch
    import torch.nn as nn

    tree = ast.parse(source.read_text())
    rms = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "DeepseekV2RMSNorm")
    scope = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[rms], type_ignores=[]), str(source), "exec"), scope)
    return torch, scope[rms.name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--torch-source", type=Path)
    args = parser.parse_args()
    torch_reference = torch_check(args.torch_source) if args.torch_source else None
    for bf16 in (False, True):
        half_ulp = 2**(-8 if bf16 else -11)
        assert round_low(1 + half_ulp, bf16) == 1
        assert round_low(1 + 3*half_ulp, bf16) == 1 + 4*half_ulp
        assert math.copysign(1, round_low(-0.0, bf16)) == -1
    output = csv.writer(sys.stdout)
    output.writerow(("case", "comparison", "elements", "max_abs", "max_rel", "mean_abs", "different"))
    cases = 0
    for path in sorted(args.fixtures.glob("*.fixture")):
        fixture = read_norm(path)
        if fixture is None:
            continue
        values, gamma, epsilon, bf16 = fixture
        gamma = [round_low(weight, bf16) for weight in gamma]
        model, unrounded, late_cast = references(values, gamma, epsilon, bf16)
        output.writerow((path.stem, "fp32_post_gamma_proxy", len(model), *metrics(model, unrounded)))
        output.writerow((path.stem, "late_cast_proxy", len(model), *metrics(model, late_cast)))
        if torch_reference:
            torch, cls = torch_reference
            dtype = torch.bfloat16 if bf16 else torch.float16
            layer = cls(len(gamma), eps=epsilon).to(dtype)
            with torch.no_grad():
                layer.weight.copy_(torch.tensor(gamma, dtype=dtype))
                actual = layer(torch.tensor(values, dtype=dtype).reshape(-1, len(gamma)))
            assert actual.dtype == dtype
            result = metrics(model, actual.float().flatten().tolist())
            output.writerow((path.stem, "checkpoint_source_cpu", len(model), *result))
            if result[3]:
                raise AssertionError(f"CPU source differential: {path.name}: {result}")
        cases += 1
    if cases != 10:
        raise ValueError(f"expected ten RMSNorm fixtures, found {cases}")


if __name__ == "__main__":
    main()
