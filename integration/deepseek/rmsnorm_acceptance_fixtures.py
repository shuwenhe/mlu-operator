"""Add acceptance-only odd widths without altering the original fixtures."""
import argparse
import shutil
import struct
from pathlib import Path

from fixtures import activation, write_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for dtype, tag in enumerate(("fp16", "bf16")):
        for name in ("kv_norm", "input_norm", "synthetic_norm_65"):
            file = args.original / f"{tag}_{name}.fixture"
            shutil.copyfile(file, args.output / file.name)
        for width in (127, 193):
            # Two adjacent, distinct rows plus a zero row.
            gamma = [0.5 + (i % 11) / 16 for i in range(width)]
            values = activation(width, 7) + [0.0]*width + activation(width, 13)
            write_case(args.output, f"{tag}_synthetic_norm_{width}", 1, dtype,
                       3, width, width, 1e-6, values, struct.pack(f"<{width}f", *gamma))


if __name__ == "__main__":
    main()
