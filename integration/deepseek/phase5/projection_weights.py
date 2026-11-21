import argparse
import struct
from pathlib import Path
from fixtures import load_headers

PROJECTIONS = {
    "q": ("q_proj", 3072, 2048),
    "kv_a": ("kv_a_proj_with_mqa", 576, 2048),
    "kv_b": ("kv_b_proj", 4096, 512),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    weights = load_headers(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    for label, (suffix, n, k) in PROJECTIONS.items():
        name = f"model.layers.0.self_attn.{suffix}.weight"
        path, offset, shape = weights[name]
        if shape != [n, k]:
            raise ValueError((name, shape))
        with path.open("rb") as src, (args.output / (label + ".bin")).open("wb") as dst:
            dst.write(struct.pack("<II", n, k))
            src.seek(offset)
            remaining = 2*n*k
            while remaining:
                chunk = src.read(min(remaining, 1 << 20))
                if not chunk:
                    raise EOFError(name)
                dst.write(chunk)
                remaining -= len(chunk)
        print(f"{name} BF16 {shape}")

    gamma = "model.layers.0.self_attn.kv_a_layernorm.weight"
    path, offset, shape = weights[gamma]
    if shape != [512]:
        raise ValueError((gamma, shape))
    with path.open("rb") as src:
        src.seek(offset)
        data = src.read(1024)
    if len(data) != 1024:
        raise EOFError(gamma)
    (args.output / "kv_gamma.bin").write_bytes(data)
    print(f"{gamma} BF16 {shape}")

if __name__ == "__main__":
    main()
