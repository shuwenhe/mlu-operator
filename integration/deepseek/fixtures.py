"""Read local safetensors headers and emit small operator fixtures. No model imports."""
import argparse
import json
import math
import struct
from pathlib import Path


def load_headers(root):
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for shard in sorted(set(index.values())):
        path = root / shard
        with path.open("rb") as stream:
            length, = struct.unpack("<Q", stream.read(8))
            if not 0 < length <= 100_000_000:
                raise ValueError(f"invalid header length in {shard}")
            header = json.loads(stream.read(length))
        header.pop("__metadata__", None)
        if set(header) != {name for name, owner in index.items() if owner == shard}:
            raise ValueError(f"index mismatch in {shard}")
        if max(item["data_offsets"][1] for item in header.values()) + 8 + length != path.stat().st_size:
            raise ValueError(f"file size mismatch in {shard}")
        for name, item in header.items():
            start, end = item["data_offsets"]
            if item["dtype"] != "BF16" or start < 0 or end - start != 2 * math.prod(item["shape"]):
                raise ValueError(f"unexpected tensor encoding for {name}")
            tensors[name] = (path, 8 + length + start, item["shape"])
    return tensors


def slice_bf16(tensors, name, count):
    path, offset, shape = tensors[name]
    if count > math.prod(shape):
        raise ValueError(name)
    with path.open("rb") as stream:
        stream.seek(offset)
        return struct.unpack(f"<{count}H", stream.read(2 * count))


def bf16_float(value):
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def encode(values, dtype):
    if dtype == 0:
        return struct.pack(f"<{len(values)}e", *values)
    words = []
    for value in values:
        bits, = struct.unpack("<I", struct.pack("<f", value))
        words.append((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16)
    return struct.pack(f"<{len(words)}H", *words)


def write_case(out, name, op, dtype, m, n, k, epsilon, a, payload):
    with (out / f"{name}.fixture").open("wb") as stream:
        stream.write(struct.pack("<8s5If", b"DSV2OP01", op, dtype, m, n, k, epsilon))
        stream.write(encode(a, dtype))
        stream.write(payload)


def activation(count, salt):
    return [((i * 17 + salt) % 31 - 15) / 32 for i in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tensors = load_headers(args.model)
    config = json.loads((args.model / "config.json").read_text())
    if config["architectures"] != ["DeepseekV2ForCausalLM"]:
        raise ValueError("unexpected model architecture")
    args.output.mkdir(parents=True, exist_ok=True)
    for dtype, tag in enumerate(("fp16", "bf16")):
        cases = [
            ("q", "model.layers.0.self_attn.q_proj.weight", 3, 17),
            ("kv_a", "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", 1, 129),
            ("kv_b", "model.layers.0.self_attn.kv_b_proj.weight", 2, 33),
            ("expert_up", "model.layers.1.mlp.experts.0.up_proj.weight", 3, 65),
            ("logits", "lm_head.weight", 1, 17),
        ]
        for name, tensor, m, n in cases:
            outputs, k = tensors[tensor][2]
            if n > outputs:
                raise ValueError(tensor)
            weights = slice_bf16(tensors, tensor, n*k)
            # Checkpoint [N,K] to operator [K,N].
            b = [bf16_float(weights[j*k+i]) for i in range(k) for j in range(n)]
            write_case(args.output, f"{tag}_{name}", 0, dtype, m, n, k, 0,
                       activation(m*k, 7), encode(b, dtype))
        for name, tensor in [
            ("input_norm", "model.layers.0.input_layernorm.weight"),
            ("kv_norm", "model.layers.0.self_attn.kv_a_layernorm.weight"),
            ("final_norm", "model.norm.weight"),
        ]:
            k, = tensors[tensor][2]
            gamma = [bf16_float(bits) for bits in slice_bf16(tensors, tensor, k)]
            write_case(args.output, f"{tag}_{name}", 1, dtype, 3, k, k,
                       config["rms_norm_eps"], [0.0]*k + activation(2*k, 11),
                       struct.pack(f"<{k}f", *gamma))
        for m, n, k in [(1, 1, 1), (3, 129, 65), (16, 16, 16)]:
            write_case(args.output, f"{tag}_synthetic_{m}_{n}_{k}", 0, dtype, m, n, k, 0,
                       activation(m*k, 9), encode(activation(k*n, 3), dtype))
        for k in (1, 65):
            write_case(args.output, f"{tag}_synthetic_norm_{k}", 1, dtype, 2, k, k,
                       config["rms_norm_eps"], [0.0]*k + activation(k, 4),
                       struct.pack(f"<{k}f", *([1.0]*k)))
    print(f"verified {len(tensors)} tensor headers and generated 26 fixtures")


if __name__ == "__main__":
    main()
