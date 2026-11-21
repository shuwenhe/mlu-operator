"""Export only the actual BF16 slices exercised by the runtime test."""
import hashlib
import json
import struct
import sys
from pathlib import Path

from fixtures import load_headers

def main():
    root, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    tensors = load_headers(root)
    config = json.loads((root / "config.json").read_text())
    if (config["model_type"], config["hidden_size"], config["kv_lora_rank"]) != ("deepseek_v2", 2048, 512):
        raise ValueError("unexpected checkpoint")
    fingerprint = hashlib.sha256((root / "model.safetensors.index.json").read_bytes()).hexdigest()
    aliases = {
        "q": "model.layers.0.self_attn.q_proj.weight",
        "kv_a": "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "kv_b": "model.layers.0.self_attn.kv_b_proj.weight",
        "gamma": "model.layers.0.self_attn.kv_a_layernorm.weight",
        "input_gamma": "model.layers.0.input_layernorm.weight",
        "post_gamma": "model.layers.0.post_attention_layernorm.weight",
        "o": "model.layers.0.self_attn.o_proj.weight",
        "gate": "model.layers.0.mlp.gate_proj.weight",
        "up": "model.layers.0.mlp.up_proj.weight",
        "down": "model.layers.0.mlp.down_proj.weight",
    }
    with (output / "manifest").open("w") as manifest:
        manifest.write(f"{fingerprint}\n")
        for alias, name in aliases.items():
            shard, offset, shape = tensors[name]
            with shard.open("rb") as src:
                src.seek(offset)
                count = 2 * (shape[0] if len(shape) == 1 else shape[0]*shape[1])
                raw = src.read(count)
            if len(raw) != count:
                raise EOFError(name)
            (output / f"{alias}.bin").write_bytes(raw)
            manifest.write(f"{alias} {name} {shard.name} {offset}\n")
            print(alias, name, shape, shard.name, offset)
    shard, offset, shape = tensors["model.embed_tokens.weight"]
    if shape != [102400, 2048]:
        raise ValueError(shape)
    with shard.open("rb") as src, (output / "tokens.bin").open("wb") as dst:
        for token in (100000, 42, 100, 777, 512, 16):
            src.seek(offset + token*4096)
            raw = src.read(4096)
            if len(raw) != 4096:
                raise EOFError(token)
            dst.write(struct.pack("<I", token))
            dst.write(raw)

if __name__ == "__main__":
    main()
