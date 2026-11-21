"""Export exact Layer 0 tensor bytes from a local DeepSeek-V2-Lite checkpoint."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures import load_headers

TENSORS = {
    "q": ("self_attn.q_proj.weight", [3072, 2048]),
    "kv_a": ("self_attn.kv_a_proj_with_mqa.weight", [576, 2048]),
    "kv_b": ("self_attn.kv_b_proj.weight", [4096, 512]),
    "o": ("self_attn.o_proj.weight", [2048, 2048]),
    "gate": ("mlp.gate_proj.weight", [10944, 2048]),
    "up": ("mlp.up_proj.weight", [10944, 2048]),
    "input_gamma": ("input_layernorm.weight", [2048]),
    "kv_gamma": ("self_attn.kv_a_layernorm.weight", [512]),
    "post_gamma": ("post_attention_layernorm.weight", [2048]),
}

def main(root, output):
    config_bytes = (root / "config.json").read_bytes()
    config = json.loads(config_bytes)
    expected = {
        "model_type": "deepseek_v2", "hidden_size": 2048, "kv_lora_rank": 512,
        "q_lora_rank": None, "num_attention_heads": 16, "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64, "v_head_dim": 128, "intermediate_size": 10944,
        "first_k_dense_replace": 1, "rms_norm_eps": 1e-6, "rope_theta": 10000,
        "torch_dtype": "bfloat16",
        "rope_scaling": {"type": "yarn", "factor": 40, "beta_fast": 32, "beta_slow": 1,
                         "original_max_position_embeddings": 4096, "mscale": 0.707,
                         "mscale_all_dim": 0.707},
    }
    if any(key not in config or config[key] != value for key, value in expected.items()):
        raise ValueError("checkpoint does not match the adapter contract")

    headers = load_headers(root)
    digest = hashlib.sha256(config_bytes)
    digest.update((root / "model.safetensors.index.json").read_bytes())
    output.mkdir(parents=True)
    entries = []
    for alias, (suffix, shape) in TENSORS.items():
        name = "model.layers.0." + suffix
        shard, offset, actual = headers[name]
        if actual != shape:
            raise ValueError(f"unexpected shape for {name}: {actual}")
        elements = shape[0] * (shape[1] if len(shape) == 2 else 1)
        with shard.open("rb") as source:
            source.seek(offset)
            raw = source.read(2 * elements)
        if len(raw) != 2 * elements:
            raise EOFError(name)
        digest.update(raw)
        (output / f"{alias}.bin").write_bytes(raw)
        entries.append(f"{alias} {name} {shard.name} {offset}")
        print(alias, name, shard.name, offset, shape, "BF16")

    shard, offset, shape = headers["model.embed_tokens.weight"]
    if shape != [102400, 2048]:
        raise ValueError(f"unexpected embedding shape: {shape}")
    with shard.open("rb") as source:
        source.seek(offset + 100000 * 4096)
        token = source.read(4096)
    if len(token) != 4096:
        raise EOFError("embedding token 100000")
    (output / "token.bin").write_bytes(token)
    manifest = output / "manifest.pending"
    manifest.write_text(digest.hexdigest() + "\n" + "\n".join(entries) + "\n")
    manifest.replace(output / "manifest")

if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
