"""Validate all DeepSeek-V2-Lite checkpoint tensors from their on-disk headers."""

import json
from pathlib import Path
import struct
import sys


def audit(root):
    config = json.loads((root / "config.json").read_text())
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    expected = {
        "model.embed_tokens.weight": (102400, 2048),
        "model.norm.weight": (2048,),
        "lm_head.weight": (102400, 2048),
    }

    def matrix(prefix, width):
        expected[prefix + "gate_proj.weight"] = (width, 2048)
        expected[prefix + "up_proj.weight"] = (width, 2048)
        expected[prefix + "down_proj.weight"] = (2048, width)

    dimensions = {
        "hidden_size": 2048, "num_hidden_layers": 27, "vocab_size": 102400,
        "num_attention_heads": 16, "num_key_value_heads": 16,
        "q_lora_rank": None, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64, "v_head_dim": 128, "first_k_dense_replace": 1,
        "moe_layer_freq": 1, "intermediate_size": 10944, "moe_intermediate_size": 1408,
        "n_routed_experts": 64, "n_shared_experts": 2, "num_experts_per_tok": 6,
        "tie_word_embeddings": False,
    }
    for key, value in dimensions.items():
        if config[key] != value:
            raise ValueError(f"Unexpected {key}: {config[key]!r}")
    for i in range(27):
        prefix = f"model.layers.{i}."
        for name, shape in {
            "input_layernorm.weight": (2048,), "post_attention_layernorm.weight": (2048,),
            "self_attn.q_proj.weight": (3072, 2048),
            "self_attn.kv_a_proj_with_mqa.weight": (576, 2048),
            "self_attn.kv_a_layernorm.weight": (512,),
            "self_attn.kv_b_proj.weight": (4096, 512),
            "self_attn.o_proj.weight": (2048, 2048),
        }.items():
            expected[prefix + name] = shape
        if i == 0:
            matrix(prefix + "mlp.", 10944)
        else:
            expected[prefix + "mlp.gate.weight"] = (64, 2048)
            matrix(prefix + "mlp.shared_experts.", 2816)
            for expert in range(64):
                matrix(prefix + f"mlp.experts.{expert}.", 1408)
    if expected.keys() != weight_map.keys():
        raise ValueError(f"Checkpoint tensor names differ: {expected.keys() ^ weight_map.keys()}")
    found = set()
    for name in sorted(set(weight_map.values())):
        with (root / name).open("rb") as shard:
            header_size, = struct.unpack("<Q", shard.read(8))
            header = json.loads(shard.read(header_size))
        data_bytes = (root / name).stat().st_size - 8 - header_size
        for tensor, item in header.items():
            if tensor == "__metadata__":
                continue
            begin, end = item["data_offsets"]
            from math import prod
            if (
                tensor in found or weight_map.get(tensor) != name
                or tuple(item["shape"]) != expected[tensor] or item["dtype"] != "BF16"
                or not 0 <= begin < end <= data_bytes or end - begin != prod(expected[tensor]) * 2
            ):
                raise ValueError(f"Invalid tensor header: {name}/{tensor}")
            found.add(tensor)
    if found != expected.keys():
        raise ValueError("One or more indexed tensors are missing from the shard headers")
    print(f"PASS: {len(found)} BF16 tensor headers, {len(set(weight_map.values()))} shards, 27 layers")
    print(f"checkpoint architecture={config['architectures']} rotary={config['rope_scaling']}")


if __name__ == "__main__":
    audit(Path(sys.argv[1]))
