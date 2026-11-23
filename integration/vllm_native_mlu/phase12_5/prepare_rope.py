"""Generate DeepSeek-V2-Lite rope fixtures and softmax scale from checkpoint config."""
import ast
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn


def yarn_get_mscale(scale: float, mscale: float) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def main(checkpoint: Path, out: Path, positions: int = 131) -> int:
    source = checkpoint / "modeling_deepseek.py"
    cfg = json.loads((checkpoint / "config.json").read_text())
    tree = ast.parse(source.read_text())
    needed = {
        "DeepseekV2RotaryEmbedding",
        "DeepseekV2YarnRotaryEmbedding",
        "rotate_half",
        "apply_rotary_pos_emb",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and (node.name in needed or node.name.startswith("yarn_"))
    ]
    scope = {"torch": torch, "nn": nn, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    params = dict(cfg["rope_scaling"])
    params.pop("type", None)
    params["scaling_factor"] = params.pop("factor")
    rotary = scope["DeepseekV2YarnRotaryEmbedding"](
        cfg["qk_rope_head_dim"],
        max_position_embeddings=cfg["max_position_embeddings"],
        base=cfg["rope_theta"],
        **params,
    )
    q = torch.zeros((1, 16, 1, cfg["qk_rope_head_dim"]), dtype=torch.bfloat16)
    cos, sin = rotary(q, seq_len=positions)
    out.mkdir(parents=True, exist_ok=True)
    cos[:positions].contiguous().view(torch.int16).cpu().numpy().tofile(out / "cos.bin")
    sin[:positions].contiguous().view(torch.int16).cpu().numpy().tofile(out / "sin.bin")
    scale = (
        1.0
        / math.sqrt(cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"])
        * yarn_get_mscale(cfg["rope_scaling"]["factor"], cfg["rope_scaling"].get("mscale_all_dim", 0.0))
        ** 2
    )
    (out / "scale.txt").write_text(repr(float(scale)) + "\n")
    print(json.dumps({"rope": str(out), "scale": float(scale)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
