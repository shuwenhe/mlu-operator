import ast
import json
import math
import struct
import sys
from pathlib import Path
import torch
import torch.nn as nn

source, config, out = map(Path, sys.argv[1:])
out.mkdir(exist_ok=True)
cfg = json.loads(config.read_text())
tree = ast.parse(source.read_text())
names = {"DeepseekV2RotaryEmbedding", "DeepseekV2YarnRotaryEmbedding",
         "rotate_half", "apply_rotary_pos_emb"}
nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
         and (node.name in names or node.name.startswith("yarn_"))]
scope = dict(torch=torch, nn=nn, math=math)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
params = dict(cfg["rope_scaling"])
del params["type"]
params["scaling_factor"] = params.pop("factor")
d = cfg["qk_rope_head_dim"]
assert d == 64 and cfg["num_attention_heads"] == 16
positions = [
    [0],
    [4093, 4094, 4095, 4096, 4097, 4100, 4096],
    [163839-i*17 for i in range(65)],
]
table_ids = sorted({position for items in positions for position in items})
table_index = {position: i for i, position in enumerate(table_ids)}
rotary = scope["DeepseekV2YarnRotaryEmbedding"](d, max_position_embeddings=cfg["max_position_embeddings"],
                                                base=cfg["rope_theta"], **params)
rng = torch.Generator().manual_seed(20261121)
for dtype, tag in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
    for ids in positions:
        s = len(ids)
        q = torch.randn((1,16,s,d), generator=rng).to(dtype)
        k = torch.randn((1,1,s,d), generator=rng).to(dtype)
        cos, sin = rotary(q, seq_len=cfg["max_position_embeddings"])
        qr, kr = scope["apply_rotary_pos_emb"](q, k, cos, sin, torch.tensor([ids]))
        with (out/f"{tag}_{s}.rope").open("wb") as stream:
            stream.write(struct.pack("<II", s, len(table_ids)))
            stream.write(struct.pack(f"<{s}I", *(table_index[p] for p in ids)))
            for x in (cos[table_ids], sin[table_ids], torch.cat((q,k),1), torch.cat((qr,kr),1)):
                values = x.contiguous().view(torch.int16).flatten().tolist()
                stream.write(struct.pack(f"<{len(values)}h", *values))
        print(tag,s,min(ids),max(ids),len(table_ids))
