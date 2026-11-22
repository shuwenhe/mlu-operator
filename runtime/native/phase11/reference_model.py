"""Run the original checkpoint decoder and cache arithmetic under independent CPU PyTorch."""
import ast
import math
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audit import load_headers
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"p0"))
from checkpoint_trace import names


class Cache:
    def __init__(self):
        self.keys = {}
        self.values = {}

    def get_usable_length(self, _new, layer):
        return self.keys[layer].shape[-2] if layer in self.keys else 0

    def update(self, key, value, layer, _kwargs):
        if layer in self.keys:
            key = torch.cat((self.keys[layer],key),dim=-2)
            value = torch.cat((self.values[layer],value),dim=-2)
        self.keys[layer],self.values[layer] = key,value
        return key,value


def write(path, value):
    if value.dtype == torch.bfloat16:
        value = value.contiguous().view(torch.int16)
    value.detach().contiguous().numpy().tofile(path)


def main(model,prompt,steps,dump=None):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = json.loads((model/"config.json").read_text())
    cfg["_attn_implementation"] = "eager"
    cfg = SimpleNamespace(**cfg)
    tensors = load_headers(model)

    def weight(name):
        shard, offset, shape = tensors[name]
        words = np.memmap(shard, mode="c", offset=offset, dtype="<u2", shape=tuple(shape))
        return torch.from_numpy(words).view(torch.bfloat16)

    local = model/"modeling_deepseek.py"
    keep = names | {"DeepseekV2MoE","MoEGate","AddAuxiliaryLoss"}
    nodes = [n for n in ast.parse(local.read_text()).body
             if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and
             (n.name in keep or n.name.startswith("yarn_"))]
    ctx = dict(torch=torch,nn=nn,F=F,math=math,warnings=warnings,Optional=Optional,Tuple=Tuple,
               np=np,DeepseekV2Config=SimpleNamespace,Cache=Cache,
               ACT2FN={"silu":F.silu},ATTENTION_CLASSES={})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(local),"exec"),ctx)
    ctx["ATTENTION_CLASSES"]["eager"] = ctx["DeepseekV2Attention"]
    layers = []
    attn = {}
    for index in range(cfg.num_hidden_layers):
        with torch.device("meta"):
            layer = ctx["DeepseekV2DecoderLayer"](cfg,index)
        prefix = f"model.layers.{index}."
        layer.load_state_dict({name:weight(prefix+name) for name in layer.state_dict()},assign=True)
        layer.eval()
        if dump:
            layer.self_attn.register_forward_hook(
                lambda _m,_args,out,i=index: attn.update({i:out[0]}))
        layers.append(layer)
    embedding,head = weight("model.embed_tokens.weight"),weight("lm_head.weight")
    norm = ctx["DeepseekV2RMSNorm"](cfg.hidden_size,eps=cfg.rms_norm_eps).eval()
    norm.load_state_dict({"weight":weight("model.norm.weight")},assign=True)
    ids = torch.from_numpy(np.fromfile(prompt,dtype="<u4").astype(np.int64))
    cache = Cache()
    cursor = 0
    if dump:
        dump.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for step in range(steps):
            start = time.perf_counter()
            count = ids.numel()
            x = F.embedding(ids,embedding)[None]
            prefix = dump / f"{step}-" if dump else None
            if dump:
                write(str(prefix)+"embedding.bin",x)
            pos = torch.arange(cursor,cursor+count)[None]
            mask = torch.zeros((1,1,count,cursor+count),dtype=torch.bfloat16)
            mask.masked_fill_(torch.arange(cursor+count)[None,:]>pos.T,-torch.inf)
            for index, layer in enumerate(layers):
                x = layer(x,attention_mask=mask,position_ids=pos,past_key_value=cache,use_cache=True)[0]
                if dump:
                    path = str(prefix)+f"layer{index}"
                    write(path+".bin",x)
                    write(path+"-attention.bin",attn[index])
                    write(path+"-key.bin",cache.keys[index])
                    write(path+"-value.bin",cache.values[index])
            logits = F.linear(norm(x)[:,-1,:],head).float()
            if dump:
                write(str(prefix)+"logits.bin",logits)
            chosen = logits.argmax(-1).item()
            print(f"generated {step} {chosen} {(time.perf_counter()-start)*1000:.3f}",flush=True)
            cursor += count
            if chosen==cfg.eos_token_id:
                break
            ids = torch.tensor([chosen],dtype=torch.int64)


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]),
         Path(sys.argv[4]) if len(sys.argv)==5 else None)
