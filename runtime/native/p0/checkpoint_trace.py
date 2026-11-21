"""CPU original-source layer trace. No model serving or MLU imports."""
import argparse
import ast
import json
import math
import struct
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

names = {"DeepseekV2MLP","DeepseekV2RMSNorm","DeepseekV2Attention","DeepseekV2DecoderLayer",
         "DeepseekV2RotaryEmbedding","DeepseekV2YarnRotaryEmbedding","apply_rotary_pos_emb","rotate_half"}
aliases = {
    "q": "self_attn.q_proj.weight", "kv_a": "self_attn.kv_a_proj_with_mqa.weight",
    "kv_b": "self_attn.kv_b_proj.weight", "gamma": "self_attn.kv_a_layernorm.weight",
    "input_gamma": "input_layernorm.weight", "post_gamma": "post_attention_layernorm.weight",
    "o": "self_attn.o_proj.weight", "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight", "down": "mlp.down_proj.weight",
}

class Cache:
    """The eager source's expanded K/V append contract."""
    def __init__(self):
        self.k = self.v = None
    def get_usable_length(self, _new, index):
        assert index == 0
        return 0 if self.k is None else self.k.shape[-2]
    def update(self, k, v, index, _kwargs):
        assert index == 0
        self.k = k if self.k is None else torch.cat((self.k,k),dim=-2)
        self.v = v if self.v is None else torch.cat((self.v,v),dim=-2)
        return self.k,self.v

def micro():
    # Exact input around BF16 midpoints, selected to distinguish cast ordering.
    x=torch.tensor([1.,1.,-1.,0.5,-0.5,1.0078125]).bfloat16()
    y=torch.tensor([0.00390625,0.01171875,-0.00390625,0.12475586,-0.12475586,0.0039])
    stages={
        "A_f32_f32":(x.float()+y).bfloat16(),
        "B_bf16_bf16":x+y.bfloat16(),
        "C_projection_cast_first":(x.float()+y.bfloat16().float()).bfloat16(),
        "D_f32_projection_add_then_cast":(y+x.float()).bfloat16(),
        "E_source":x+y.bfloat16(),
    }
    for label,values in stages.items():
        print("micro",label," ".join(f"{v&65535:04x}" for v in values.view(torch.int16).tolist()))
    assert torch.equal(stages["B_bf16_bf16"],stages["C_projection_cast_first"])
    assert torch.equal(stages["C_projection_cast_first"],stages["E_source"])
    assert not torch.equal(stages["D_f32_projection_add_then_cast"],stages["E_source"])

def main():
    parser=argparse.ArgumentParser()
    for name in ("model","fixture","output"): parser.add_argument(name,type=Path)
    parser.add_argument("--native",type=Path)
    parser.add_argument("--dtype",choices=("fp16","bf16"),default="bf16")
    parser.add_argument("--prefill",type=int,default=2)
    args=parser.parse_args()
    model,fixture,output=args.model,args.fixture,args.output
    dtype=torch.float16 if args.dtype=="fp16" else torch.bfloat16
    output.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    micro()
    cfg=json.loads((model/"config.json").read_text())
    cfg["_attn_implementation"]="eager"
    cfg["pretraining_tp"]=1
    tree=ast.parse((model/"modeling_deepseek.py").read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
           and (n.name in names or n.name.startswith("yarn_"))]
    ctx=dict(torch=torch,nn=nn,F=F,math=math,warnings=warnings,Optional=Optional,Tuple=Tuple,
             DeepseekV2Config=SimpleNamespace,Cache=Cache,ACT2FN={"silu":F.silu},ATTENTION_CLASSES={})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(model/"modeling_deepseek.py"),"exec"),ctx)
    ctx["ATTENTION_CLASSES"]["eager"]=ctx["DeepseekV2Attention"]
    layer=ctx["DeepseekV2DecoderLayer"](SimpleNamespace(**cfg),0).eval().to(dtype)
    state=layer.state_dict()
    with torch.no_grad():
        for alias,name in aliases.items():
            state[name].copy_(torch.frombuffer(bytearray((fixture/f"{alias}.bin").read_bytes()),
                                                dtype=torch.bfloat16).reshape(state[name].shape))
    raw=(fixture/"tokens.bin").read_bytes()
    embedding=[]
    for i in range(6):
        cursor=i*4100
        embedding.append(torch.frombuffer(bytearray(raw[cursor+4:cursor+4100]),dtype=torch.bfloat16).clone())
    active={}
    layer.self_attn.o_proj.register_forward_hook(lambda mod, inp, out: active.update(
        attention=inp[0],output_projection=out,output_projection_fp32=F.linear(inp[0].float(),mod.weight.float())))
    layer.post_attention_layernorm.register_forward_hook(lambda _, inp, out: active.update(
        residual=inp[0],post_norm=out))
    layer.input_layernorm.register_forward_hook(lambda _, inp, out: active.update(input_norm=out))
    for key,name in (("q","q_proj"),("kv_a","kv_a_proj_with_mqa"),("kv_b","kv_b_proj")):
        getattr(layer.self_attn,name).register_forward_hook(
            lambda _,inp,out,stage=key:active.update({stage:out}))
    for name in ("gate","up","down"):
        getattr(layer.mlp,name+"_proj").register_forward_hook(
            lambda _, inp, out, key=name: active.update({key:out,**({"swiglu":inp[0]} if key=="down" else {})}))
    with torch.no_grad():
        for request in range(2):
            cache=Cache()
            for cursor,count in ((0,args.prefill),(args.prefill,1)):
                x=torch.stack([embedding[request*3+j%3] for j in range(cursor,cursor+count)])[None].to(dtype)
                positions=torch.arange(cursor,cursor+count)[None]
                mask=torch.zeros((1,1,count,cursor+count),dtype=dtype)
                mask.masked_fill_(torch.arange(cursor+count)[None,:]>positions.T,float("-inf"))
                active.clear()
                active["original_hidden"]=x
                result=layer(x,attention_mask=mask,position_ids=positions,past_key_value=cache,use_cache=True)[0]
                active["residual_fp32"]=x.float()+active["output_projection"].float()
                active["final"]=result
                if args.native:
                    path=args.native/f"{request}_{cursor}_attention.bin"
                    native_attention=torch.frombuffer(bytearray(path.read_bytes()),
                        dtype=torch.float32).reshape_as(x).to(dtype)
                    actual_projection=F.linear(native_attention,layer.self_attn.o_proj.weight)
                    active["matched_output_projection"]=actual_projection
                    active["matched_output_projection_fp32"]=F.linear(native_attention.float(),layer.self_attn.o_proj.weight.float())
                    active["matched_residual"]=x+actual_projection
                    native_projection_path=args.native/f"{request}_{cursor}_output_projection.bin"
                    same_projection=torch.frombuffer(bytearray(native_projection_path.read_bytes()),
                        dtype=torch.float32).reshape_as(x).to(dtype)
                    active["identical_projection_residual"]=x+same_projection
                for name,tensor in active.items():
                    flat=tensor.float().contiguous().view(-1)
                    (output/f"{request}_{cursor}_{name}.bin").write_bytes(struct.pack(f"<{flat.numel()}f",*flat.tolist()))
                    print("trace",request,cursor,name,tuple(tensor.shape),tensor.stride(),tensor.dtype)

if __name__=="__main__":
    main()
