"""Stage-local isolation against frozen unsharded launchers."""
import json

import torch
from safetensors import safe_open

from ..attention import NativeMaterializedAttention
from ..phase12_4.projection import PackedProjection
from ..phase13_3_2.probe import LIB, MODEL
from .mla import NativeCalls


def diagnose(replay):
    ops = replay.ops
    group = replay.group
    prefix = "model.layers.1.self_attn."
    index = json.loads((MODEL/"model.safetensors.index.json").read_text())["weight_map"]
    lib = LIB/"projection/libprojection.so"
    full = NativeCalls(group,ops.device,LIB,LIB/"cache/libpaged_kv.so",ops.cos,ops.sin,ops.scale)
    full.heads = 16
    attention = NativeMaterializedAttention(LIB/"attention/libnative_mlu.so")
    layer = replay.layer(1)

    def weight(suffix):
        name = prefix+suffix
        with safe_open(MODEL/index[name],framework="pt",device="cpu") as handle:
            return handle.get_tensor(name).to(ops.device)

    try:
        with PackedProjection(lib,weight("q_proj.weight")) as query, PackedProjection(lib,weight("o_proj.weight")) as out:
            for length,step in ((1,0),(1,1),(7,0)):
                residual,k,v,target,meta = replay.fixture(length,step,1)
                rows = residual.shape[0]
                cursor = length+step-rows
                x = layer.norm.forward_native(residual,layer.input_gamma,1e-6)
                q_wide = query(x)
                replay.check(group.all_gather(layer.q(x),dim=1),q_wide,meta,"q_wide_vs_full")
                latent,rotary = layer.kv_a(x).split((512,64),dim=1)
                latent = layer.norm.forward_native(latent.to(torch.bfloat16).contiguous(),layer.gamma,1e-6)
                expanded = group.all_gather(layer.kv_b(latent),dim=1).view(rows,16,256)
                q_full,_,_,pos = full.assemble(q_wide,rotary,expanded,cursor)
                q,_,_,_ = layer.project(residual,cursor)
                replay.check(q,q_full[replay.start:replay.stop],meta,"q_rope_vs_full")
                k,v = k.to(ops.device),v.to(ops.device)
                _,_,_,full_av = attention.forward_native(q_full,k,v,pos,0,k.shape[1],ops.scale)
                av,*_ = ops.attention(q,k[replay.start:replay.stop],v[replay.start:replay.stop],pos)
                replay.check(av,full_av[replay.start:replay.stop],meta,"av_vs_frozen_launcher")
                replay.check(out(full_av.transpose(0,1).contiguous().flatten(1)).to(torch.bfloat16),
                             target,meta,"full_launcher_vs_historical")
                dist_o,_,_ = layer.output(av,residual)
                replay.check(dist_o,out(full_av.transpose(0,1).contiguous().flatten(1)).to(torch.bfloat16),
                             meta,"distributed_vs_full_o")
    finally:
        layer.close()
