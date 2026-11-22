"""Isolated native MLA with two physical attention heads per worker."""
import ctypes
import json
import time

import torch
from safetensors import safe_open

from .. import NativeRMSNorm
from ..attention import NativeMaterializedAttention
from ..phase13_2.linear import ColumnParallelLinear, LocalGemm, RowParallelLinear, validate_tp
from ..phase13_3_2.projection import load_weight
from ..phase13_3_3.probe import OrderedOutput
from .schedule import BLOCK_SIZE, digest


P, I, U = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32


def symbol(library, name, signature):
    fn = ctypes.CDLL(str(library))[name]
    fn.argtypes = signature
    fn.restype = I
    return fn


class NativeCalls:
    def __init__(self, group, device, libraries, paging, cos, sin, scale):
        self.group, self.device = group, device
        self.heads = 16 // group.world_size
        self.cos, self.sin, self.scale = cos, sin, scale
        self.stream = torch.mlu.current_stream(device)
        self.rope = symbol(libraries / "decoder/libdecoder.so", "worker_rope_bf16", [P]*6+[I])
        self.paging = symbol(paging, "phase12_paged_kv", [P]*6+[I]*4)
        self.qk, self.softmax, self.av = NativeMaterializedAttention(
            libraries / "attention/libnative_mlu.so").launchers[torch.bfloat16]
        self.timings = {}

    def timed(self, name, fn):
        start = time.perf_counter()
        out = fn()
        self.stream.synchronize()
        self.timings[name] = (time.perf_counter()-start)*1000
        return out

    def launch(self, name, function, *args):
        values = [arg.data_ptr() if isinstance(arg, torch.Tensor) else arg for arg in args]
        rc = self.timed(name, lambda: function(self.stream.mlu_stream, *values))
        if rc:
            raise RuntimeError(f"{name}: CNRT status={rc}")

    def agree(self, metadata):
        local = torch.tensor(digest(metadata), device=self.device, dtype=torch.int32)
        received = self.timed("metadata_cncl", lambda: self.group.all_gather(local, dim=0))
        if not torch.equal(received.reshape(-1,8), local.expand(self.group.world_size,-1)):
            raise ValueError("Rank schedule digest mismatch")

    def assemble(self, q, shared_k, expanded, cursor):
        rows = q.shape[0]
        heads = self.heads
        positions = torch.arange(cursor, cursor+rows, dtype=torch.int32, device=self.device)
        rope = q.new_zeros((17,rows,64), dtype=torch.bfloat16)
        q = q.view(rows,heads,192).transpose(0,1)
        rope[:heads].copy_(q[...,128:])
        rope[heads].copy_(shared_k)
        rotated = torch.empty_like(rope)
        self.launch("rope", self.rope, rope, self.cos, self.sin, positions, rotated, rows)
        key, value = expanded.transpose(0,1).split((128,128),dim=-1)
        query = torch.cat((q[...,:128].to(torch.bfloat16),rotated[:heads]),dim=-1)
        key = torch.cat((key.to(torch.bfloat16),rotated[heads:heads+1].expand(heads,-1,-1)),dim=-1)
        return query.contiguous(), key.contiguous(), value.to(torch.bfloat16).contiguous(), positions

    def attention(self, q, k, v, positions):
        h,m,_ = q.shape
        length = k.shape[1]
        score = (h,m,length)
        output = (h,m,128)

        def alloc(shape, dtype=torch.float32):
            return q.new_empty(shape, dtype=dtype)

        raw, scaled, masked = alloc(score), alloc(score,q.dtype), alloc(score)
        maximum, denominator = alloc((h,m)), alloc((h,m))
        probs, low = alloc(score), alloc(score,q.dtype)
        av, result = alloc(output), alloc(output,q.dtype)
        self.launch("qk",self.qk,q,k,positions,0,self.scale,raw,scaled,masked,h,m,length,length)
        self.launch("softmax",self.softmax,masked,None,0,maximum,denominator,probs,low,h*m,length,length)
        self.launch("av",self.av,low,v,av,result,h,m,length,length)
        return result, raw, masked, probs


class Cache:
    def __init__(self, ops, blocks):
        self.ops = ops
        self.slots = blocks * BLOCK_SIZE
        self.k = torch.empty((self.slots,ops.heads,192),dtype=torch.bfloat16,device=ops.device)
        self.v = torch.empty((self.slots,ops.heads,128),dtype=torch.bfloat16,device=ops.device)

    def write(self, k, v, slots):
        mapping = torch.tensor(slots,dtype=torch.int32,device=self.ops.device)
        self.ops.launch("cache_write",self.ops.paging,self.k,self.v,k,v,mapping,len(slots),BLOCK_SIZE,self.slots,1)

    def read(self, table, length):
        k = self.k.new_empty((self.ops.heads,length,192))
        v = self.v.new_empty((self.ops.heads,length,128))
        mapping = torch.tensor(table,dtype=torch.int32,device=self.ops.device)
        self.ops.launch("cache_read",self.ops.paging,self.k,self.v,k,v,mapping,length,BLOCK_SIZE,self.slots,0)
        return k,v


class MLA:
    def __init__(self, ops, checkpoint, layer, libraries, strategy, ordered_library):
        group, device = ops.group, ops.device
        validate_tp(group,device)
        rank,tp = group.rank_in_group,group.world_size
        self.ops, self.strategy, self.ordered_library = ops,strategy,ordered_library
        self.parts = []
        index = json.loads((checkpoint/"model.safetensors.index.json").read_text())["weight_map"]
        lib = libraries/"projection/libprojection.so"

        def weight(name):
            return load_weight(checkpoint,index,layer,name,rank,tp).to(device)

        def pack(layer_type,*args):
            packed = layer_type(*args)
            self.parts.append(packed)
            return packed

        try:
            self.q = pack(ColumnParallelLinear,weight("q_proj.weight"),(3072,),group,lib)
            self.kv_a = pack(LocalGemm,weight("kv_a_proj_with_mqa.weight"),lib)
            self.gamma = weight("kv_a_layernorm.weight")
            self.kv_b = pack(ColumnParallelLinear,weight("kv_b_proj.weight"),(4096,),group,lib)
            self.norm = NativeRMSNorm(libraries/"attention/libnative_mlu.so")
            prefix = f"model.layers.{layer}.self_attn."
            if strategy == "output-axis":
                name = prefix+"o_proj.weight"
                width = 2048//tp
                with safe_open(checkpoint/index[name],framework="pt",device="cpu") as f:
                    tensor = f.get_slice(name)
                    assert tensor.get_shape() == [2048,2048] and tensor.get_dtype() == "BF16"
                    local = tensor[rank*width:(rank+1)*width].to(device)
                self.o = pack(ColumnParallelLinear,local,(2048,),group,lib)
            elif strategy == "ordered":
                self.o = pack(RowParallelLinear,weight("o_proj.weight"),2048,group,lib)
            else:
                raise ValueError("Explicit O strategy required")
            name = f"model.layers.{layer}.input_layernorm.weight"
            with safe_open(checkpoint/index[name],framework="pt",device="cpu") as f:
                self.input_gamma = f.get_tensor(name).to(device)
        except Exception:
            self.close()
            raise

    def project(self, residual, cursor):
        x = self.norm.forward_native(residual,self.input_gamma,1e-6)
        q = self.q(x)
        latent, rotary = self.kv_a(x).split((512,64),dim=1)
        latent = self.norm.forward_native(latent.to(torch.bfloat16).contiguous(),self.gamma,1e-6)
        expanded = self.kv_b(latent).view(x.shape[0],self.ops.heads,256)
        return self.ops.assemble(q,rotary,expanded,cursor)

    def output(self, av, residual):
        local = av.transpose(0,1).contiguous().flatten(1)
        if self.strategy == "ordered":
            projection = OrderedOutput(self.ordered_library,self.o,self.ops.group,self.ops.device,local.shape[0])
            rounded, added, stages = self.ops.timed("o",lambda: projection(local,residual))
            self.ops.timings.update(stages)
            partial = self.ops.timed("o_fp32_local_partial", lambda: self.o.gemm(local))
            reduced = self.ops.timed("o_fp32_cncl_sum", lambda: self.ops.group.all_reduce(partial.clone()))
            return rounded, added, {
                "local_fp32_o_partial": partial,
                "cncl_fp32_o_sum": reduced,
                "bf16_o_output": rounded,
                "bf16_residual_output": added,
            }
        group = self.ops.group
        value = self.ops.timed("o_input_cncl",lambda: group.all_gather(local,dim=1))
        partial = self.ops.timed("o",lambda: self.o(value.contiguous()))
        reduced = self.ops.timed("o_output_cncl",lambda: self.o.gather(partial))
        rounded = reduced.to(torch.bfloat16)
        added = rounded+residual
        return rounded, added, {
            "local_fp32_o_partial": partial,
            "cncl_fp32_o_sum": reduced,
            "bf16_o_output": rounded,
            "bf16_residual_output": added,
        }

    def close(self):
        for part in reversed(self.parts):
            part.close()
        self.parts.clear()
