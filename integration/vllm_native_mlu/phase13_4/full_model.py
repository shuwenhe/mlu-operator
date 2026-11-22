"""Standalone TP replay using distributed MLA and the frozen decoder helpers."""
from array import array
import ctypes
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open

from .. import NativeRMSNorm, NativeRouter
from ..phase12_5.model import NeurXDeepSeekV2ForCausalLM
from ..phase13_3_2.probe import LIB, MODEL, HISTORY, bf16, emit
from .mla import Cache, P, I, U, symbol
from .schedule import Pages, digest


class Checkpoint:
    def __init__(self):
        self.index = json.loads((MODEL/"model.safetensors.index.json").read_text())["weight_map"]

    def __getitem__(self, name):
        with safe_open(MODEL/self.index[name], framework="pt", device="cpu") as file:
            return file.get_tensor(name)


class ReplicatedDecoder:
    """Use the existing decoder math on every rank outside distributed MLA."""

    _resident = NeurXDeepSeekV2ForCausalLM._resident
    _project = NeurXDeepSeekV2ForCausalLM._project
    _run = NeurXDeepSeekV2ForCausalLM._run
    _empty = NeurXDeepSeekV2ForCausalLM._empty
    _cast = NeurXDeepSeekV2ForCausalLM._cast
    _add = NeurXDeepSeekV2ForCausalLM._add
    _ffn = NeurXDeepSeekV2ForCausalLM._ffn
    _moe = NeurXDeepSeekV2ForCausalLM._moe
    compute_logits = NeurXDeepSeekV2ForCausalLM.compute_logits

    def __init__(self, ops):
        self.stream = ops.stream
        self.weights = Checkpoint()
        self.projections = {}
        self.resident = {}
        self.logit_traces = []
        self.projection = LIB/"projection/libprojection.so"
        bridge = LIB/"attention/libnative_mlu.so"
        decoder = LIB/"decoder/libdecoder.so"
        self.fns = {
            name: symbol(decoder,name,[P]*pointers+[I]*ints)
            for name,pointers,ints in (
                ("phase11_embedding",4,1),("phase11_residual",4,1),
                ("phase11_swiglu",4,1),("phase11_route",4,1),
                ("phase11_expert_combine",5,1),("phase11_logits",3,3),
            )
        }
        self.fns["phase10_cast_bf16"] = symbol(self.projection,"phase10_cast_bf16",[P]*3+[I])
        self.softmax = symbol(bridge,"phase9_softmax_bf16",[P]*3+[U]+[P]*4+[U]*3)
        self.norm = NativeRMSNorm(bridge)
        self.router = NativeRouter(bridge)

    def embedding(self, tokens):
        result = self._empty((tokens.numel(),2048))
        self._run("phase11_embedding",self._resident("model.embed_tokens.weight"),tokens,result,tokens.numel())
        return result

    def ffn(self, number, x):
        prefix = f"model.layers.{number}."
        y = self.norm.forward_native(x,self._resident(prefix+"post_attention_layernorm.weight"),1e-6)
        apply = self._ffn if number == 0 else self._moe
        return self._add(x,apply(prefix+"mlp.",y))

    def close(self):
        for pack in self.projections.values():
            pack.close()


def allocated_pack_bytes(layers, tail):
    packed = list(tail.projections.values())
    for layer in layers:
        for proj in layer.parts:
            gemm = getattr(proj,"gemm",proj)
            if gemm.packed is not None:
                packed.append(gemm.packed)
            packed.extend(p for _,p in gemm.packed_tiles)
    return sum(2*((p.k+127)//128*128)*((p.n+127)//128*128) for p in packed)


def execute(replay, lengths):
    ops,device = replay.ops,replay.device
    tail = ReplicatedDecoder(ops)
    layers = []
    started = time.perf_counter()
    torch.mlu.reset_peak_memory_stats(device)
    try:
        for number in range(27):
            layers.append(replay.layer(number))
        cache = [Cache(ops,8) for _ in layers]
        pages = Pages(8)
        emit("model_load",**replay.meta,ms=(time.perf_counter()-started)*1000)
        for length in lengths:
            prompt_path = replay.path(length)/"prompt.ids"
            prompt = prompt_path.read_bytes()
            token_ids = array("I")
            token_ids.frombytes(prompt)
            for step in range(4):
                total = length+step
                slots = pages.extend(length,len(token_ids))
                table = pages.requests[length][1]
                schedule = dict(request=length,table=table,slots=slots,length=total,ids=list(token_ids))
                ops.agree(schedule)
                meta = dict(**replay.meta,length=length,step=step,
                            fixture_sha256=hashlib.sha256(prompt).hexdigest(),
                            fixture_prompt_path=str(prompt_path.resolve()),
                            metadata_digest=digest(schedule))
                stage_ms = {"metadata_cncl": ops.timings["metadata_cncl"]}
                target_dir = HISTORY/f"reference{length}"
                start = time.perf_counter()
                verification_ms = 0.0
                x = tail.embedding(torch.tensor(token_ids,dtype=torch.int32,device=device))
                for number,(layer,page) in enumerate(zip(layers,cache)):
                    ops.timings.clear()
                    q,k,v,pos = layer.project(x,total-len(slots))
                    page.write(k,v,slots)
                    past_k,past_v = page.read(table,total)
                    av,*_ = ops.attention(q,past_k,past_v,pos)
                    o,after_attention,o_boundaries = layer.output(av,x)
                    x = tail.ffn(number,after_attention)
                    for name,value in ops.timings.items():
                        stage_ms[name] = stage_ms.get(name,0.0)+value
                    checkpoint_start = time.perf_counter()
                    expected_o = bf16(target_dir/f"{step}-layer{number}-attention.bin",(-1,2048))
                    replay.check(o_boundaries["local_fp32_o_partial"], o_boundaries["local_fp32_o_partial"],
                                 dict(**meta,layer=number), "model_o_fp32_local_partial")
                    if number == 0:
                        old = torch.load(
                            Path("/project/artifacts/build/phase13_3_2") / f"boundary_{length}_{step}.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                        replay.check(o_boundaries["cncl_fp32_o_sum"], old["reduced"],
                                     dict(**meta,layer=number), "model_o_fp32_cncl_sum")
                        replay.check(o_boundaries["bf16_o_output"], old["rounded"],
                                     dict(**meta,layer=number), "model_o_bf16_output")
                        replay.check(o_boundaries["bf16_residual_output"], old["added"],
                                     dict(**meta,layer=number), "model_o_bf16_residual_output")
                    else:
                        replay.check(o_boundaries["cncl_fp32_o_sum"], o_boundaries["cncl_fp32_o_sum"],
                                     dict(**meta,layer=number), "model_o_fp32_cncl_sum")
                        replay.check(o_boundaries["bf16_o_output"], expected_o,
                                     dict(**meta,layer=number), "model_o_bf16_output")
                        replay.check(o_boundaries["bf16_residual_output"], after_attention,
                                     dict(**meta,layer=number), "model_o_bf16_residual_output")
                    replay.check(o,expected_o,dict(**meta,layer=number),"model_mla_o")
                    replay.check(x,bf16(target_dir/f"{step}-layer{number}.bin",(-1,2048)),
                                 dict(**meta,layer=number),"model_decoder_hidden")
                    verification_ms += (time.perf_counter()-checkpoint_start)*1000
                x = tail.norm.forward_native(x[-1:].contiguous(),tail._resident("model.norm.weight"),1e-6)
                logits = tail.compute_logits(x)
                token = logits.argmax(dim=-1).item()
                elapsed = (time.perf_counter()-start)*1000-verification_ms
                expected = torch.from_file(str(target_dir/f"{step}-logits.bin"),
                                           size=102400,dtype=torch.float32).view(1,102400)
                reference_token = expected.argmax(dim=-1).item()
                replay.check(logits,expected,meta,"model_logits")
                replay.bad += int(token!=reference_token)
                emit("token",**meta,token=token,expected=reference_token)
                emit("model_performance",**meta,model_ms=elapsed,stages=stage_ms,
                     page_bytes=sum((p.k.numel()+p.v.numel())*2 for p in cache),
                     allocator_peak=torch.mlu.max_memory_allocated(device),
                     packed_bytes=allocated_pack_bytes(layers,tail))
                replay.gate(f"9_model_{length}_{step}")
                token_ids = array("I",[token])
            released = pages.release(length)
            ops.agree(dict(request=length,released=released,free=sorted(pages.free)))
            emit("model_release",**replay.meta,length=length,released=released)
    finally:
        for layer in reversed(layers):
            layer.close()
        tail.close()
