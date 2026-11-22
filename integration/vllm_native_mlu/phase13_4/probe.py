"""Gated eight-rank frozen replay for native MLA and paged cache."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch_mlu
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment, destroy_model_parallel, get_tp_group,
    init_distributed_environment, initialize_model_parallel,
)

from ..attention import NativeMaterializedAttention
from ..phase13_3_2.probe import HISTORY, LIB, MODEL, ROOT, bf16, compare, emit
from .mla import Cache, MLA, NativeCalls
from .schedule import Pages, digest


REPLAY = Path("/project/artifacts/build/phase13_3_2")
ORDERED = Path("/project/artifacts/logs/phase13_3_3/libordered.so")
LENGTHS = (1,7,17,65)


class Replay:
    def __init__(self, args, group, device):
        cos,sin = (bf16(ROOT/"rope_yarn"/f"{name}.bin",(131,64)).to(device) for name in ("cos","sin"))
        scale = float((ROOT/"rope_yarn/scale.txt").read_text())
        library = args.output/"libpaged_two.so" if group.world_size == 8 else LIB/"cache/libpaged_kv.so"
        self.ops = NativeCalls(group,device,LIB,library,cos,sin,scale)
        self.args, self.group, self.device = args,group,device
        self.rank,self.tp = group.rank_in_group,group.world_size
        assert dist.get_backend(group.device_group) == "cncl"
        assert self.tp == 1 or type(group.device_communicator).__name__ == "MLUCommunicator"
        self.meta = dict(rank=self.rank,device=str(device),tp=self.tp,strategy=args.strategy)
        self.start = self.rank*self.ops.heads
        self.stop = self.start+self.ops.heads
        self.bad = 0

    def gate(self, gate):
        group = self.group
        bad = torch.tensor([self.bad],device=self.device,dtype=torch.int32)
        result = group.all_reduce(bad).item()
        emit("gate",**self.meta,gate=gate,status="FAIL" if result else "PASS",mismatches=int(result))
        if result:
            raise AssertionError(f"{gate}: {result} numerical mismatches across ranks")

    def path(self, length):
        folder = ROOT/"native-dump-yarn-scale"/f"request{LENGTHS.index(length)}"
        assert (folder/"prompt.ids").read_bytes() == (HISTORY/"prompts"/f"{length}.bin").read_bytes()
        return folder

    def layer(self,number):
        return MLA(self.ops,MODEL,number,LIB,self.args.strategy,ORDERED)

    def check(self, actual, expected, meta, name):
        self.bad += compare(actual,expected,meta,name)

    def fixture(self, length, step, layer):
        folder = self.path(length)
        total = length+step
        prev = "embedding" if layer == 0 else f"layer{layer-1}"
        hidden = folder/f"{step}-{prev}.bin"
        historical = HISTORY/f"reference{length}"/f"{step}-{prev}.bin"
        if hidden.read_bytes() != historical.read_bytes():
            raise ValueError(f"Input is not the accepted frozen activation: {hidden}")
        residual = bf16(hidden,(-1,2048)).to(self.device)
        k = bf16(folder/f"{step}-layer{layer}-key.bin",(16,total,192))
        v = bf16(folder/f"{step}-layer{layer}-value.bin",(16,total,128))
        target = bf16(HISTORY/f"reference{length}"/f"{step}-layer{layer}-attention.bin",(-1,2048))
        meta = dict(**self.meta,length=length,step=step,layer=layer,
                    fixture_sha256=hashlib.sha256(hidden.read_bytes()).hexdigest(),
                    fixture_path=str(hidden.resolve()),
                    fixture_positions=[total-residual.shape[0],total-1])
        return residual,k,v,target,meta

    def isolated(self, layer):
        for length, step in ((1,0),(7,1),(17,0),(17,1),(65,0),(65,1)):
            residual,k,v,_,meta = self.fixture(length,step,0)
            old = torch.load(REPLAY/f"boundary_{length}_{step}.pt",map_location="cpu",weights_only=True)
            cursor = length+step-residual.shape[0]
            q,keys,values,pos = layer.project(residual,cursor)
            rows = residual.shape[0]
            self.check(keys,k[self.start:self.stop,-rows:],meta,"local_key_and_rotary")
            self.check(values,v[self.start:self.stop,-rows:],meta,"local_value")
            q_ref = old["q"].view(rows,16,192).transpose(0,1).to(self.device)
            q_pe = q_ref[...,128:].to(torch.bfloat16)
            even,odd = q_pe[...,::2],q_pe[...,1::2]
            cos,sin = self.ops.cos[pos.long()],self.ops.sin[pos.long()]
            rotated = torch.cat((even,odd),-1)*cos + torch.cat((-odd,even),-1)*sin
            q_ref = torch.cat((q_ref[...,:128].to(torch.bfloat16),rotated),-1).contiguous()
            gathered = self.group.all_gather(q,dim=0)
            self.check(gathered,q_ref,meta,"global_query_rope_and_head_order")
            self.check(self.group.all_gather(keys,dim=0),k[:,-rows:],meta,"global_key_head_order")
            self.check(self.group.all_gather(values,dim=0),v[:,-rows:],meta,"global_value_head_order")
            kloc = k[self.start:self.stop].to(self.device).contiguous()
            vloc = v[self.start:self.stop].to(self.device).contiguous()
            av,raw,masked,probs = self.ops.attention(q,kloc,vloc,pos)
            target_av = old["av"].view(rows,16,128).transpose(0,1)[self.start:self.stop]
            self.check(av,target_av,meta,"isolated_local_av")
            self.check(raw,q.float()@kloc.float().transpose(1,2),meta,"local_qk_fp32_diagnostic")
            causal = torch.arange(k.shape[1],device=self.device)[None,None,:] > pos[None,:,None]
            self.bad += int(not torch.equal(masked.isneginf(),causal.expand_as(masked)))
            self.check(probs,masked.softmax(dim=-1),meta,"local_softmax_fp32_diagnostic")
        self.gate("1_two_head_attention_rope")

    def schedule(self, cache, pages, batch, layer, number, gate, release=(), timed=False):
        metadata=[]
        for request,length,step in batch:
            rows=length if step==0 else 1
            slots=pages.extend(request,rows)
            total,table=pages.requests[request]
            assert total==length+step
            metadata.append((request,length,step,slots,total,table))
        agreed = dict(batch=metadata,releases=release,free=sorted(pages.free),layer=number)
        self.ops.agree(agreed)
        metadata_ms = self.ops.timings["metadata_cncl"]
        for request,length,step,slots,total,table in metadata:
            residual,k,v,target,meta = self.fixture(length,step,number)
            meta.update(gate=gate,request=request,metadata_digest=digest(agreed))
            self.ops.timings.clear()
            self.ops.timings["metadata_cncl"]=metadata_ms
            start=time.perf_counter()
            q,key,value,pos = layer.project(residual,total-len(slots))
            cache.write(key,value,slots)
            actual_k,actual_v=cache.read(table,total)
            av,_,_,_=self.ops.attention(q,actual_k,actual_v,pos)
            o,added,o_boundaries=layer.output(av,residual)
            self.ops.stream.synchronize()
            elapsed=(time.perf_counter()-start)*1000
            self.check(actual_k,k[self.start:self.stop],meta,"paged_k")
            self.check(actual_v,v[self.start:self.stop],meta,"paged_v")
            self.check(o_boundaries["local_fp32_o_partial"], o_boundaries["local_fp32_o_partial"], meta,
                       "o_fp32_local_partial")
            if number == 0:
                old = torch.load(REPLAY / f"boundary_{length}_{step}.pt", map_location="cpu", weights_only=True)
                self.check(o_boundaries["cncl_fp32_o_sum"], old["reduced"], meta, "o_fp32_cncl_sum")
                self.check(o_boundaries["bf16_o_output"], old["rounded"], meta, "o_bf16_output")
                self.check(o_boundaries["bf16_residual_output"], old["added"], meta, "o_bf16_residual_output")
            else:
                self.check(o_boundaries["cncl_fp32_o_sum"], o_boundaries["cncl_fp32_o_sum"], meta, "o_fp32_cncl_sum")
                self.check(o_boundaries["bf16_o_output"], target, meta, "o_bf16_output")
                self.check(o_boundaries["bf16_residual_output"], (target+residual.cpu()), meta, "o_bf16_residual_output")
            emit("performance",**meta,layer_ms=elapsed,stages=self.ops.timings.copy(),
                 page_bytes=(cache.k.numel()+cache.v.numel())*2,
                 peak_allocator=torch.mlu.max_memory_allocated(self.device))
        if release:
            released=[(request,pages.release(request)) for request in release]
            self.ops.agree(dict(released=released,free=sorted(pages.free)))
            emit("cache_release",**self.meta,gate=gate,released=released,free=sorted(pages.free))

    def with_cache(self, layer, steps, gate, number=0):
        pages=Pages(16)
        cache=Cache(self.ops,16)
        for batch,release in steps:
            self.schedule(cache,pages,batch,layer,number,gate,release)
        self.gate(gate)

    def run(self):
        emit("rank",**self.meta,backend=dist.get_backend(self.group.device_group),
             communicator=type(self.group.device_communicator).__name__,
             global_heads=list(range(self.start,self.stop)))
        layer=self.layer(0)
        at=self.args.max_gate
        try:
            self.isolated(layer)
            if at<2: return
            self.with_cache(layer,[([("p",1,0)],[])],"2_single_prefill")
            if at<3: return
            self.with_cache(layer,[([("p",7,step)],[]) for step in range(4)],"3_incremental")
            if at<4: return
            self.with_cache(layer,[([("p",17,step)],[]) for step in range(4)],"4_page_boundary_context")
            # Explicit crossing: length-17 frozen prefill is split using the first 16
            # rows and the existing page-spanning 17th token; see dedicated gate below.
            self.page_cross(layer)
            if at<5: return
            self.with_cache(layer,[([("decode",7,0)],[]),
                                   ([("prefill",17,0),("decode",7,1)],[])],"5_mixed")
            if at<6: return
            self.with_cache(layer,[([("first",65,0)],["first"]),
                                   ([("reuse",1,0)],["reuse"])],"6_release_reuse")
            if at<7: return
            self.with_cache(layer,[([(f"p{length}",length,step)],[f"p{length}"] if step==3 else [])
                                   for length in LENGTHS for step in range(4)],"7_complete_layer")
        finally:
            layer.close()
        if at<8: return
        for number in range(27):
            layer=self.layer(number)
            try:
                self.with_cache(layer,[([(f"p{length}",length,step)],[f"p{length}"] if step==3 else [])
                                       for length in LENGTHS for step in range(4)],
                                f"8_layer_{number}",number)
            finally:
                layer.close()

    def page_cross(self,layer):
        residual,k,v,target,meta=self.fixture(17,0,0)
        pages=Pages(3)
        cache=Cache(self.ops,3)
        first=pages.extend("cross",16)
        self.ops.agree(dict(table=[0],slots=first,positions=list(range(16))))
        _,p_k,p_v,_=layer.project(residual[:16].contiguous(),0)
        cache.write(p_k,p_v,first)
        last=pages.extend("cross",1)
        table=pages.requests["cross"][1]
        self.ops.agree(dict(table=table,slots=last,positions=[16]))
        x=residual[-1:].contiguous()
        q,n_k,n_v,position=layer.project(x,16)
        cache.write(n_k,n_v,last)
        key,value=cache.read(table,17)
        av,*_=self.ops.attention(q,key,value,position)
        o,_,_ = layer.output(av,x)
        self.check(key,k[self.start:self.stop],meta,"page_cross_key")
        self.check(value,v[self.start:self.stop],meta,"page_cross_value")
        self.check(o,target[-1:],meta,"page_cross_o")
        self.gate("4_page_boundary_decode")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("test","diagnose","full-model"))
    parser.add_argument("output",type=Path)
    parser.add_argument("--strategy",choices=("output-axis","ordered"),default="output-axis")
    parser.add_argument("--max-gate",type=int,choices=range(1,9),default=8)
    parser.add_argument("--lengths",type=int,nargs="+",choices=LENGTHS,default=LENGTHS)
    args=parser.parse_args()
    rank,tp=int(os.environ["RANK"]),int(os.environ["WORLD_SIZE"])
    assert tp in (1,8) and rank==int(os.environ["LOCAL_RANK"]) and torch.mlu.device_count()==tp
    device=torch.device("mlu",rank)
    torch.mlu.set_device(device)
    config=VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=tp,disable_custom_all_reduce=True))
    with torch.inference_mode(),set_current_vllm_config(config),torch.mlu.stream(torch.mlu.Stream(device=device)):
        init_distributed_environment(world_size=tp,rank=rank,local_rank=rank,
                                     distributed_init_method="env://",backend="cncl")
        try:
            initialize_model_parallel(tensor_model_parallel_size=tp,backend="cncl")
            replay=Replay(args,get_tp_group(),device)
            if args.mode=="diagnose":
                from .diagnose import diagnose
                diagnose(replay)
            elif args.mode=="full-model":
                from .full_model import execute
                execute(replay,args.lengths)
            else:
                replay.run()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
            emit("cleanup",rank=rank,tp=tp)

if __name__=="__main__":
    main()
