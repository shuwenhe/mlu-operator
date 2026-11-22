"""Native TP=2 path loaded by the frozen Vendor model runner."""
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm_mlu.v1.attention.backends.utils import get_common_metadata

from ..phase12_3.scheduled_attention import validate_batch
from ..phase13_2.linear import ColumnParallelLinear, LocalGemm, RowParallelLinear, validate_tp
from ..phase13_3_2.mapping import RULES
from ..phase13_3_2.projection import checksum
from ..phase13_3_3.probe import OrderedOutput
from ..phase13_4.full_model import ReplicatedDecoder
from ..phase13_4.mla import Cache, MLA, NativeCalls
from .cache import TP2CacheLayer
from .contracts import spans


def emit(event, **fields):
    print(json.dumps({"phase14_1":event, **fields}), flush=True)


class Observed:
    def __init__(self, operation, observer):
        self.operation, self.observer = operation, observer

    def __call__(self, tensor, *args):
        result = self.operation(tensor, *args)
        self.observer(tensor, result)
        return result


class ObservedNorm:
    def __init__(self, norm, observer):
        self.forward_native = Observed(norm.forward_native, observer)


class EngineMLA(MLA):
    def __init__(self, ops, library, ordered, weights, norm, audit=None):
        self.ops, self.strategy, self.ordered_library = ops, "ordered", ordered
        self.parts = []
        group = ops.group
        self.norm = norm
        self.gamma = weights["kv_a_layernorm.weight"]
        self.input_gamma = weights["input_layernorm.weight"]

        def add(cls, *args):
            projected = cls(*args)
            self.parts.append(projected)
            return projected

        try:
            self.q = add(ColumnParallelLinear,weights["q_proj.weight"],(3072,),group,library)
            self.kv_a = add(LocalGemm,weights["kv_a_proj_with_mqa.weight"],library)
            self.kv_b = add(ColumnParallelLinear,weights["kv_b_proj.weight"],(4096,),group,library)
            self.o = add(RowParallelLinear,weights["o_proj.weight"],2048,group,library)
            if getattr(ops, "benchmark", False):
                self.q = lambda x,q=self.q: ops.timed("q_gemm",lambda:q(x))
                self.kv_a = lambda x,kv=self.kv_a: ops.timed("kv_a_gemm",lambda:kv(x))
                self.kv_b = lambda x,kv=self.kv_b: ops.timed("kv_b_gemm",lambda:kv(x))
            if audit is not None:
                self.q = Observed(self.q,lambda x,y:audit("Q",x,y))
                self.kv_a = Observed(self.kv_a,lambda x,y:audit("KV_A",x,y))
                self.kv_b = Observed(self.kv_b,lambda x,y:audit("KV_B",x,y))
                self.norm = ObservedNorm(norm,lambda x,y:audit(
                    "KV_A_RMSNorm" if x.shape[-1]==512 else "input_RMSNorm",x,y))
        except Exception:
            self.close()
            raise


    def output(self, av, residual):
        local = av.transpose(0,1).contiguous().flatten(1)
        projection = OrderedOutput(self.ordered_library,self.o,self.ops.group,self.ops.device,local.shape[0])
        rounded, added, stages = self.ops.timed("o",lambda: projection(local,residual))
        self.ops.timings.update(stages)
        return rounded, added, projection.out


class WorkerCache(Cache):
    def __init__(self, ops, views):
        self.ops = ops
        self.k, self.v = views
        self.slots = self.k.shape[0]


class NeurXDeepSeekV2ForCausalLM(torch.nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        cfg = vllm_config.model_config
        hf = cfg.hf_config
        parallel = vllm_config.parallel_config
        if (
            (hf.hidden_size,hf.num_hidden_layers,hf.num_attention_heads,hf.kv_lora_rank,hf.q_lora_rank)
            != (2048,27,16,512,None)
            or cfg.dtype != torch.bfloat16 or not 0 < cfg.max_model_len <= 128
            or (parallel.tensor_parallel_size,parallel.pipeline_parallel_size,parallel.data_parallel_size)!=(2,1,1)
            or vllm_config.cache_config.enable_prefix_caching
        ):
            raise ValueError("Unsupported NeurX native TP=2 model configuration")
        self.layers = torch.nn.ModuleList(
            TP2CacheLayer(f"model.layers.{i}.self_attn.native_mla",vllm_config) for i in range(27)
        )
        self.group = get_tp_group()
        self.rank = self.group.rank_in_group
        self.device = torch.device("mlu",torch.mlu.current_device())
        validate_tp(self.group,self.device)
        if self.group.world_size != 2 or self.device.index != self.rank or dist.get_rank()!=self.rank:
            raise ValueError("Expected two local workers with matching rank/device indices")
        self.libraries = Path(os.environ["NATIVE_DEEPSEEK_LIBRARIES"])
        self.ordered = Path(os.environ["NEURX_ORDERED_O"])
        self.paged = Path(os.environ["NEURX_PAGED_EIGHT"])
        self.trace = bool(os.environ.get("NEURX_TP2_TRACE"))
        self.boundary_capture = os.environ.get("NEURX_TP2_BOUNDARIES","1") == "1"
        self.dump = os.environ.get("NEURX_TP2_DUMP")
        self.sequence = 0
        self.acceptance_fault = None
        self.forward_sequence = 0
        self.pending_logits = False
        self.cache_reported = False
        self.capture_projection = False
        self.benchmark = os.environ.get("NEURX_TP2_BENCHMARK")=="1"
        self.benchmark_steps = []
        self.mla = []
        emit("constructed",rank=self.rank,device=str(self.device),world_size=2,
             global_rank=dist.get_rank(),local_rank=self.device.index,
             model=type(self).__name__,heads=list(range(8*self.rank,8*self.rank+8)))

    def load_weights(self, weights):
        stream = torch.mlu.Stream(device=self.device)
        self.decoder = None
        local = [dict() for _ in self.layers]
        remaining = {}
        loaded = set()
        with torch.mlu.stream(stream):
            for name, tensor in weights:
                loaded.add(name)
                parts = name.split(".")
                if parts[:2]==["model","layers"] and len(parts)>4 and parts[3]=="self_attn":
                    suffix = ".".join(parts[4:])
                    shape, axis = RULES[suffix]
                    if tuple(tensor.shape)!=shape or tensor.dtype!=torch.bfloat16:
                        raise ValueError(f"Unexpected MLA weight {name}")
                    _,offset,width = spans(suffix,self.rank)
                    own = (tensor if axis is None else tensor.narrow(axis,offset,width)).contiguous()
                    local[int(parts[2])][suffix] = own.to(self.device)
                    if self.trace:
                        emit("shard",rank=self.rank,device=str(self.device),parameter=name,
                             axis=axis,offset=offset,shape=list(own.shape),sha256=checksum(own))
                elif len(parts)==5 and parts[:2]==["model","layers"] and parts[3]=="input_layernorm":
                    local[int(parts[2])]["input_layernorm.weight"]=tensor.to(self.device)
                else:
                    remaining[name] = tensor
            if len(loaded)!=5291:
                raise ValueError(f"Unexpected checkpoint parameter count {len(loaded)}")
            rope = Path(os.environ["NATIVE_DEEPSEEK_ROPE"])
            cos,sin = (torch.from_file(str(rope/f"{name}.bin"),size=131*64,dtype=torch.int16)
                       .view(torch.bfloat16).reshape(131,64).to(self.device) for name in ("cos","sin"))
            scale=float((rope/"scale.txt").read_text())
            self.ops = NativeCalls(self.group,self.device,self.libraries,self.paged,cos,sin,scale)
            self.ops.benchmark = self.benchmark
            self.decoder = ReplicatedDecoder(self.ops)
            self.decoder.weights = remaining
            try:
                for number,rank_weights in enumerate(local):
                    observer = (lambda op,x,y,layer=number: self.audit(op,layer,x,y)) if self.trace else None
                    self.mla.append(EngineMLA(self.ops,self.libraries/"projection/libprojection.so",
                                              self.ordered,rank_weights,self.decoder.norm,observer))
                stream.synchronize()
            except Exception:
                self.close()
                raise
        emit("loaded",rank=self.rank,device=str(self.device),parameters=len(loaded),layers=len(self.mla))
        return loaded

    def close(self):
        for layer in reversed(self.mla):
            layer.close()
        self.mla.clear()
        if self.decoder is not None:
            self.decoder.close()

    def embed_input_ids(self, input_ids):
        self.ops.stream.wait_stream(torch.mlu.current_stream(self.device))
        with torch.mlu.stream(self.ops.stream):
            return self.decoder.embedding(input_ids.int().contiguous())

    def validate(self, metadata, positions):
        error = None
        try:
            if any(len(layer.kv_cache)!=2 for layer in self.layers):
                raise ValueError("Worker KV cache not bound")
            blocks = self.layers[0].kv_cache[0].shape[0]//16
            spans_ = validate_batch(metadata,range(metadata.num_reqs),positions.int(),16,blocks)
            if any(span.length>128 for span in spans_):
                raise ValueError("Context exceeds max_model_len")
        except (ValueError, AttributeError, IndexError, TypeError) as exc:
            error = str(exc)
        failure = torch.tensor([int(error is not None)],device=self.device,dtype=torch.int32)
        if self.group.all_reduce(failure).item():
            raise ValueError(f"Scheduler validation failed across ranks: {error}")
        schedule = [(s.request_id,s.start,s.end,s.length,
                     metadata.block_table_tensor[i,:s.blocks].tolist(),
                     metadata.slot_mapping[s.start:s.end].tolist(),
                     positions[s.start:s.end].tolist()) for i,s in enumerate(spans_)]
        self.ops.agree(schedule)
        return spans_, schedule

    def audit(self, operator, layer, data, result):
        if self.capture_projection and operator in ("Q","KV_A","KV_B","KV_A_RMSNorm","input_RMSNorm"):
            self.capture(self.forward_sequence,layer,operator,result)
        if self.trace:
            emit("operator",rank=self.rank,device=str(self.device),layer=layer,operator=operator,
                 input_shape=list(data.shape),output_shape=list(result.shape),dtype=str(result.dtype),
                 stream=int(self.ops.stream.mlu_stream),status="PASS")

    def arm_acceptance_fault(self, kind):
        from .faults import FAULTS

        if kind not in FAULTS:
            raise ValueError("Unknown isolated acceptance fault")
        if self.rank==1:
            self.acceptance_fault = kind
            emit("fault_armed",rank=self.rank,kind=kind)
        return self.rank

    def capture(self, step, layer, name, tensor):
        if self.dump is None or not self.boundary_capture:
            return
        suffix = f"{layer}_{name}" if layer is not None else name
        path = Path(self.dump)/f"rank{self.rank}_forward{step}_{suffix}.bin"
        data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        with path.open("xb") as stream:
            stream.write(data)

    def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None,**kwargs):
        self.ops.stream.wait_stream(torch.mlu.current_stream(self.device))
        with torch.mlu.stream(self.ops.stream):
            error = None
            metadata = None
            profile = False
            try:
                if intermediate_tensors is not None or inputs_embeds is not None:
                    raise ValueError("Unsupported TP=2 model input")
                contexts = get_forward_context().attn_metadata
                profile = contexts is None or self.acceptance_fault == "missing-metadata"
                if not profile:
                    metadata = get_common_metadata()
                    if metadata is None:
                        metadata = contexts[self.layers[0].layer_name]
                    if self.acceptance_fault:
                        from .faults import alter
                        blocks = self.layers[0].kv_cache[0].shape[0]//16
                        metadata = alter(metadata,self.acceptance_fault,blocks)
                        emit("fault_injected",rank=self.rank,kind=self.acceptance_fault)
            except (ValueError, AttributeError, KeyError, TypeError) as exc:
                error = str(exc)
            flags = torch.tensor([int(error is not None),int(profile)],dtype=torch.int32,device=self.device)
            failures, profiles = self.group.all_reduce(flags).tolist()
            if failures or profiles not in (0,self.group.world_size):
                raise ValueError(f"Peer metadata preflight failed: {error}")
            if profiles:
                self.pending_logits = False
                return torch.zeros((input_ids.numel(),2048),device=self.device,dtype=torch.bfloat16)
            start_forward = time.perf_counter() if self.benchmark else 0
            spans_, schedule = self.validate(metadata,positions)
            timings = {"metadata_cncl":self.ops.timings["metadata_cncl"]} if self.benchmark else None
            if not self.cache_reported:
                for number,worker in enumerate(self.layers):
                    k,v = worker.kv_cache
                    if self.trace:
                        emit("cache_bound",rank=self.rank,device=str(self.device),layer=number,
                             k_shape=list(k.shape),v_shape=list(v.shape),
                             shared_storage=k.untyped_storage().data_ptr()==v.untyped_storage().data_ptr())
                self.cache_reported = True
            outputs = []
            self.capture_projection = len(spans_)==1 and self.dump is not None and self.boundary_capture
            for span,(_,start,end,total,table,slots,_) in zip(spans_,schedule):
                x = self.decoder.embedding(input_ids[start:end].int().contiguous())
                self.audit("embedding",-1,input_ids[start:end],x)
                if len(spans_)==1:
                    self.capture(self.forward_sequence,None,"embedding",x)
                for number,(layer,worker) in enumerate(zip(self.mla,self.layers)):
                    page = WorkerCache(self.ops,worker.kv_cache)
                    self.ops.timings.clear()
                    q,k,v,pos = layer.project(x,total-(end-start))
                    page.write(k,v,slots)
                    k,v = page.read(table,total)
                    av,raw,_,prob = self.ops.attention(q,k,v,pos)
                    o,added,wide = layer.output(av,x)
                    if len(spans_)==1:
                        for label,tensor in (("key",k),("value",v),("attention",o),
                                             ("ordered_fp32",wide),("residual",added),("rope_q",q),
                                             ("qk",raw),("softmax",prob),("av",av)):
                            self.capture(self.forward_sequence,number,label,tensor)
                    self.audit("ordered_O_FP32",number,av,wide)
                    for name,input_,output in (("Q_RoPE",x,q),("cache_write_read",q,k),
                                               ("QK",q,raw),("Softmax",raw,prob),("AV",prob,av),
                                               ("ordered_O_BF16",av,o),("residual",o,added)):
                        self.audit(name,number,input_,output)
                    if timings is not None:
                        for label,ms in self.ops.timings.items():
                            timings[label] = timings.get(label,0.0)+ms
                    x = self.decoder.ffn(number,added)
                    if len(spans_)==1:
                        self.capture(self.forward_sequence,number,"hidden",x)
                outputs.append(self.decoder.norm.forward_native(x,self.decoder._resident("model.norm.weight"),1e-6))
            self.capture_projection = False
            result = torch.cat(outputs)
            self.ops.stream.synchronize()
            if self.trace:
                emit("forward",rank=self.rank,device=str(self.device),requests=len(outputs),schedule=schedule,
                     step=self.forward_sequence)
            if timings is not None:
                self.benchmark_steps.append(dict(ms=(time.perf_counter()-start_forward)*1000,stages=timings))
            self.forward_sequence += 1
            self.pending_logits = True
            return result

    def compute_logits(self, hidden_states):
        logits = self.decoder.compute_logits(hidden_states)
        if self.dump and self.pending_logits:
            path = Path(self.dump)/f"rank{self.rank}_logits{self.sequence}.bin"
            with path.open("xb") as stream:
                stream.write(logits.cpu().contiguous().numpy().tobytes())
            self.sequence += 1
        self.pending_logits = False
        self.audit("logits",-1,hidden_states,logits)
        return logits
