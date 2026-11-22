"""Supplemental full-head boundaries from frozen activation bytes on one MLU."""
from pathlib import Path
import argparse

import torch
import torch_mlu
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel, get_tp_group,
    destroy_model_parallel, destroy_distributed_environment,
)

from ..phase13_3_2.probe import ROOT, bf16, LIB, MODEL
from ..phase13_4.mla import NativeCalls, MLA


def run(output, group, device):
    folder = ROOT/"native-dump-yarn-scale/request0"
    cos,sin = [bf16(ROOT/"rope_yarn"/f"{name}.bin",(131,64)).to(device) for name in ("cos","sin")]
    scale=float((ROOT/"rope_yarn/scale.txt").read_text())
    ops=NativeCalls(group,device,LIB,LIB/"cache/libpaged_kv.so",cos,sin,scale)
    for number in range(27):
        layer=MLA(ops,MODEL,number,LIB,"ordered",None)
        try:
            for step in (0,1):
                prev="embedding" if number==0 else f"layer{number-1}"
                x=bf16(folder/f"{step}-{prev}.bin",(1,2048)).to(device)
                norm=layer.norm.forward_native(x,layer.input_gamma,1e-6)
                q=layer.q(norm)
                kva=layer.kv_a(norm)
                latent=layer.norm.forward_native(
                    kva[:,:512].to(torch.bfloat16).contiguous(),layer.gamma,1e-6)
                kvb=layer.kv_b(latent)
                query,_,_,pos=ops.assemble(q,kva[:,512:],kvb.view(1,16,256),step)
                key=bf16(folder/f"{step}-layer{number}-key.bin",(16,step+1,192)).to(device)
                value=bf16(folder/f"{step}-layer{number}-value.bin",(16,step+1,128)).to(device)
                av,raw,_,prob=ops.attention(query,key,value,pos)
                wide=layer.o.gemm(av.transpose(0,1).contiguous().flatten(1))
                frozen=bf16(folder/f"{step}-layer{number}-attention.bin",(1,2048))
                if not torch.equal(wide.to(torch.bfloat16).cpu(),frozen):
                    raise ValueError(f"Supplemental reference disagrees with frozen O: {number=}, {step=}")
                for name,tensor in (("input_RMSNorm",norm),("Q",q),("KV_A",kva),("KV_A_RMSNorm",latent),
                                    ("KV_B",kvb),("rope_q",query),("qk",raw),("softmax",prob),
                                    ("av",av),("ordered_fp32",wide)):
                    with (output/f"forward{step}_{number}_{name}.bin").open("xb") as stream:
                        stream.write(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        finally:
            layer.close()
    print("Supplemental diagnostic unsharded boundaries and frozen O anchors PASS",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=False)
    assert torch.mlu.device_count()==1
    device=torch.device("mlu",0)
    torch.mlu.set_device(device)
    cfg=VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=1,disable_custom_all_reduce=True))
    with torch.inference_mode(),set_current_vllm_config(cfg),torch.mlu.stream(torch.mlu.Stream(device=device)):
        init_distributed_environment(world_size=1,rank=0,local_rank=0,
                                     distributed_init_method="tcp://127.0.0.1:29641",backend="cncl")
        try:
            initialize_model_parallel(tensor_model_parallel_size=1,backend="cncl")
            run(args.out,get_tp_group(),device)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__=="__main__":
    main()
