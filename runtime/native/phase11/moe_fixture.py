"""Pinned PyTorch matched-input reference for real per-layer MoE tensors."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
from audit import load_headers


def main(root, baseline, out, layer, trace=None):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    headers = load_headers(root)
    prefix = f"model.layers.{layer}.mlp."
    with (out/"weights.txt").open("w") as manifest:
        for name, (file, offset, shape) in headers.items():
            if name.startswith(prefix):
                print(name[len(prefix):], file.name, offset, *shape, file=manifest)

    def weights(suffix):
        shard, offset, shape = headers[prefix+suffix]
        return torch.from_numpy(np.memmap(shard, dtype="<u2", mode="c", offset=offset,
                                          shape=tuple(shape))).view(torch.bfloat16)

    def expert(x, name):
        gate = F.linear(x, weights(name+".gate_proj.weight"))
        up = F.linear(x, weights(name+".up_proj.weight"))
        return F.linear(F.silu(gate)*up, weights(name+".down_proj.weight"))

    if trace is None:
        inputs = [(rows, torch.from_numpy(np.fromfile(
            baseline/f"p0-torch-bf16-{rows}"/"0_0_final.bin",dtype=np.float32))
            .reshape(rows,2048).bfloat16()) for rows in (1,7)]
    else:
        read = lambda stage: torch.from_numpy(np.fromfile(trace/f"0-{stage}.bin",dtype="<i2")).view(torch.bfloat16).view(-1,2048)
        residual = read(f"layer{layer-1}") + read(f"layer{layer}-attention")
        shard, offset, shape = headers[f"model.layers.{layer}.post_attention_layernorm.weight"]
        gamma = torch.from_numpy(np.memmap(shard,dtype="<u2",mode="c",offset=offset,
                                         shape=tuple(shape))).view(torch.bfloat16)
        wide = residual.float()
        h = (wide * (wide.square().mean(-1,keepdim=True)+1e-6).rsqrt()).bfloat16()*gamma
        inputs = [(h.shape[0],h)]
    for rows,h in inputs:
        logits = F.linear(h.float(), weights("gate.weight").float())
        probs = logits.softmax(-1)
        routed_weights, ids = probs.topk(6,dim=-1,sorted=False)
        routed = torch.empty((rows,6,2048),dtype=torch.bfloat16)
        for expert_id in ids.unique().tolist():
            row, selected = (ids==expert_id).nonzero(as_tuple=True)
            routed[row,selected] = expert(h[row],f"experts.{expert_id}")
        shared = expert(h,"shared_experts")
        result = (routed.float()*routed_weights[:,:,None]).sum(dim=1).bfloat16()+shared
        for name, value in (("input",h),("logits",logits),("prob",probs),("weights",routed_weights),
                            ("ids",ids.int()),("routed",routed),("shared",shared),("output",result)):
            if value.dtype==torch.bfloat16:
                value=value.view(torch.int16)
            value.contiguous().numpy().tofile(out/f"{name}-{rows}.bin")


if __name__=="__main__":
    layer = int(sys.argv[4]) if len(sys.argv)>4 else 1
    if not 1 <= layer < 27:
        raise ValueError("MoE layer must be in 1–26")
    main(*map(Path,sys.argv[1:4]), layer, Path(sys.argv[5]) if len(sys.argv)>5 else None)
