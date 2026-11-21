"""Check the corrected KV latent RMSNorm against the original checkpoint KV-A trace."""
import csv
import sys
from pathlib import Path

import torch

from projections import compare, fp32_file

root, target, output = map(Path, sys.argv[1:])
torch.set_num_threads(2)
with output.open("w") as f:
    w=csv.writer(f,lineterminator="\n")
    w.writerow(("dtype","prefill","request","position","norm","stage","elements","mismatches",
                "nonfinite","max_abs","max_rel","first_mismatch","max_ulp","bit_differences",
                "first_bit_difference","expected_bits","device_bits","first_ulp"))
    for dtype,length,trace in map(str.split, (target/"trace_manifest").read_text().splitlines()):
        low=torch.bfloat16 if dtype=="bf16" else torch.float16
        gamma=torch.from_file(str(root/"runtime-full-fixtures/gamma.bin"),size=512,dtype=torch.int16)
        gamma=gamma.view(torch.bfloat16).to(low)
        reference=root/("p0-torch" if length=="2" else f"p0-torch-{dtype}-{length}")
        for req in range(2):
            for pos in (0,int(length)):
                prefix=f"{req}_{pos}_"
                x=fp32_file(reference/f"{prefix}kv_a.bin",576)[:,:512].contiguous()
                scale=(x.square().mean(-1,keepdim=True)+1e-6).rsqrt()
                want=((x*scale).to(low)*gamma).float()
                got=fp32_file(Path(trace)/f"{prefix}kv_norm.bin",512)
                w.writerow((dtype,length,req,pos,"latent","revised_device_only_final",*compare(want,got,low)))
