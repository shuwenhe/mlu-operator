"""Compare full fixture and exact-input projections with the pinned checkpoint oracle."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from projections import PROJECTIONS, compare, fp32_file
from importlib.machinery import SourceFileLoader
_legacy=SourceFileLoader("phase82_legacy", "runtime/native/p0/compare.py").load_module()
ORDER, WIDTH = _legacy.ORDER, _legacy.WIDTH

trace, fixture, reference, output = map(Path,sys.argv[1:5])
dtype, length = sys.argv[5], int(sys.argv[6])
low=torch.bfloat16 if dtype=="bf16" else torch.float16
torch.set_num_threads(2)
with (output/"layer.csv").open("w") as f:
    out=csv.writer(f)
    out.writerow(("run_dtype","prefill","comparison","request","position","stage","shape",
                  "element_strides","logical_dtype","elements","max_abs","max_rel","mean_abs",
                  "mismatches","nonfinite","bit_differences","first_mismatch","expected_bits",
                  "actual_bits","first_bit_difference","expected_first_bits","actual_first_bits"))
    for req in range(2):
        for pos in (0,length):
            prefix=f"{req}_{pos}_"
            for stage in ORDER:
                w=WIDTH.get(stage,2048)
                a=fp32_file(reference/f"{prefix}{stage}.bin",w)
                b=fp32_file(trace/f"{prefix}{stage}.bin",w)
                assert a.shape==b.shape
                e=a.flatten().numpy()
                r=b.flatten().numpy()
                finite=np.isfinite(e)&np.isfinite(r)
                d=np.abs(e.astype(np.float64)-r)
                good=d[finite]
                failed=np.flatnonzero(~finite|(d>1e-5+1e-5*np.abs(e)))
                is_fp32=stage.endswith("fp32")
                if is_fp32:
                    eb,rb=e.view(np.uint32),r.view(np.uint32)
                else:
                    eb=a.to(low).view(torch.int16).flatten().numpy().view(np.uint16)
                    rb=b.to(low).view(torch.int16).flatten().numpy().view(np.uint16)
                diff=np.flatnonzero(eb!=rb)
                def first(delta):
                    if not len(delta):return ("","","")
                    i=delta[0]
                    bits=8 if is_fp32 else 4
                    return i,f"{eb[i]:0{bits}x}",f"{rb[i]:0{bits}x}"
                out.writerow((dtype,length,"original_source",req,pos,stage,f"1x{a.shape[0]}x{w}",
                              f"{len(e)}/{w}/1","fp32" if is_fp32 else dtype,len(e),good.max(initial=0),
                              (good/np.maximum(np.abs(e[finite]),1e-6)).max(initial=0),
                              good.mean() if len(good) else 0,len(failed),int((~finite).sum()),len(diff),
                              *first(failed),*first(diff)))
with (output/"projections.csv").open("w") as f:
    out=csv.writer(f)
    out.writerow(("dtype","prefill","request","position","projection","elements","mismatches",
                  "nonfinite","max_abs","max_rel","first_mismatch","max_ulp","bit_differences",
                  "first_bit_difference","torch_bits","device_bits","first_ulp"))
    for stage,(source,k,n) in PROJECTIONS.items():
        words=torch.from_file(str(fixture/f"{stage}.bin"),size=k*n,dtype=torch.int16)
        weight=words.view(torch.bfloat16).reshape(n,k).to(low)
        for req in range(2):
            for pos in (0,length):
                prefix=f"{req}_{pos}_"
                x=fp32_file(trace/f"{prefix}{source}.bin",k).to(low)
                want=torch.nn.functional.linear(x,weight).float()
                got=fp32_file(trace/f"{prefix}{stage}_fp32.bin",n).to(low).float()
                out.writerow((dtype,length,req,pos,stage,*compare(want,got,low)))
