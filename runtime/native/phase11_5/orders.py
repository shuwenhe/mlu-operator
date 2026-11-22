"""Screen FP32 evaluation orders against the pinned router captures."""
import csv
from pathlib import Path
import sys

import numpy as np


def partials(p, lanes):
    a = np.zeros((*p.shape[:2],lanes), dtype=np.float32)
    for base in range(0,p.shape[2],lanes):
        np.add(a,p[:,:,base:base+lanes],out=a)
    return a


def fold(a, order):
    if order=="sequential":
        v = a[:,:,0].copy()
        for pos in range(1,a.shape[2]):
            v+=a[:,:,pos]
        return v
    while a.shape[2]>1:
        if order=="adjacent":
            a=a[:,:,::2]+a[:,:,1::2]
        else:
            half=a.shape[2]//2
            a=a[:,:,:half]+a[:,:,half:]
    return a[:,:,0]


def existing(p):
    a=partials(p,64)
    b=((a[:,:,:8]+a[:,:,32:40])+(a[:,:,16:24]+a[:,:,48:56]))+((a[:,:,24:32]+a[:,:,56:64])+(a[:,:,8:16]+a[:,:,40:48]))
    return fold(b,"split")


def panel112(p):
    parts = np.empty((*p.shape[:2], 19), dtype=np.float32)
    for i, start in enumerate(range(0, 2048, 112)):
        parts[..., i] = partials(p[..., start:start + 112], 1)[..., 0]
    total = parts[..., -1].copy()
    for i in range(18):
        total += parts[..., i]
    return total, parts


def gemv_prefix4(p):
    sums = np.zeros((*p.shape[:2], 4), dtype=np.float32)
    for k in range(4):
        sums[..., 0] += p[..., k]
    for k in range(4, 2048, 4):
        sums += p[..., k:k + 4]
    return fold(sums, "split"), sums


def compare(writer,case,name,value,ref):
    error=np.abs(value.astype(np.float64)-ref)
    rank=lambda x: np.where(x<0,-(x.view(np.int32).astype(np.int64)&0x7fffffff),
                            x.view(np.int32).astype(np.int64))
    changed=np.argwhere(value.view(np.uint32)!=ref.view(np.uint32))
    first=tuple(changed[0]) if changed.size else None
    writer.writerow((case,name,value.size,len(changed),int((error>1e-5+1e-5*np.abs(ref)).sum()),
                     error.max(),(error/np.maximum(np.abs(ref),1e-30)).max(),
                     np.max(np.abs(rank(value)-rank(ref))),
                     "" if first is None else f"{first[0]}:{first[1]}",
                     "" if first is None else hex(int(value.view(np.uint32)[first])),
                     "" if first is None else hex(int(ref.view(np.uint32)[first]))))


def main(root):
    writer=csv.writer(sys.stdout,lineterminator="\n")
    writer.writerow(("case","strategy","elements","bit_differences","tolerance_mismatches",
                     "max_abs","max_rel","max_ulp","first_index","actual_bits","reference_bits"))
    for case in sorted(root.glob("layer*-m*")):
        wid=lambda p: (np.fromfile(p,dtype="<u2").astype(np.uint32)<<16).view(np.float32)
        a=wid(case/"input.bin").reshape(-1,2048)
        w=wid(case/"weights.bin").reshape(64,2048)
        ref=np.fromfile(case/"reference.bin",dtype="<f4").reshape(-1,64)
        p=a[:,None,:]*w[None,:,:]
        value, _ = gemv_prefix4(p)
        if len(a) == 17:
            value[:16], _ = panel112(p[:16])
        compare(writer, case.name, "mkl_sampled_order", value, ref)
        compare(writer,case.name,"packed64_host_reconstruction",existing(p),ref)
        compare(writer,case.name,"fp64_then_fp32",p.sum(axis=-1,dtype=np.float64).astype(np.float32),ref)
        for lanes in (1,2,4,8,16,32,64,128):
            sums=partials(p,lanes)
            for order in ("split","adjacent","sequential"):
                compare(writer,case.name,f"lanes{lanes}_{order}",fold(sums,order),ref)
        for prefix in (1,2,3,4,8):
            v=np.zeros_like(ref)
            for i in range(prefix):
                v+=p[:,:,i]
            sums=np.zeros((*ref.shape,4),dtype=np.float32)
            sums[:,:,0]=v
            end=prefix+(2048-prefix)//4*4
            for i in range(prefix,end,4):
                sums+=p[:,:,i:i+4]
            v=fold(sums,"split")
            for i in range(end,2048):
                v+=p[:,:,i]
            compare(writer,case.name,f"sgemv_prefix{prefix}_lanes4",v,ref)
        for chunk in (128,192,256,272,288,320,384,480,512,640,768,896,1024,1152,1280,1536):
            for lanes in (1,4,8):
                total=np.zeros_like(ref)
                for start in range(0,2048,chunk):
                    total+=fold(partials(p[:,:,start:start+chunk],lanes),"split")
                compare(writer,case.name,f"chunk{chunk}_lanes{lanes}",total,ref)


if __name__=="__main__":
    main(Path(sys.argv[1]))
