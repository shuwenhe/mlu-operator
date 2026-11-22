"""Compare recorded production tensors to immutable and supplemental targets."""
import array
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bf16(value):
    return struct.unpack("<f", struct.pack("<I",value << 16))[0]


def inverse(value):
    word=struct.unpack("<I",struct.pack("<f",value))[0]
    return (word+0x7fff+((word>>16)&1))>>16


def residual(left,right):
    a,b=array.array("H"),array.array("H")
    a.frombytes(left); b.frombytes(right)
    return array.array("H",(inverse(bf16(x)+bf16(y)) for x,y in zip(a,b))).tobytes()


def values(data,dtype):
    bits = array.array("H" if dtype=="bf16" else "I")
    bits.frombytes(data)
    if dtype=="bf16":
        return bits,[bf16(i) for i in bits]
    decoded=array.array("f")
    decoded.frombytes(data)
    return bits,decoded


def measure(actual,target,dtype):
    ai,a=values(actual,dtype)
    bi,b=values(target,dtype)
    if not a or len(a)!=len(b):
        raise ValueError(f"Shape mismatch: {len(a)} vs {len(b)}")
    pivot=1<<(15 if dtype=="bf16" else 31)

    def order(i):
        return pivot-(i-pivot) if i & pivot else pivot+i

    errors=[abs(x-y) for x,y in zip(a,b)]
    bad=[i for i,(x,y,e) in enumerate(zip(a,b,errors))
         if not math.isfinite(x) or not math.isfinite(y) or e>1e-5+1e-5*abs(y)]
    return dict(dtype=dtype,elements=len(a),max_abs=max(errors),
                max_rel=max(e/max(abs(y),1e-30) for e,y in zip(errors,b)),
                mean_abs=sum(errors)/len(a),max_ulp=max(abs(order(x)-order(y)) for x,y in zip(ai,bi)),
                bitwise_mismatch=sum(x!=y for x,y in zip(ai,bi)),tolerance_mismatch=len(bad),
                first_mismatch=bad[0] if bad else "",nan=sum(math.isnan(x) for x in a),
                inf=sum(math.isinf(x) for x in a),fixture_sha256=sha(target),output_sha256=sha(actual))


def main(evidence,frozen):
    capture=evidence/"tp2_retry/tp2_dump"
    supplemental=evidence/"tp1_supplemental"
    rows=[]
    missing=[]

    def check(rank,step,layer,boundary,actual,target,dtype,local=False):
        if not actual.exists():
            missing.append(str(actual))
            return
        if local:
            half=len(target)//2
            target=target[rank*half:(rank+1)*half]
        rows.append(dict(rank=rank,device=f"mlu:{rank}",step=step,layer=layer,boundary=boundary,
                         local=local,**measure(actual.read_bytes(),target,dtype)))

    for rank in (0,1):
        for step in (0,1):
            prefix=capture/f"rank{rank}_forward{step}"
            emb=(frozen/f"{step}-embedding.bin").read_bytes()
            check(rank,step,-1,"embedding",prefix.with_name(prefix.name+"_embedding.bin"),emb,"bf16")
            check(rank,step,-1,"logits",capture/f"rank{rank}_logits{step}.bin",
                  (frozen/f"{step}-logits.bin").read_bytes(),"fp32")
            for layer in range(27):
                stem=prefix.with_name(prefix.name+f"_{layer}")
                inp=emb if layer==0 else (frozen/f"{step}-layer{layer-1}.bin").read_bytes()
                att=(frozen/f"{step}-layer{layer}-attention.bin").read_bytes()
                for name,target,local in (
                        ("key",(frozen/f"{step}-layer{layer}-key.bin").read_bytes(),True),
                        ("value",(frozen/f"{step}-layer{layer}-value.bin").read_bytes(),True),
                        ("attention",att,False),("residual",residual(inp,att),False),
                        ("hidden",(frozen/f"{step}-layer{layer}.bin").read_bytes(),False)):
                    check(rank,step,layer,name,stem.with_name(stem.name+f"_{name}.bin"),target,"bf16",local)
                for name,dtype,local in (
                        ("input_RMSNorm","bf16",False),("Q","fp32",True),("KV_A","fp32",False),
                        ("KV_A_RMSNorm","bf16",False),("KV_B","fp32",True),("rope_q","bf16",True),
                        ("qk","fp32",True),("softmax","fp32",True),("av","bf16",True),
                        ("ordered_fp32","fp32",False)):
                    ref=supplemental/f"forward{step}_{layer}_{name}.bin"
                    if ref.exists():
                        check(rank,step,layer,name,stem.with_name(stem.name+f"_{name}.bin"),ref.read_bytes(),dtype,local)
                    else:
                        missing.append(str(ref))
    if rows:
        with (evidence/"tp2_accuracy.csv").open("w",newline="") as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups={}
    for row in rows:
        if not row["local"]:
            groups.setdefault((row["step"],row["layer"],row["boundary"]),set()).add(row["output_sha256"])
    counts=dict(rows=len(rows),missing=len(missing),tolerance_mismatches=sum(r["tolerance_mismatch"] for r in rows),
                replicated_hash_disagreements=sum(len(v)!=1 for v in groups.values()))
    print(json.dumps(counts),flush=True)
    if missing:
        (evidence/"missing_boundaries.txt").write_text("\n".join(missing)+"\n")
    return bool(counts["missing"] or counts["tolerance_mismatches"] or counts["replicated_hash_disagreements"])


if __name__=="__main__":
    raise SystemExit(main(*map(Path,sys.argv[1:3])))
