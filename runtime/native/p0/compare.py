"""Compare float carrier traces while reporting their logical storage bits."""
import csv
import math
import struct
import sys
from pathlib import Path

ORDER = ("original_hidden","input_norm","q","kv_a","kv_b","attention",
         "output_projection_fp32","output_projection","residual_fp32","residual",
         "post_norm","gate","up","swiglu","down","final")
WIDTH = {"q":3072,"kv_a":576,"kv_b":4096,"gate":10944,"up":10944,"swiglu":10944}

def values(path):
    raw=path.read_bytes()
    return [f for (f,) in struct.iter_unpack("<f",raw)]

def bits(value,dtype):
    word,=struct.unpack("<I",struct.pack("<f",value))
    if dtype=="fp32":return f"{word:08x}"
    if dtype=="bf16":return f"{word>>16:04x}"
    return struct.pack(">e",value).hex()

def main():
    native,ref=map(Path,sys.argv[1:3])
    base_dtype=sys.argv[3] if len(sys.argv)>3 else "bf16"
    prefill=int(sys.argv[4]) if len(sys.argv)>4 else 2
    output=csv.writer(sys.stdout)
    output.writerow(("run_dtype","prefill","comparison","request","position","stage","shape","element_strides",
        "logical_dtype","elements","max_abs","max_rel","mean_abs","mismatches","nonfinite",
        "bit_differences","first_mismatch","expected_bits","actual_bits",
        "first_bit_difference","expected_first_bits","actual_first_bits"))
    for req in range(2):
        for pos in (0,prefill):
            prefix=f"{req}_{pos}_"
            pairs=[("original_source", s, s) for s in ORDER]
            if (ref/f"{prefix}matched_residual.bin").exists():
                pairs += [("identical_attention_input",s,"matched_"+s)
                          for s in ("output_projection_fp32","output_projection","residual")]
                pairs += [("identical_projection_input","residual","identical_projection_residual")]
            for comparison,stage,source in pairs:
                a,b=values(ref/f"{prefix}{source}.bin"),values(native/f"{prefix}{stage}.bin")
                assert len(a)==len(b)>0
                dtype="fp32" if stage.endswith("fp32") else base_dtype
                delta=[abs(x-y) for x,y in zip(a,b)]
                ok=[math.isfinite(x) and math.isfinite(y) for x,y in zip(a,b)]
                bad=[i for i,(d,x) in enumerate(zip(delta,a)) if not ok[i] or d>1e-5+1e-5*abs(x)]
                different=[i for i,(x,y) in enumerate(zip(a,b)) if bits(x,dtype)!=bits(y,dtype)]
                def first(indices):
                    if not indices:return "", "", ""
                    i=indices[0]
                    return i,bits(a[i],dtype),bits(b[i],dtype)
                good=[d for d,valid in zip(delta,ok) if valid]
                rel=[d/max(abs(x),1e-6) for d,x,valid in zip(delta,a,ok) if valid]
                w=WIDTH.get(stage,2048)
                output.writerow((base_dtype,prefill,comparison,req,pos,stage,f"1x{len(a)//w}x{w}",
                    f"{len(a)}/{w}/1",dtype,len(a),max(good,default=math.nan),
                    max(rel,default=math.nan),sum(good)/len(good) if good else math.nan,
                    len(bad),ok.count(False),len(different),*first(bad),*first(different)))

if __name__=="__main__":
    main()
