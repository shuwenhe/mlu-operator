"""High precision diagnostics for observed low-precision FFN disagreements."""
import csv
import math
import struct
import sys
from pathlib import Path
from array import array

def floats(path):
    a=array("f");a.frombytes(path.read_bytes());return a
def bf16(word):
    return struct.unpack("<f",struct.pack("<I",word<<16))[0]
def round_bf16(value):
    word,=struct.unpack("<I",struct.pack("<f",value))
    return (word+0x7fff+((word>>16)&1))>>16
def bf_bits(value):
    return struct.unpack("<I",struct.pack("<f",value))[0]>>16

def main():
    fixture,native,ref=map(Path,sys.argv[1:])
    output=csv.writer(sys.stdout)
    output.writerow(("stage","request","position","flat_index","same_inputs",
                     "expected_bits","actual_bits","accurate_bf16_bits","accurate_fp64"))
    for stage, width, n, input_name in (
        ("gate",2048,10944,"post_norm"),("up",2048,10944,"post_norm"),
        ("down",10944,2048,"swiglu")):
        w=array("H");w.frombytes((fixture/f"{stage}.bin").read_bytes())
        assert len(w)==width*n
        for req in range(2):
            for pos in (0,2):
                prefix=f"{req}_{pos}_"
                x,y=[floats(p/f"{prefix}{input_name}.bin") for p in (ref,native)]
                a,b=[floats(p/f"{prefix}{stage}.bin") for p in (ref,native)]
                for i,(want,got) in enumerate(zip(a,b)):
                    if want==got:continue
                    row,col=divmod(i,n)
                    start=row*width
                    same=x[start:start+width]==y[start:start+width]
                    exact=math.fsum(float(x[start+k])*bf16(w[col*width+k]) for k in range(width))
                    output.writerow((stage,req,pos,i,int(same),f"{bf_bits(want):04x}",
                                     f"{bf_bits(got):04x}",f"{round_bf16(exact):04x}",repr(exact)))

if __name__=="__main__":
    main()
