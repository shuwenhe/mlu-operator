"""Prepare matched RMSNorm operands and compare measured device intermediates."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from projections import fp32_file, compare, word_array

mode, w, candidate, stage = sys.argv[1:]
w, candidate, stage = map(Path, (w, candidate, stage))
torch.set_num_threads(2)
cases = {
    ("fp16", "7"): ("input",),
    ("fp16", "32"): ("input",),
    ("fp16", "128"): ("input",),
    ("fp16", "1"): ("post",),
    ("bf16", "32"): ("post",),
}
report = csv.writer(sys.stdout)
report.writerow(("dtype","prefill","request","position","norm","stage","elements","mismatches",
                 "nonfinite","max_abs","max_rel","first_mismatch","max_ulp","bit_differences",
                 "first_bit_difference","expected_bits","device_bits","first_ulp"))
manifest = []
for line in (candidate / "trace_manifest").read_text().splitlines():
    dtype, length, path = line.split()
    low = torch.float16 if dtype == "fp16" else torch.bfloat16
    for kind in cases.get((dtype, length), ()):
        gamma = torch.from_file(str(w/f"{kind}_gamma.bin"),size=2048,dtype=torch.int16).view(torch.bfloat16).to(low)
        for request in range(2):
            for position in (0, int(length)):
                rows = int(length) if position == 0 else 1
                name = f"{dtype}-{length}-{request}-{position}-{kind}"
                p = stage / name
                x = fp32_file(Path(path)/f"{request}_{position}_{'original_hidden' if kind == 'input' else 'residual'}.bin", 2048).to(low)
                if mode == "prep":
                    word_array(x).tofile(str(p)+".in.u16")
                    word_array(gamma).tofile(str(p)+".gamma.u16")
                    manifest.append(f"{dtype} {rows} {name}")
                    continue
                def load(suffix, typ):
                    return torch.from_numpy(np.fromfile(str(p)+suffix, typ))
                square = x.float().pow(2)
                mean = square.mean(-1)
                eps = mean + 1e-6
                inverse = eps.rsqrt()
                normalized = x.float() * inverse[:, None]
                rounded = normalized.to(low)
                stats = load(".stats.f32", "<f4").reshape(rows,4)
                parts = [
                    ("input", x.float(), load(".in.u16","<i2").view(low).float(), low),
                    ("square", square, load(".squares.f32","<f4"), None),
                    ("sum", square.sum(-1), stats[:,0], None),
                    ("mean", mean, stats[:,1], None),
                    ("epsilon", eps, stats[:,2], None),
                    ("rsqrt", inverse, stats[:,3], None),
                    ("rsqrt_matched_epsilon", stats[:,2].rsqrt(), stats[:,3], None),
                    ("normalized", normalized, load(".normalized.f32","<f4"), None),
                    ("pre_gamma", rounded.float(), load(".rounded.u16","<i2").view(low).float(), low),
                    ("gamma_matched_input", (load(".rounded.u16","<i2").view(low).reshape(rows,2048)*gamma).float(),
                     load(".out.u16","<i2").view(low).float(),low),
                    ("final", (rounded*gamma).float(), load(".out.u16","<i2").view(low).float(),low),
                ]
                for label, a, b, logical in parts:
                    values = list(compare(a,b,low))
                    if logical is None:
                        aw=a.contiguous().flatten().numpy().view(np.uint32)
                        bw=b.contiguous().flatten().numpy().view(np.uint32)
                        delta=np.flatnonzero(aw!=bw)
                        values[6]=int(np.max(np.abs(aw.astype(np.int64)-bw.astype(np.int64)),initial=0))
                        values[7]=len(delta)
                        values[8:11]=[int(delta[0]),f"{aw[delta[0]]:08x}",f"{bw[delta[0]]:08x}"] if len(delta) else ["","",""]
                        first=values[5]
                        values[11]=abs(int(aw[first])-int(bw[first])) if first is not None else ""
                    report.writerow((dtype,length,request,position,kind,label,*values))
if mode == "prep":
    (stage/"norm_manifest").write_text("\n".join(manifest)+"\n")
