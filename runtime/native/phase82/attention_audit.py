"""Matched-input comparison of the serial fixture against eager checkpoint attention."""
import ctypes
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from projections import compare, fp32_file

root = Path(sys.argv[1])
config = json.loads(Path(sys.argv[2]).read_text())
lib = ctypes.CDLL(str(root / "serial_attention.so"))
pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
lib.serial_attention.argtypes = [pointer]*3+[ctypes.c_int]*3+[pointer]*9
lib.serial_attention.restype = None
torch.set_num_threads(2)
out = csv.writer(sys.stdout,lineterminator="\n")
out.writerow(("request","position","m","n","stage","elements","mismatches","nonfinite",
              "max_abs","max_rel","first_mismatch","max_ulp","bit_differences",
              "first_bit_difference","expected_bits","fixture_bits","first_ulp"))
params = config["rope_scaling"]
scaling = 0.1 * params["mscale_all_dim"] * np.log(params["factor"]) + 1.
scale = 192 ** -0.5 * scaling * scaling
for req in range(2):
    for pos in (0,128):
        m,n = (128,128) if pos==0 else (1,129)
        pre = f"{req}_{pos}_"
        query = fp32_file(root/f"{pre}q.bin",3072).reshape(m,16,192).transpose(0,1).clone()
        kv = fp32_file(root/f"{pre}attention_cache.bin",4096).reshape(n,16,256).transpose(0,1)
        key = torch.empty(16,n,192)
        val = kv[:,:,128:].contiguous()
        rope = fp32_file(root/f"{pre}attention_rotary.bin",64).reshape(17,n,64)
        query[:,:,128:] = rope[:16,pos:pos+m]
        key[:,:,:128] = kv[:,:,:128]
        key[:,:,128:] = rope[16]
        q,k,v = (z.contiguous() for z in (query,key,val))
        ql,kl,vl = (z.bfloat16() for z in (q,k,v))
        shape = (16,m,n)
        serial = [np.empty(s,np.float32) for s in [shape]*3+[(16,m)]*2+[shape]*2+[(16,m,128)]*2]
        lib.serial_attention(q.numpy(),k.numpy(),v.numpy(),m,n,pos,*serial)
        old_qk,old_scaled,old_masked,old_max,old_den,old_p,old_p_low,old_av,old_out=map(torch.from_numpy,serial)
        traced = fp32_file(root/f"{pre}attention.bin",2048).reshape(m,16,128).transpose(0,1)
        assert torch.equal(traced,old_out), (req,pos)
        allow = torch.arange(n)[None,:]<=torch.arange(pos,pos+m)[:,None]
        mask = torch.zeros(m,n,dtype=torch.bfloat16).masked_fill(~allow,-torch.inf)
        qk = ql @ kl.transpose(-1,-2)
        scaled = qk * scale
        masked = scaled + mask
        maxes = masked.float().amax(-1)
        denom = (masked.float()-maxes[:,:,None]).exp().sum(-1)
        probs = torch.softmax(masked,dim=-1,dtype=torch.float32)
        low = probs.bfloat16()
        av = low @ vl
        matched_scale = old_qk.bfloat16()*scale
        matched_softmax = torch.softmax(old_masked,dim=-1,dtype=torch.float32)
        matched_av = old_p_low.bfloat16() @ vl
        stages = (
            ("q_input", ql.float(),q),("k_input",kl.float(),k),("v_input",vl.float(),v),
            ("qk_fp32_diagnostic",q@k.transpose(-1,-2),old_qk),
            ("qk_bf16",qk.float(),old_qk.bfloat16().float()),("scaling",scaled.float(),old_scaled),
            ("scaling_matched_qk",matched_scale.float(),old_scaled),
            ("causal_mask",torch.isneginf(masked).float(),torch.isneginf(old_masked).float()),
            ("softmax_maximum",maxes,old_max),("softmax_denominator_reconstructed",denom,old_den),
            ("probabilities_fp32",probs,old_p),("probabilities_bf16",low.float(),old_p_low),
            ("probabilities_fp32_matched_scores",matched_softmax,old_p),
            ("probabilities_bf16_matched_scores",matched_softmax.bfloat16().float(),old_p_low),
            ("av_fp32_diagnostic",low.float()@v,old_av),
            ("av_matched_probabilities",matched_av.float(),old_out),
            ("final_attention",av.float(),old_out))
        np.savez(root/f"{pre}attention_audit.npz",q=q.numpy(),k=k.numpy(),v=v.numpy(),
                 torch_qk=qk.float().numpy(),torch_scaled=scaled.float().numpy(),torch_mask=mask.float().numpy(),
                 torch_softmax_max=maxes.numpy(),torch_denominator_reconstructed=denom.numpy(),
                 torch_fp32_probabilities=probs.numpy(),torch_bf16_probabilities=low.float().numpy(),
                 torch_av=av.float().numpy(),serial_qk=old_qk.numpy(),serial_scaling=old_scaled.numpy(),
                 serial_masked=old_masked.numpy(),serial_max=old_max.numpy(),serial_den=old_den.numpy(),
                 serial_fp32_prob=old_p.numpy(),serial_bf16_prob=old_p_low.numpy(),serial_av=old_av.numpy())
        for name,a,b in stages:out.writerow((req,pos,m,n,name,*compare(a,b,torch.bfloat16)))
