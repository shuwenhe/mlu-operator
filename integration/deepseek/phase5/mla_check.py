"""Standalone MLA subgraph check on one card. No server or model imports."""
import csv
import math
import struct
import sys
from pathlib import Path

import torch
import torch_mlu  # noqa: F401: registers the MLU device


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--authorized":
        return 2
    if torch.mlu.device_count() != 1:
        return 2
    source = Path(sys.argv[2])
    data = bytearray(source.read_bytes())
    if struct.unpack_from("<II", data) != (4096,512):
        raise ValueError("KV-B checkpoint shape mismatch")
    weight = torch.frombuffer(data, dtype=torch.bfloat16, offset=8).clone()
    weight = weight.reshape(16,256,512).float()
    scale = (192 ** -0.5) * (1 + 0.1 * 0.707 * math.log(40))**2
    random = torch.Generator().manual_seed(512)
    writer = csv.writer(sys.stdout)
    writer.writerow(("test","dtype","queries","cache","elements","max_abs","max_rel","mean_abs","mismatches"))
    passed = True
    for dtype in (torch.float16, torch.bfloat16):
        for query, cache in ((7,7),(1,65),(3,65)):
            # Normalized latents and already-rotated PE enter this fixture.
            latent = (torch.randn(cache,512,generator=random)*0.125).to(dtype)
            q_nope = (torch.randn(16,query,128,generator=random)*0.125).to(dtype)
            q_pe = (torch.randn(16,query,64,generator=random)*0.125).to(dtype)
            k_pe = (torch.randn(1,cache,64,generator=random)*0.125).to(dtype)
            mask = torch.arange(cache)[None,:] > (torch.arange(query)+cache-query)[:,None]
            w = weight.to(dtype)
            def attention(device, low_precision):
                z, q, qr, kr, weights, causal = [
                    t.to(device) for t in (latent,q_nope,q_pe,k_pe,w,mask)]
                z,q,qr,kr,weights = [t.float() for t in (z,q,qr,kr,weights)]
                materialized = z @ weights.transpose(-1,-2)
                if low_precision:
                    materialized = materialized.to(dtype).float()
                k,v = materialized.split(128,dim=-1)
                qk = torch.cat((q,qr),dim=-1) @ torch.cat((k,kr.expand(16,-1,-1)),dim=-1).transpose(-1,-2)
                if low_precision:
                    qk=qk.to(dtype).float()
                    scaled=(qk*scale).to(dtype).float()
                else:
                    scaled=qk*scale
                materialized_prob = scaled.masked_fill(causal,float("-inf")).softmax(-1)
                if low_precision:
                    materialized_prob=materialized_prob.to(dtype).float()
                    return (materialized_prob @ v).to(dtype).float().cpu(), None
                wk,wv = weights.split(128,dim=1)
                absorbed = ((q @ wk) @ z.T + qr @ kr.transpose(-1,-2))*scale
                prob = absorbed.masked_fill(causal,float("-inf")).softmax(-1)
                return (materialized_prob @ v).cpu(), ((prob @ z) @ wv.transpose(-1,-2)).cpu()

            def check(name, oracle, observed, atol, rtol):
                delta = (oracle-observed).abs()
                bad=(~observed.isfinite()) | (delta > atol + rtol*oracle.abs())
                writer.writerow((name,str(dtype),query,cache,delta.numel(),float(delta.max()),
                                 float((delta/oracle.abs().clamp_min(1e-6)).max()),
                                 float(delta.mean()),int(bad.sum())))
                return not bool(bad.any())
            cpu, _ = attention("cpu", True)
            device, _ = attention("mlu:0", True)
            passed &= check("materialized_model_order",cpu,device,1e-5,1e-5)
            cpu, _ = attention("cpu", False)
            materialized, compressed = attention("mlu:0", False)
            passed &= check("materialized_fp32",cpu,materialized,1e-5,1e-5)
            passed &= check("absorbed_latent_fp32",cpu,compressed,1e-5,1e-5)
    torch.mlu.synchronize()
    return not passed

if __name__ == "__main__":
    sys.exit(main())
