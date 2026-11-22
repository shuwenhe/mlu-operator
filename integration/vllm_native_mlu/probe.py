"""Run the vLLM CustomOp on a real MLU stream with an independent CPU reference."""

import csv
from pathlib import Path
import sys

import torch
import torch_mlu

from integration.vllm_native_mlu import NativeRMSNorm, NativeRouter
from vllm.config import VllmConfig, set_current_vllm_config


def main(library, report, checkpoint=None, fixtures=None, router_fixtures=None):
    with set_current_vllm_config(VllmConfig()):
        op = NativeRMSNorm(library)
        router = NativeRouter(library) if router_fixtures else None
    stream = torch_mlu.mlu.Stream(device=0)
    print(f"device={torch_mlu.mlu.get_device_name(0)} queue={stream.mlu_stream:#x}", flush=True)
    assert torch_mlu.mlu.get_device_name(0) == "MLU590-M9"
    rng = torch.Generator().manual_seed(590)
    with open(report, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["case", "elements", "bit_mismatch", "tolerance_mismatch", "max_abs",
                         "max_rel", "max_ulp", "unexpected_nonfinite", "status"])
        failed = False
        cases = []
        for rows, width in ((1, 512), (7, 2048)):
            x = torch.randn(rows, width, generator=rng).bfloat16()
            gamma = torch.randn(width, generator=rng).bfloat16()
            cases.append((f"synthetic_rmsnorm_{rows}x{width}", x, gamma, None))
        if checkpoint and fixtures:
            from safetensors import safe_open

            from json import loads
            index = loads((Path(checkpoint) / "model.safetensors.index.json").read_text())
            name = "model.norm.weight"
            with safe_open(Path(checkpoint) / index["weight_map"][name],
                           framework="pt", device="cpu") as weights:
                gamma = weights.get_tensor(name)
            for rows in (1, 5):
                tensors = []
                for stage in ("embedding", "normalized"):
                    path = Path(fixtures) / f"{stage}-{rows}.bin"
                    bits = torch.from_file(str(path), size=rows * 2048, dtype=torch.int16)
                    tensors.append(bits.view(torch.bfloat16).reshape(rows, 2048))
                cases.append((f"checkpoint_rmsnorm_{rows}x2048", tensors[0], gamma, tensors[1]))
        for name, x, gamma, checkpoint_reference in cases:
            eps = 1e-6
            fp32 = x.float()
            reference = (fp32 * (fp32.square().mean(-1, keepdim=True) + eps).rsqrt()).bfloat16() * gamma
            if checkpoint_reference is not None:
                if not torch.equal(reference.view(torch.int16), checkpoint_reference.view(torch.int16)):
                    raise RuntimeError("Independent reference disagrees with the Phase 11 checkpoint fixture")
                reference = checkpoint_reference
            with torch_mlu.mlu.stream(stream):
                actual = op(x.to("mlu:0"), gamma.to("mlu:0"), eps).cpu()
            error = (actual.float() - reference.float()).abs()
            relative = error / reference.float().abs().clamp_min(torch.finfo(torch.float32).tiny)
            bits_a, bits_r = (t.view(torch.int16).int() for t in (actual, reference))
            ordered = lambda bits: torch.where(bits < 0, -32768 - bits, bits)
            ulp = (ordered(bits_a) - ordered(bits_r)).abs()
            bad = int((error > 1e-5 + 1e-5 * reference.float().abs()).sum())
            nonfinite = int((~actual.isfinite()).sum())
            failed |= bool(bad or nonfinite)
            writer.writerow([name, x.numel(),
                             int((bits_a != bits_r).sum()), bad, float(error.max()),
                             float(relative.max()), int(ulp.max()), nonfinite,
                             "FAIL" if bad or nonfinite else "PASS"])
        if router is not None:
            torch.set_num_threads(2)
            for folder in sorted(Path(router_fixtures).glob("layer*-m*")):
                rows = int(folder.name.rsplit("-m", 1)[1])

                def fixture(name, shape, dtype):
                    return torch.from_file(str(folder / name), dtype=dtype,
                                           size=torch.Size(shape).numel()).reshape(shape)

                x = fixture("input.bin", (rows, 2048), torch.int16).view(torch.bfloat16)
                w = fixture("weights.bin", (64, 2048), torch.int16).view(torch.bfloat16)
                reference = fixture("reference.bin", (rows, 64), torch.float32)
                fp32_reference = torch.nn.functional.linear(x.float(), w.float())
                if not torch.equal(fp32_reference.view(torch.int32), reference.view(torch.int32)):
                    raise RuntimeError(f"Independent reference disagrees with {folder.name}")
                with torch_mlu.mlu.stream(stream):
                    actual = router(x.mlu(), w.mlu()).cpu()
                error = (actual.double() - reference.double()).abs()
                relative = error / reference.double().abs().clamp_min(torch.finfo(torch.float32).tiny)
                actual_bits, ref_bits = (t.view(torch.int32).long() for t in (actual, reference))
                ranks = [torch.where(t < 0, -2147483648 - t, t) for t in (actual_bits, ref_bits)]
                bad = int((error > 1e-5 + 1e-5 * reference.abs()).sum())
                nonfinite = int((~actual.isfinite()).sum())
                failed |= bool(bad or nonfinite)
                writer.writerow([f"router_{folder.name}", actual.numel(),
                                 int((actual_bits != ref_bits).sum()), bad, float(error.max()),
                                 float(relative.max()), int((ranks[0] - ranks[1]).abs().max()),
                                 nonfinite, "FAIL" if bad or nonfinite else "PASS"])
    if failed:
        raise SystemExit("Native CustomOp tolerance failure")


if __name__ == "__main__":
    main(*sys.argv[1:])
