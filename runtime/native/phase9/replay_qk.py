"""Compare device scores against eager PyTorch on matched Layer 0 Q/K."""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch


def compare(stage, ref, actual, expected_mask=None):
    keep = np.ones(ref.shape, dtype=bool) if expected_mask is None else ~expected_mask
    valid = keep & np.isfinite(ref) & np.isfinite(actual)
    error = np.abs(actual[valid].astype(np.float64) - ref[valid])
    magnitude = np.abs(ref[valid].astype(np.float64))
    unexpected = (keep & (~np.isfinite(actual) | ~np.isfinite(ref))).sum()
    bad_mask = 0 if expected_mask is None else (~np.isneginf(actual[expected_mask])).sum()
    mismatches = (error > 1e-5 + 1e-5 * magnitude).sum() + unexpected + bad_mask
    print(f"{stage} shape={ref.shape} max_abs={error.max(initial=0):.9g} "
          f"max_rel={(error / np.maximum(magnitude, 1e-30)).max(initial=0):.9g} "
          f"unexpected_nonfinite={unexpected} bad_mask={bad_mask} "
          f"bit_differences={(ref.view(np.uint32) != actual.view(np.uint32)).sum()} "
          f"mismatches={mismatches}", flush=True)
    return int(mismatches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    yarn = config["rope_scaling"]
    mscale = 1 + 0.1 * yarn["mscale_all_dim"] * math.log(yarn["factor"])
    scale = 192 ** -0.5 * mscale * mscale
    torch.set_num_threads(2)
    failures = 0
    for path in args.captures:
        base = int(path.name.split("_")[1])
        with np.load(path) as capture:
            q = torch.from_numpy(capture["q"])
            k = torch.from_numpy(capture["k"])
        seq, length = q.shape[1], k.shape[1]
        positions = np.arange(seq, dtype=np.uint32)
        mask = torch.arange(length)[None, :] > (torch.arange(seq) + base)[:, None]
        shape = (16, seq, length)
        size = math.prod(shape)
        for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            q_low, k_low = q.to(dtype), k.to(dtype)
            raw_ref = q_low.float() @ k_low.float().transpose(-1, -2)
            scaled_ref = (q_low @ k_low.transpose(-1, -2)) * scale
            masked_ref = scaled_ref + torch.zeros_like(scaled_ref).masked_fill(mask, -torch.inf)
            inputs = b"".join(x.contiguous().view(torch.int16).numpy().tobytes()
                              for x in (q_low, k_low)) + positions.tobytes()
            result = subprocess.run(
                [str(args.binary), args.device, name, str(seq), str(length), str(base), str(scale)],
                input=inputs, stdout=subprocess.PIPE, check=True).stdout
            if len(result) != size * 10:
                raise ValueError(f"Unexpected output size: {len(result)}")
            raw = np.frombuffer(result, np.float32, size).reshape(shape)
            scaled_bits = np.frombuffer(result, np.int16, size, 4 * size).copy()
            scaled = torch.from_numpy(scaled_bits).view(dtype).float().numpy().reshape(shape)
            masked = np.frombuffer(result, np.float32, size, 6 * size).reshape(shape)
            print(f"{path.name} dtype={name} q={tuple(q.shape)} k={tuple(k.shape)}", flush=True)
            compare("raw_fp32_diagnostic", raw_ref.numpy(), raw)
            failures += compare("scaled", scaled_ref.float().numpy(), scaled)
            failures += compare("masked", masked_ref.float().numpy(), masked,
                                mask.numpy()[None, :, :].repeat(16, axis=0))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
