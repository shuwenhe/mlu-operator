"""Exercise native AV on checkpoint and actual native-Softmax probabilities."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "softmax"))
from acceptance import measure, native


def bytes_of(tensor):
    return tensor.contiguous().view(torch.int16).numpy().tobytes()


def replay(binary, name, p, v, padding=2):
    heads, seq, keys = p.shape
    # Poison unused capacity so an incorrect per-head stride affects the result.
    padded = torch.nn.functional.pad(v.contiguous(), (0, 0, 0, padding), value=float("nan"))
    data = native(binary, [name, heads, seq, keys, keys + padding],
                  bytes_of(p) + bytes_of(padded))
    shape = (heads, seq, 128)
    count = heads * seq * 128
    if len(data) != 6 * count:
        raise ValueError("Malformed device AV payload")
    return (torch.frombuffer(data, dtype=torch.float32, count=count).view(shape),
            torch.frombuffer(data, dtype=p.dtype, count=count, offset=4 * count).view(shape))


def report(writer, case, p, raw, out, reference_raw, reference_low):
    failures = 0
    for stage, ref, got in (("av_fp32", reference_raw, raw), ("av_lowp", reference_low, out)):
        stats = measure(ref, got)
        writer.writerow((*case, *p.shape, 128, stage, ref.numel(), *stats))
        failures += stats[1]
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qk_fixtures", "av_fixtures", "qk", "softmax", "av"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(f"reference PyTorch={torch.__version__} capability={torch.backends.cpu.get_cpu_capability()}",
          file=sys.stderr)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("dtype", "prefill", "request", "pass", "reference", "heads", "seq",
                     "cache", "width", "stage", "elements", "bit_mismatches",
                     "tolerance_mismatches", "max_abs", "max_rel", "max_ulp", "nan", "inf",
                     "unexpected_nonfinite"))
    fixtures, failures, chain_failures, synthetic = 0, 0, 0, 0
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        for length in (1, 7, 32, 128):
            pos = torch.arange(length)
            p = (0.1 + 0.01 * (pos % 10)).expand(2, length, length).clone()
            p.masked_fill_(pos[None, :] > pos[:, None], 0)
            p = (p / p.sum(-1, keepdim=True)).to(dtype)
            v = (0.1 + 0.01 * (torch.arange(2 * length * 128) % 50)).view(2, length, 128).to(dtype)
            raw, out = replay(args.av, name, p, v)
            failures += report(writer, (name, length, 0, "synthetic", "matched"),
                               p, raw, out, p.float() @ v.float(), p @ v)
            synthetic += 1
            tag = f"{name}-{length}"
            for index in range(4):
                with np.load(args.av_fixtures / tag / f"av_{index}.npz") as f:
                    p, values, out_ref = (torch.from_numpy(f[k]).view(dtype)
                                          for k in ("probs", "v", "out"))
                    v = torch.empty_strided(values.shape, tuple(f["v_strides"]), dtype=dtype)
                    v.copy_(values)
                    av_positions = f["positions"]
                with np.load(args.qk_fixtures / tag / f"qk_{index}.npz") as f:
                    q, k, pos, scale, masked = (f[key] for key in
                                                ("q", "k", "positions", "scale", "masked"))
                assert np.array_equal(av_positions, pos)
                heads, seq, keys = p.shape
                count = heads * seq * keys
                base = int(pos[0])
                expected_scores = torch.from_numpy(masked).view(dtype).float()
                assert torch.equal(torch.softmax(expected_scores, -1).to(dtype), p)
                assert torch.equal(p @ v, out_ref), (tag, index, "AV capture roundtrip")
                qk = native(args.qk, [0, name, seq, keys, base, float(scale)],
                            q.tobytes() + k.tobytes() + (pos - base).astype(np.uint32).tobytes())
                scores = torch.frombuffer(qk, dtype=torch.float32, count=count,
                                          offset=6 * count).view_as(p)
                assert torch.equal(scores.view(torch.int32), expected_scores.view(torch.int32))
                softmax = native(args.softmax, [name, heads * seq, keys, keys, 0],
                                 scores.numpy().tobytes())
                p_native = torch.frombuffer(softmax, dtype=dtype, count=count,
                                            offset=8 * heads * seq + 4 * count).view_as(p)
                case = (name, length, index // 2, "decode" if base else "prefill")
                raw, out = replay(args.av, name, p, v)
                ref_raw = p.float() @ v.float()
                failures += report(writer, (*case, "matched"), p, raw, out, ref_raw, out_ref)
                raw, out = replay(args.av, name, p_native, v)
                failures += report(writer, (*case, "native_softmax_matched"), p_native, raw, out,
                                   p_native.float() @ v.float(), p_native @ v)
                chain_delta = report(writer, (*case, "native_chain"), p_native, raw, out,
                                     ref_raw, out_ref)
                chain_failures += chain_delta
                if chain_delta:
                    same_input = p_native @ v
                    error = (out.float() - out_ref.float()).abs()
                    bad = error > 1e-5 + 1e-5 * out_ref.float().abs()
                    for head, query, channel in torch.nonzero(bad).tolist():
                        index = (head, query, channel)
                        keys_changed = torch.nonzero(p_native[head, query] != p[head, query]).flatten()
                        changes = [(key, p[head, query, key].item(),
                                    p_native[head, query, key].item()) for key in keys_changed.tolist()]
                        print(f"CHAIN_FAIL case={case} index={index} "
                              f"checkpoint={out_ref[index].item():.12g} native={out[index].item():.12g} "
                              f"matched_pytorch={same_input[index].item():.12g} "
                              f"error={error[index].item():.12g} changed_probs={changes}",
                              file=sys.stderr)
                fixtures += 1
    print(f"checkpoint_fixtures={fixtures} synthetic_cases={synthetic} "
          f"matched_mismatches={failures} chain_mismatches={chain_failures}", file=sys.stderr)
    return int(fixtures != 32 or synthetic != 8 or failures != 0 or chain_failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
