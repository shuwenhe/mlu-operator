"""Capture V and AV boundaries from the original eager Layer 0 forward."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "p0"))
import checkpoint_trace as trace

model, _, output = map(Path, sys.argv[1:4])
source = model / "modeling_deepseek.py"
lines = source.read_text().splitlines()
before = next(i for i, line in enumerate(lines, 1) if
              line.strip() == "attn_output = torch.matmul(attn_weights, value_states)")
after = before + 2
pending = None
call = 0


def bits(tensor):
    return tensor.contiguous().view(trace.torch.int16).numpy().copy()


def instrument(frame, event, _arg):
    global pending, call
    if event == "call":
        if frame.f_code.co_filename == str(source) and frame.f_code.co_name == "forward":
            return instrument
        return None
    if event != "line":
        return instrument
    if frame.f_lineno == before:
        loc = frame.f_locals
        v = loc["value_states"]
        pending = dict(v=bits(v[0]), probs=bits(loc["attn_weights"][0]),
                       v_strides=np.array(v[0].stride()),
                       positions=loc["position_ids"][0].numpy().copy())
    elif frame.f_lineno == after and pending is not None:
        pending["out"] = bits(frame.f_locals["attn_output"][0])
        np.savez(output / f"av_{call}.npz", **pending)
        call += 1
        pending = None
    return instrument


sys.settrace(instrument)
try:
    trace.main()
finally:
    sys.settrace(None)
assert call == 4, call
(output / "av_provenance.json").write_text(json.dumps({
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "torch": trace.torch.__version__, "arguments": sys.argv[1:],
}, indent=2) + "\n")
