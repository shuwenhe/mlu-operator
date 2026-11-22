"""Read rank-local checkpoint slices and verify exact reconstruction on the host."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .shards import column_spans, row_span


PROJECTIONS = {
    "model.layers.0.self_attn.q_proj.weight": 0,
    "model.layers.0.self_attn.kv_b_proj.weight": 0,
    "model.layers.0.self_attn.o_proj.weight": 1,
    "model.layers.0.mlp.gate_proj.weight": 0,
    "model.layers.0.mlp.up_proj.weight": 0,
    "model.layers.0.mlp.down_proj.weight": 1,
}


def checksum(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_local(model, tp, rank):
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    for name, axis in PROJECTIONS.items():
        with safe_open(model / index[name], framework="pt", device="cpu") as f:
            data = f.get_slice(name)
            shape = data.get_shape()
            offset, width = row_span(shape[axis], rank, tp)
            shard = data[offset:offset + width] if axis == 0 else data[:, offset:offset + width]
        yield name, axis, shape, offset, width, shard


def verify(model, tp, device_records=None):
    by_rank = [dict((item[0], item[1:]) for item in load_local(model, tp, rank)) for rank in range(tp)]
    if device_records is not None:
        for rank, parts in enumerate(by_rank):
            for name, (_, _, _, _, shard) in parts.items():
                if device_records[rank, name] != checksum(shard):
                    raise AssertionError(f"MLU shard differs from checkpoint: {rank=}, {name}")
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    for name, axis in PROJECTIONS.items():
        reconstructed = torch.cat([parts[name][-1] for parts in by_rank], dim=axis)
        with safe_open(model / index[name], framework="pt", device="cpu") as f:
            original = f.get_tensor(name)
            if checksum(reconstructed) != checksum(original):
                raise AssertionError(f"Checkpoint reconstruction mismatch: {name}, TP={tp}")
    first = "model.layers.0.mlp."
    gate, up = first + "gate_proj.weight", first + "up_proj.weight"
    with safe_open(model / index[gate], framework="pt", device="cpu") as f:
        original = torch.cat([f.get_tensor(gate), f.get_tensor(up)])
    rebuilt = torch.empty_like(original)
    for rank, parts in enumerate(by_rank):
        gate_local, up_local = parts[gate][-1], parts[up][-1]
        spans = column_spans((original.shape[0] // 2,) * 2, rank, tp)
        packed = torch.cat([gate_local, up_local])
        for (offset, width), section in zip(spans, (gate_local, up_local)):
            rebuilt[offset:offset + width].copy_(section)
        expected = torch.cat([original[offset:offset + width] for offset, width in spans])
        if checksum(packed) != checksum(expected):
            raise AssertionError(f"Packed gate/up mismatch: rank {rank}, TP={tp}")
    if checksum(rebuilt) != checksum(original):
        raise AssertionError("Packed gate/up reconstruction mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    device_records = None
    if args.evidence is not None:
        rows = [json.loads(line) for line in args.evidence.read_text().splitlines()
                if line.startswith('{"kind": "weight",')]
        device_records = {(row["rank"], row["parameter"]): row["sha256"] for row in rows}
        if len(rows) != 8 * len(PROJECTIONS) or len(rows) != len(device_records):
            raise AssertionError("Expected one checkpoint shard per rank and parameter")
    for tp in (1, 2, 4, 8):
        verify(args.model, tp, device_records if tp == 8 else None)
        print(f"CHECKPOINT_RECONSTRUCTION PASS TP={tp}", flush=True)
    if device_records is not None:
        print("CHECKPOINT_MLU_SHA256 PASS 48/48", flush=True)
