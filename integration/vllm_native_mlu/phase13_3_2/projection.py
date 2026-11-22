"""Rank-local DeepSeek-V2-Lite MLA projection boundaries."""
import hashlib
import json

from safetensors import safe_open
import torch

from .. import NativeRMSNorm
from ..phase13_2.linear import ColumnParallelLinear, LocalGemm, RowParallelLinear, validate_tp
from .mapping import RULES, shard


def checksum(value):
    return hashlib.sha256(value.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_weight(checkpoint, index, layer, name, rank, tp):
    full = f"model.layers.{layer}.self_attn.{name}"
    axis, offset, width = shard(name, rank, tp)
    with safe_open(checkpoint / index[full], framework="pt", device="cpu") as file:
        data = file.get_slice(full)
        if tuple(data.get_shape()) != RULES[name][0] or data.get_dtype() != "BF16":
            raise ValueError(f"Unexpected checkpoint metadata: {full}")
        if axis == 0:
            return data[offset:offset + width]
        if axis == 1:
            return data[:, offset:offset + width]
        return data[:]


class MLAProjection:
    def __init__(self, checkpoint, layer, group, device, libraries, record):
        validate_tp(group, device)
        self.group = group
        self.rank = group.rank_in_group
        self.tp = group.world_size
        self.parts = []
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        library = libraries / "projection/libprojection.so"

        def weight(name):
            host = load_weight(checkpoint, index, layer, name, self.rank, self.tp)
            local = host.to(device)
            record(name, local, shard(name, self.rank, self.tp))
            return local

        try:
            self.q = ColumnParallelLinear(weight("q_proj.weight"), (3072,), group, library)
            self.parts.append(self.q)
            self.kv_a = LocalGemm(weight("kv_a_proj_with_mqa.weight"), library)
            self.parts.append(self.kv_a)
            self.gamma = weight("kv_a_layernorm.weight")
            self.norm = NativeRMSNorm(libraries / "attention/libnative_mlu.so")
            self.kv_b = ColumnParallelLinear(weight("kv_b_proj.weight"), (4096,), group, library)
            self.parts.append(self.kv_b)
            self.o = RowParallelLinear(weight("o_proj.weight"), 2048, group, library)
            self.parts.append(self.o)
        except Exception:
            self.close()
            raise

    def query(self, hidden):
        return self.q(hidden)

    def latent(self, hidden):
        packed = self.kv_a(hidden)
        return packed[:, :512].contiguous(), packed[:, 512:].contiguous()

    def latent_norm(self, latent):
        return self.norm.forward_native(latent.to(torch.bfloat16), self.gamma, 1e-6)

    def expand(self, latent):
        return self.kv_b(latent).view(latent.shape[0], 16 // self.tp, 256)

    def output(self, local_value, residual):
        reduced = self.o(local_value, input_is_parallel=True)
        rounded = reduced.to(torch.bfloat16)
        return reduced, rounded, rounded + residual

    def close(self):
        for part in reversed(self.parts):
            part.close()
        self.parts.clear()
