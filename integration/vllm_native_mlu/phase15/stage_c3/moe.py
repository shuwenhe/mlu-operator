"""MoE experiments that retain the frozen route, activation and combine order."""
import torch
import torch.distributed as dist

from ...phase12_4.projection import PackedProjection
from ..tp8.contracts import expert_owner
from ..tp8.model import ExpertParallelDecoder


def owner_slots(chosen, world_size):
    owners = [[] for _ in range(world_size)]
    for slot, expert in enumerate(chosen):
        owners[expert_owner(expert)].append(slot)
    return owners


def combine_slots(group, x, owners, local, compact, overlap=False):
    count = sum(map(len, owners))
    if not compact:
        routed = x.new_zeros((count, x.shape[-1]))
        for slot, value in local.items():
            routed[slot:slot + 1].copy_(value)
        if overlap:
            work = dist.all_reduce(routed, group=group.device_group, async_op=True)
            return routed, work
        return group.all_reduce(routed)
    capacity = max(map(len, owners))
    own_slots = owners[group.rank_in_group]
    operand = x.new_zeros((capacity, x.shape[-1]))
    for offset, slot in enumerate(own_slots):
        operand[offset:offset + 1].copy_(local[slot])
    received = group.all_gather(operand, dim=0)
    source = [0] * count
    for rank, slots in enumerate(owners):
        for offset, slot in enumerate(slots):
            source[slot] = rank * capacity + offset
    return received.index_select(0, torch.tensor(source, dtype=torch.long, device=x.device))


class TunableDecoder(ExpertParallelDecoder):
    def __init__(self, ops, *, compact=False, fuse=False, batch=False, overlap=False):
        super().__init__(ops)
        self.compact, self.fuse, self.batch = compact, fuse, batch
        if overlap and compact:
            raise ValueError("Overlap requires the sparse sum dispatch")
        self.overlap = overlap

    def project_gate_up(self, x, prefix):
        gate_name = prefix + "gate_proj.weight"
        up_name = prefix + "up_proj.weight"
        key = (gate_name, "gate_up", None)
        if key not in self.projections:
            gate, up = self.weights[gate_name], self.weights[up_name]
            if gate.shape != up.shape:
                raise ValueError(f"Gate and up shapes differ for {prefix}")
            self.projections[key] = PackedProjection(self.projection, torch.cat((gate, up)).to(x.device))
        return self.projections[key](x)

    def _ffn(self, prefix, x):
        if not self.fuse or ".experts." not in prefix and ".shared_experts." not in prefix:
            return super()._ffn(prefix, x)
        gate, up = self.project_gate_up(x, prefix).chunk(2, dim=1)
        gate, up = gate.contiguous(), up.contiguous()
        mid = self._empty(gate.shape)
        self._run("phase11_swiglu", gate, up, mid, mid.numel())
        return self._cast(self._project(mid, prefix + "down_proj.weight"))

    def route(self, prefix, x):
        rows = x.shape[0]
        scores = self.router(x, self._resident(prefix + "gate.weight"))
        probs = self._empty(scores.shape, torch.float32)
        low = self._empty(scores.shape)
        maximum = self._empty((rows,), torch.float32)
        denominator = self._empty((rows,), torch.float32)
        status = self.softmax(self.stream.mlu_stream, scores.data_ptr(), None, 0,
                              maximum.data_ptr(), denominator.data_ptr(), probs.data_ptr(),
                              low.data_ptr(), rows, 64, 64)
        if status:
            raise RuntimeError(f"Native router softmax status={status}")
        ids = self._empty((rows, 6), torch.int32)
        weights = self._empty((rows, 6), torch.float32)
        self._run("phase11_route", probs, ids, weights, rows)
        return ids.flatten().tolist(), weights

    def routed_outputs(self, prefix, x, chosen, slots):
        if not self.batch:
            return {
                slot: self._ffn(f"{prefix}experts.{chosen[slot]}.", x[slot // 6:slot // 6 + 1])
                for slot in slots
            }
        grouped = {}
        for slot in slots:
            grouped.setdefault(chosen[slot], []).append(slot)
        outputs = {}
        for expert, expert_slots in grouped.items():
            if len(expert_slots) == 1:
                token = expert_slots[0] // 6
                inputs = x[token:token + 1]
            else:
                tokens = torch.tensor([slot // 6 for slot in expert_slots], dtype=torch.long, device=x.device)
                inputs = x.index_select(0, tokens)
            values = self._ffn(f"{prefix}experts.{expert}.", inputs)
            for row, slot in enumerate(expert_slots):
                outputs[slot] = values[row:row + 1]
        return outputs

    def _moe(self, prefix, x):
        chosen, weights = self.route(prefix, x)
        self.ops.agree((prefix, chosen))
        group = self.ops.group
        owners = owner_slots(chosen, group.world_size)
        local = self.routed_outputs(prefix, x, chosen, owners[group.rank_in_group])
        routed = combine_slots(group, x, owners, local, self.compact, self.overlap)
        shared = self._ffn(prefix + "shared_experts.", x)
        if self.overlap:
            routed, work = routed
            work.wait()
        result = self._empty(x.shape)
        self._run("phase11_expert_combine", routed, weights, shared, result, x.shape[0])
        return result
