"""Integrate selected MoE alternatives into the preserved Stage C.2 architecture."""
import json
import os
from pathlib import Path

from ..stage_c2.model import StageC2Model, WorkerProfile
from ..tp8.contracts import expert_owner
from .moe import TunableDecoder


class MoEProfile(WorkerProfile):
    def attach(self):
        super().attach()
        model = self.model
        decoder = model.decoder
        if isinstance(decoder, TunableDecoder):
            decoder.project_gate_up = self.wrap(
                "fused_expert_gate_up_gemm", decoder.project_gate_up, synchronize=True,
            )
        run = decoder._run

        def track_route(name, *args):
            result = run(name, *args)
            if self.active and name == "phase11_route":
                experts = args[1].flatten().tolist()
                owned = [expert for expert in experts if expert_owner(expert) == model.rank]
                directory = Path(f"/results/moe_workload_rank{model.rank}.jsonl")
                with directory.open("a") as file:
                    file.write(json.dumps(dict(rank=model.rank, step=model.forward_sequence,
                                               slots=len(owned), active_experts=len(set(owned)),
                                               rows=args[-1])) + "\n")
            return result

        decoder._run = track_route


class StageC3Model(StageC2Model):
    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        flags = set(os.getenv("NEURX_C3_EXPERIMENT", "").split(",")) - {""}
        if flags - {"compact", "fused", "batched", "overlap"}:
            raise ValueError(f"Unknown Stage C.3 experiment flags: {flags}")
        if flags:
            original = self.decoder
            candidate = TunableDecoder(self.ops, compact="compact" in flags, fuse="fused" in flags,
                                       batch="batched" in flags, overlap="overlap" in flags)
            candidate.weights = original.weights
            original.close()
            self.decoder = candidate
        if os.getenv("NEURX_C3_PROFILE") == "1":
            self.profile = MoEProfile(self)
            self.profile.attach()
        return loaded
