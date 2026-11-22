"""Instrument the Stage C.1 worker at native operator and collective boundaries."""
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist

from ...phase12_4.projection import PackedProjection
from ..tp8.model import NeurXDeepSeekV2ForCausalLMTP8


class WorkerProfile:
    def __init__(self, model):
        self.model = model
        self.elapsed = Counter()
        self.calls = Counter()
        self.payload = Counter()
        self.active = False
        self.section = "other"

    def wrap(self, name, operation, *, synchronize=False, collective=False):
        @wraps(operation)
        def measured(*args, **kwargs):
            if not self.active:
                return operation(*args, **kwargs)
            start = perf_counter()
            result = operation(*args, **kwargs)
            if synchronize:
                torch.mlu.current_stream(self.model.device).synchronize()
            key = f"{name}_{self.section}" if collective else name
            self.elapsed[key] += (perf_counter() - start) * 1000
            self.calls[key] += 1
            if collective:
                self.payload[key] += args[0].numel() * args[0].element_size()
            return result
        return measured

    def scope(self, name, operation):
        @wraps(operation)
        def scoped(*args, **kwargs):
            previous = self.section
            self.section = name
            try:
                return operation(*args, **kwargs)
            finally:
                self.section = previous
        return scoped

    def attach(self):
        model = self.model
        self.original_broadcast = dist.broadcast
        self.original_pack = PackedProjection.__init__
        for name in ("all_reduce", "all_gather"):
            setattr(model.group, name, self.wrap(f"cncl_{name}", getattr(model.group, name),
                                                synchronize=True, collective=True))
        native = model.ops.timed

        def timed(name, operation):
            if name in ("qk", "softmax", "av", "rope", "cache_read", "cache_write"):
                return self.wrap(name, native)(name, operation)
            return native(name, operation)

        model.ops.timed = timed
        model.validate = self.scope("metadata", self.wrap("scheduler_metadata_total", model.validate))
        for layer in model.mla:
            for name in ("q", "kv_a", "kv_b"):
                setattr(layer, name, self.wrap("mla_projection_gemm", getattr(layer, name), synchronize=True))
            layer.output = self.scope("o", self.wrap("o_total", layer.output))
        decoder = model.decoder
        decoder._moe = self.scope("moe", self.wrap("moe_total", decoder._moe))
        decoder.router = self.wrap("moe_router_projection", decoder.router, synchronize=True)
        project = decoder._project

        def measured_project(x, name, *args, **kwargs):
            category = ("routed_expert_gemm" if ".experts." in name else
                        "shared_expert_gemm" if ".shared_experts." in name else
                        "logits_gemm" if name == "lm_head.weight" else "dense_gemm")
            return self.wrap(category, project, synchronize=True)(x, name, *args, **kwargs)

        decoder._project = measured_project
        run = decoder._run

        def measured_run(name, *args):
            if name in ("phase11_route", "phase11_expert_combine"):
                return self.wrap(name, run)(name, *args)
            return run(name, *args)

        decoder._run = measured_run
        decoder.softmax = self.wrap("moe_router_softmax", decoder.softmax, synchronize=True)

    @contextmanager
    def collect(self):
        dist.broadcast = self.wrap("cncl_o_broadcast", self.original_broadcast,
                                   synchronize=True, collective=True)
        PackedProjection.__init__ = self.wrap("weight_pack", self.original_pack)
        self.active = True
        try:
            yield
        finally:
            self.active = False
            dist.broadcast = self.original_broadcast
            PackedProjection.__init__ = self.original_pack

    def emit(self):
        model = self.model
        row = dict(rank=model.rank, step=model.forward_sequence - 1, milliseconds=dict(self.elapsed),
                   calls=dict(self.calls), payload_bytes=dict(self.payload))
        path = Path(f"/results/profile_rank{model.rank}.jsonl")
        with path.open("a") as file:
            file.write(json.dumps(row) + "\n")
        self.elapsed.clear()
        self.calls.clear()
        self.payload.clear()


class StageC2Model(NeurXDeepSeekV2ForCausalLMTP8):
    def load_weights(self, weights):
        variant = os.environ.get("NEURX_C2_VARIANT", "candidate")
        if variant == "candidate":
            from .output_projection import install, output_stripes

            stripes = {}
            loaded = super().load_weights(output_stripes(weights, self, stripes))
            try:
                install(self, stripes)
            except Exception:
                self.close()
                raise
        else:
            loaded = super().load_weights(weights)
        self.profile = None
        if variant == "profile" or os.environ.get("NEURX_C2_PROFILE") == "1":
            self.profile = WorkerProfile(self)
            self.profile.attach()
        self.capture_logits = os.environ.get("NEURX_C2_CAPTURE_LOGITS") == "1"
        return loaded

    def forward(self, *args, **kwargs):
        forward = super().forward
        if self.profile is None:
            return forward(*args, **kwargs)
        with self.profile.collect():
            return self.profile.wrap("worker_forward_total", forward)(*args, **kwargs)

    def compute_logits(self, hidden_states):
        pending = self.pending_logits
        compute = super().compute_logits
        if self.profile is None:
            logits = compute(hidden_states)
        else:
            with self.profile.collect():
                logits = self.profile.wrap("worker_logits_total", compute)(hidden_states)
            if pending:
                self.profile.emit()
            else:
                self.profile.elapsed.clear()
                self.profile.calls.clear()
                self.profile.payload.clear()
        if pending and self.capture_logits:
            path = Path(f"/results/logits_rank{self.rank}.bin")
            with path.open("ab") as file:
                file.write(logits.detach().cpu().contiguous().numpy().tobytes())
        return logits
