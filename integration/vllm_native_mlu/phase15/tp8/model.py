"""Native DeepSeek-V2-Lite worker for eight MLU ranks."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_tp_group

from ...phase13_2.linear import validate_tp
from ...phase13_3_2.mapping import RULES
from ...phase13_3_2.probe import compare
from ...phase13_4.full_model import ReplicatedDecoder
from ...phase13_4.mla import NativeCalls
from ...phase14_1.model import EngineMLA, NeurXDeepSeekV2ForCausalLM
from .cache import TP8CacheLayer
from .contracts import BLOCK_SIZE, HEADS, TP, expert_owner, spans


class ExpertParallelDecoder(ReplicatedDecoder):
    def __init__(self, ops):
        super().__init__(ops)
        self.ops = ops

    def _moe(self, prefix, x):
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
        chosen = ids.flatten().tolist()
        group = self.ops.group
        self.ops.agree((prefix, chosen))
        routed = x.new_zeros((rows * 6, 2048))
        for slot, expert in enumerate(chosen):
            if expert_owner(expert) == group.rank_in_group:
                token = slot // 6
                routed[slot:slot + 1].copy_(self._ffn(f"{prefix}experts.{expert}.", x[token:token + 1]))
        routed = group.all_reduce(routed)
        shared = self._ffn(prefix + "shared_experts.", x)
        result = self._empty(x.shape)
        self._run("phase11_expert_combine", routed, weights, shared, result, rows)
        return result


class NeurXDeepSeekV2ForCausalLMTP8(NeurXDeepSeekV2ForCausalLM):
    """Use the vendor scheduler bridge and the ordered native MLA operators."""

    def __init__(self, *, vllm_config, prefix=""):
        torch.nn.Module.__init__(self)
        cfg = vllm_config.model_config
        hf = cfg.hf_config
        expected = dict(hidden_size=2048, num_hidden_layers=27, num_attention_heads=16,
                        num_key_value_heads=16, kv_lora_rank=512, q_lora_rank=None,
                        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
                        n_routed_experts=64, n_shared_experts=2, num_experts_per_tok=6,
                        intermediate_size=10944, moe_intermediate_size=1408,
                        first_k_dense_replace=1, moe_layer_freq=1, vocab_size=102400,
                        topk_method="greedy", scoring_func="softmax", norm_topk_prob=False,
                        rms_norm_eps=1e-6, routed_scaling_factor=1.0)
        parallel = vllm_config.parallel_config
        cache = vllm_config.cache_config
        if (
            any(getattr(hf, key, None) != value for key, value in expected.items())
            or cfg.dtype != torch.bfloat16 or not 0 < cfg.max_model_len <= 161
            or (parallel.tensor_parallel_size, parallel.pipeline_parallel_size,
                parallel.data_parallel_size) != (TP, 1, 1)
            or cache.enable_prefix_caching or cache.block_size != BLOCK_SIZE
        ):
            raise ValueError("Unsupported native DeepSeek-V2-Lite TP8 configuration")
        self.max_model_len = cfg.max_model_len
        self.group = get_tp_group()
        self.rank = self.group.rank_in_group
        self.device = torch.device("mlu", torch.mlu.current_device())
        validate_tp(self.group, self.device)
        if self.group.world_size != TP or self.rank != self.device.index or self.rank != dist.get_rank():
            raise ValueError("Expected eight local CNCL workers with matching rank and device")
        self.layers = torch.nn.ModuleList(
            TP8CacheLayer(f"model.layers.{i}.self_attn.native_mla", vllm_config) for i in range(27)
        )
        self.libraries = Path(os.environ["NATIVE_DEEPSEEK_LIBRARIES"])
        self.ordered = Path(os.environ["NEURX_ORDERED_O"])
        self.paged = Path(os.environ["NEURX_PAGED_TWO"])
        self.mla = []
        self.decoder = None
        self.trace = self.benchmark = self.boundary_capture = False
        self.dump = self.acceptance_fault = None
        self.sequence = self.forward_sequence = 0
        self.fixture = os.environ.get("NEURX_TP8_LOGIT_FIXTURE")
        self.pending_logits = self.cache_reported = self.capture_projection = False
        print(json.dumps(dict(stage_c1="constructed", rank=self.rank, device=str(self.device),
                              heads=list(range(self.rank * HEADS, (self.rank + 1) * HEADS)))),
              flush=True)

    def load_weights(self, weights):
        local = [dict() for _ in self.layers]
        remaining = {}
        loaded = set()
        stream = torch.mlu.Stream(device=self.device)
        with torch.mlu.stream(stream):
            for name, tensor in weights:
                if name in loaded:
                    raise ValueError(f"Duplicate checkpoint parameter {name}")
                loaded.add(name)
                parts = name.split(".")
                if parts[:2] == ["model", "layers"] and parts[3] == "self_attn":
                    suffix = ".".join(parts[4:])
                    shape, _ = RULES[suffix]
                    if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
                        raise ValueError(f"Unexpected MLA weight {name}")
                    axis, start, width = spans(suffix, self.rank)
                    own = tensor if axis is None else tensor.narrow(axis, start, width)
                    local[int(parts[2])][suffix] = own.contiguous().to(self.device)
                elif parts[:2] == ["model", "layers"] and parts[3] == "input_layernorm":
                    if tensor.shape != (2048,) or tensor.dtype != torch.bfloat16:
                        raise ValueError(f"Unexpected input normalization weight {name}")
                    local[int(parts[2])]["input_layernorm.weight"] = tensor.to(self.device)
                elif parts[3:5] != ["mlp", "experts"] or expert_owner(int(parts[5])) == self.rank:
                    remaining[name] = tensor
            if len(loaded) != 5291:
                raise ValueError(f"Unexpected checkpoint parameter count {len(loaded)}")
            rope = Path(os.environ["NATIVE_DEEPSEEK_ROPE"])
            table_rows = max(131, self.max_model_len)
            cos, sin = (
                torch.from_file(str(rope / f"{name}.bin"), size=table_rows * 64, dtype=torch.int16)
                .view(torch.bfloat16).reshape(table_rows, 64).to(self.device)
                for name in ("cos", "sin")
            )
            scale = float((rope / "scale.txt").read_text())
            self.ops = NativeCalls(self.group, self.device, self.libraries, self.paged, cos, sin, scale)
            try:
                self.decoder = ExpertParallelDecoder(self.ops)
                self.decoder.weights = remaining
                for shard in local:
                    self.mla.append(EngineMLA(self.ops, self.libraries / "projection/libprojection.so",
                                              self.ordered, shard, self.decoder.norm))
                stream.synchronize()
            except Exception:
                self.close()
                raise
        return loaded

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        self.active_attention = None
        try:
            result = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"TP8 rank {self.rank}, forward {self.forward_sequence}, "
                f"(layer, first_position, last_position)={self.active_attention}: {exc}"
            ) from exc
        if self.pending_logits and self.rank == 0 and os.getenv("NEURX_CONTEXT_TRACE") == "1":
            layer, first, last = self.active_attention
            print(json.dumps(dict(rank=self.rank, forward=self.forward_sequence - 1,
                                  layer=layer, first_position=first, last_position=last)), flush=True)
        if self.fixture and self.pending_logits and self.rank == 0:
            print(json.dumps(dict(stage_c1="native_forward", step=self.forward_sequence - 1,
                                  rows=input_ids.numel(), first_position=positions.flatten()[0].item())),
                  flush=True)
        return result

    def compute_logits(self, hidden_states):
        logits = self.decoder.compute_logits(hidden_states)
        if logits is not None and self.pending_logits:
            bad = int(not torch.isfinite(logits).all().item())
            if self.fixture and self.sequence < 4:
                path = Path(self.fixture) / f"{self.sequence}-logits.bin"
                expected = torch.from_file(str(path), size=102400, dtype=torch.float32).view(1, -1)
                bad += compare(logits, expected, dict(rank=self.rank, step=self.sequence), "worker_logits")
            failure = torch.tensor([bad], dtype=torch.int32, device=self.device)
            if self.group.all_reduce(failure).item():
                raise ValueError("TP8 worker logits failed finiteness or frozen tolerance")
            self.sequence += 1
        self.pending_logits = False
        return logits
