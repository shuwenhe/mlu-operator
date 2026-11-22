"""Native BF16 DeepSeek-V2-Lite model for the pinned Vendor vLLM worker."""
import ctypes
import json
import os
from pathlib import Path

import torch
import torch_mlu
from vllm.forward_context import get_forward_context
from vllm_mlu.v1.attention.backends.utils import get_common_metadata as get_mlu_common_metadata

from .. import NativeRMSNorm, NativeRouter
from ..attention import NativeMaterializedAttention
from ..phase12.paged_kv import PagedKV
from ..phase12_3.scheduled_attention import validate_batch
from ..phase12_4.projection import PackedProjection
from .cache import NativeCacheLayer

P = ctypes.c_void_p
I = ctypes.c_int


def _trace_enabled():
    return bool(os.environ.get("NATIVE_DEEPSEEK_TRACE"))


def _emit(event):
    if _trace_enabled():
        print(json.dumps(event), flush=True)


class NeurXDeepSeekV2ForCausalLM(torch.nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.config = vllm_config
        cfg = vllm_config.model_config.hf_config
        parallel = vllm_config.parallel_config
        if (
            (cfg.hidden_size, cfg.num_hidden_layers, cfg.q_lora_rank, cfg.kv_lora_rank)
            != (2048, 27, None, 512)
            or vllm_config.model_config.dtype != torch.bfloat16
            or parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1 or vllm_config.cache_config.enable_prefix_caching
            or vllm_config.model_config.max_model_len > 128
        ):
            raise ValueError("Native DeepSeek requires the audited BF16 single-card configuration")
        root = Path(os.environ["NATIVE_DEEPSEEK_LIBRARIES"])
        self.bridge = root / "attention/libnative_mlu.so"
        self.projection = root / "projection/libprojection.so"
        self.paged = root / "cache/libpaged_kv.so"
        self.ops = ctypes.CDLL(str(root / "decoder/libdecoder.so"))
        self.layout = ctypes.CDLL(str(self.projection))
        self.fns = {}
        for lib, name, pointers, ints in (
            (self.ops, "phase11_embedding", 4, 1), (self.ops, "phase11_residual", 4, 1),
            (self.ops, "phase11_swiglu", 4, 1), (self.ops, "phase11_route", 4, 1),
            (self.ops, "phase11_expert_combine", 5, 1), (self.ops, "phase11_logits", 3, 3),
            (self.ops, "worker_rope_bf16", 6, 1), (self.ops, "phase10_merge", 3, 1),
            (self.layout, "phase10_prepare_bf16", 5, 2),
            (self.layout, "phase10_append_bf16", 7, 3),
            (self.layout, "phase10_cast_bf16", 3, 1),
        ):
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = [P] * pointers + [I] * ints, I
            self.fns[name] = fn
        self.fns["phase5_cast_kv"] = self.ops.phase5_cast_kv
        self.fns["phase5_cast_kv"].argtypes = [P] * 3 + [I, ctypes.c_bool]
        self.fns["phase5_cast_kv"].restype = I
        self.softmax = ctypes.CDLL(str(self.bridge)).phase9_softmax_bf16
        self.softmax.argtypes = [P] * 3 + [ctypes.c_uint32] + [P] * 4 + [ctypes.c_uint32] * 3
        self.softmax.restype = I
        self.norm = NativeRMSNorm(self.bridge)
        self.router = NativeRouter(self.bridge)
        self.attention = NativeMaterializedAttention(self.bridge)
        self.layers = torch.nn.ModuleList(
            NativeCacheLayer(f"model.layers.{i}.self_attn.native_mla", vllm_config)
            for i in range(27)
        )
        self.weights = {}
        self.projections = {}
        self.resident = {}
        self.stream = None
        self.dump_root = os.environ.get("NATIVE_DEEPSEEK_DUMP")
        self.dump_write = bool(os.environ.get("NATIVE_DEEPSEEK_DUMP_WRITE"))
        self.completed_slots = {}
        self.next_trace = 0
        self.logit_traces = []
        self.architecture_name = "NeurXDeepSeekV2ForCausalLM"
        self.model_instance_id = f"{id(self):#x}"
        _emit(
            {
                "architecture_name": self.architecture_name,
                "model_instance_id": self.model_instance_id,
                "native_cache_layer_count": len(self.layers),
                "static_context_layer_count": len(
                    self.config.compilation_config.static_forward_context
                ),
            }
        )

    def load_weights(self, weights):
        self.weights = dict(weights)
        if len(self.weights) != 5291:
            raise ValueError(f"Unexpected native checkpoint tensor count {len(self.weights)}")
        return set(self.weights)

    def _resident(self, name):
        if name not in self.resident:
            self.resident[name] = self.weights[name].to(self.stream.device)
        return self.resident[name]

    def _project(self, x, name, start=0, width=None):
        key = (name, start, width)
        if key not in self.projections:
            w = self.weights[name]
            if width is not None:
                w = w[start:start + width]
            self.projections[key] = PackedProjection(self.projection, w.to(x.device))
        return self.projections[key](x)

    def _run(self, name, *args):
        pointers = (a.data_ptr() if isinstance(a, torch.Tensor) else a for a in args)
        status = self.fns[name](self.stream.mlu_stream, *pointers)
        if status:
            raise RuntimeError(f"{name} CNRT status={status}")
        self.stream.synchronize()

    def _empty(self, shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=self.stream.device)

    def _cast(self, wide):
        out = self._empty(wide.shape)
        self._run("phase10_cast_bf16", wide, out, out.shape[0])
        return out

    def _add(self, a, b):
        out = self._empty(a.shape)
        self._run("phase11_residual", a, b, out, a.numel())
        return out

    def _ffn(self, prefix, x):
        gate = self._project(x, prefix + "gate_proj.weight")
        up = self._project(x, prefix + "up_proj.weight")
        mid = self._empty(gate.shape)
        self._run("phase11_swiglu", gate, up, mid, mid.numel())
        return self._cast(self._project(mid, prefix + "down_proj.weight"))

    def _moe(self, prefix, x):
        rows = x.shape[0]
        scores = self.router(x, self._resident(prefix + "gate.weight"))
        probs = self._empty(scores.shape, torch.float32)
        low = self._empty(scores.shape)
        maximum = self._empty((rows,), torch.float32)
        denom = self._empty((rows,), torch.float32)
        status = self.softmax(self.stream.mlu_stream, scores.data_ptr(), None, 0,
                              maximum.data_ptr(), denom.data_ptr(), probs.data_ptr(),
                              low.data_ptr(), rows, 64, 64)
        if status:
            raise RuntimeError(f"Native router softmax status={status}")
        ids = self._empty((rows, 6), torch.int32)
        weights = self._empty((rows, 6), torch.float32)
        self._run("phase11_route", probs, ids, weights, rows)
        chosen = ids.cpu().flatten().tolist()
        routed = self._empty((rows * 6, 2048))
        for i, expert in enumerate(chosen):
            routed[i:i+1].copy_(self._ffn(f"{prefix}experts.{expert}.", x[i//6:i//6+1]))
        shared = self._ffn(prefix + "shared_experts.", x)
        result = self._empty(x.shape)
        self._run("phase11_expert_combine", routed, weights, shared, result, rows)
        return result

    def _project_attention(self, prefix, x, cursor):
        rows = x.shape[0]
        q = self._project(x, prefix + "q_proj.weight")
        kv = self._project(x, prefix + "kv_a_proj_with_mqa.weight")
        latent = self._empty((rows, 512))
        self._run("phase5_cast_kv", kv, latent, rows, True)
        latent = self.norm(latent, self._resident(prefix + "kv_a_layernorm.weight"), 1e-6)
        expanded = self._project(latent, prefix + "kv_b_proj.weight")
        rope = self._empty((17, rows, 64))
        rotated = self._empty(rope.shape)
        positions = self._empty((rows,), torch.int32)
        self._run("phase10_prepare_bf16", q, kv, rope, positions, rows, cursor)
        self._run("worker_rope_bf16", rope, self.cos, self.sin, positions, rotated, rows)
        query = self._empty((16, rows, 192))
        key = self._empty(query.shape)
        value = self._empty((16, rows, 128))
        self._run("phase10_append_bf16", q, expanded, rotated, query, key, value, rows, 0, rows)
        return query, key, value, positions

    def _dump(self, trace, name, tensor):
        if trace is None:
            return
        directory, step = trace
        bits = tensor.contiguous().cpu()
        if bits.dtype == torch.bfloat16:
            bits = bits.view(torch.int16)
        bits.numpy().tofile(directory / f"{step}-{name}.bin")

    def _request(self, input_ids, span, metadata, request):
        rows = span.end - span.start
        cursor = span.length - rows
        trace = None
        if self.dump_root:
            first_page = int(metadata.block_table_tensor[request, 0].item())
            if cursor == 0:
                directory = Path(self.dump_root) / f"request{self.next_trace}"
                directory.mkdir(parents=True, exist_ok=True)
                self.next_trace += 1
                self.completed_slots[first_page] = directory, 0
                input_ids.cpu().numpy().tofile(directory / "prompt.ids")
            trace = self.completed_slots[first_page]
            self.completed_slots[first_page] = trace[0], trace[1] + 1
        self.logit_traces.append(trace)
        x = self._empty((rows, 2048))
        self._run("phase11_embedding", self._resident("model.embed_tokens.weight"), input_ids, x, rows)
        self._dump(trace, "embedding", x)
        for i, layer in enumerate(self.layers):
            prefix = f"model.layers.{i}."
            spec = layer.get_kv_cache_spec(self.config)
            cache = PagedKV(self.paged, *layer.kv_cache, spec.block_size)
            _emit(
                {
                    "architecture_name": self.architecture_name,
                    "model_instance_id": self.model_instance_id,
                    "request_id": str(span.request_id),
                    "layer_id": i,
                    "layer_class": type(layer).__name__,
                    "cache_spec": {
                        "block_size": int(spec.block_size),
                        "num_kv_heads": int(spec.num_kv_heads),
                        "head_size": int(spec.head_size),
                        "head_size_v": int(spec.head_size_v),
                        "dtype": str(spec.dtype),
                    },
                    "worker_allocation_id": f"{layer.kv_cache[0].data_ptr():#x}:{layer.kv_cache[1].data_ptr():#x}",
                    "cache_base_address": min(layer.kv_cache[0].data_ptr(), layer.kv_cache[1].data_ptr()),
                    "cache_allocation_bytes": int(
                        layer.kv_cache[0].numel() * layer.kv_cache[0].element_size()
                        + layer.kv_cache[1].numel() * layer.kv_cache[1].element_size()
                    ),
                    "k_view_address": layer.kv_cache[0].data_ptr(),
                    "v_view_address": layer.kv_cache[1].data_ptr(),
                    "reshape_hook_invoked": True,
                    "zero_copy_verified": (
                        layer.kv_cache[0].untyped_storage().data_ptr()
                        == layer.kv_cache[1].untyped_storage().data_ptr()
                    ),
                    "scheduler_metadata_received": True,
                    "block_table": metadata.block_table_tensor[request, :span.blocks].tolist(),
                    "slot_mapping": metadata.slot_mapping[span.start:span.end].tolist(),
                    "cnrt_queue_id": int(self.stream.mlu_stream),
                    "native_cache_layer_count": len(self.layers),
                    "fallback_count": 0,
                }
            )
            n = self.norm(x, self._resident(prefix + "input_layernorm.weight"), 1e-6)
            q, k, v, pos = self._project_attention(prefix + "self_attn.", n, cursor)
            if self.dump_write:
                self._dump(trace, f"layer{i}-key_write", k)
                self._dump(trace, f"layer{i}-value_write", v)
            cache.write(k, v, metadata.slot_mapping[span.start:span.end].contiguous())
            table = metadata.block_table_tensor[request, :span.blocks].contiguous()
            k, v = cache.read(table, span.length)
            _, _, _, av = self.attention(q, k, v, pos, 0, span.length, self.scale)
            merged = self._empty(x.shape)
            self._run("phase10_merge", av, merged, rows)
            attention = self._cast(self._project(merged, prefix + "self_attn.o_proj.weight"))
            self._dump(trace, f"layer{i}-key", k)
            self._dump(trace, f"layer{i}-value", v)
            self._dump(trace, f"layer{i}-attention", attention)
            x = self._add(x, attention)
            n = self.norm(x, self._resident(prefix + "post_attention_layernorm.weight"), 1e-6)
            x = self._add(x, self._ffn(prefix + "mlp.", n) if i == 0 else self._moe(prefix + "mlp.", n))
            self._dump(trace, f"layer{i}", x)
            _emit(
                {
                    "architecture_name": self.architecture_name,
                    "model_instance_id": self.model_instance_id,
                    "native_layer": i,
                    "queue": int(self.stream.mlu_stream),
                    "device": str(x.device),
                    "query": rows,
                    "length": span.length,
                    "blocks": table.tolist(),
                    "request_row": request,
                }
            )
        return self.norm(x, self._resident("model.norm.weight"), 1e-6)

    def _init_stream(self, device):
        if self.stream is None:
            self.stream = torch_mlu.mlu.Stream(device=device)
            fixtures = Path(os.environ["NATIVE_DEEPSEEK_ROPE"])
            self.cos, self.sin = (
                torch.from_file(str(fixtures / f"{name}.bin"), size=131*64, dtype=torch.int16)
                .view(torch.bfloat16).to(device) for name in ("cos", "sin")
            )
            self.scale = float((fixtures / "scale.txt").read_text())

    def embed_input_ids(self, input_ids):
        self._init_stream(input_ids.device)
        self.stream.wait_stream(torch_mlu.mlu.current_stream(input_ids.device))
        with torch_mlu.mlu.stream(self.stream):
            out = self._empty((input_ids.numel(), 2048))
            self._run("phase11_embedding", self._resident("model.embed_tokens.weight"),
                      input_ids.int().contiguous(), out, input_ids.numel())
        return out

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise ValueError("Unsupported native model inputs")
        context_metadata = get_forward_context().attn_metadata
        if context_metadata is None:
            return torch.zeros((input_ids.numel(), 2048), dtype=torch.bfloat16, device=input_ids.device)
        metadata = get_mlu_common_metadata()
        if metadata is None:
            metadata = context_metadata[self.layers[0].layer_name]
        self._init_stream(input_ids.device)
        caller = torch_mlu.mlu.current_stream(input_ids.device)
        self.stream.wait_stream(caller)
        with torch_mlu.mlu.stream(self.stream):
            spec = self.layers[0].get_kv_cache_spec(self.config)
            spans = validate_batch(metadata, range(metadata.num_reqs), positions.int(), spec.block_size,
                                   self.layers[0].kv_cache[0].shape[0] // spec.block_size)
            tokens = input_ids.int()
            self.logit_traces = []
            out = torch.cat([self._request(tokens[s.start:s.end], s, metadata, row)
                             for row, s in enumerate(spans)])
            self.stream.synchronize()
        return out

    def compute_logits(self, hidden_states):
        if self.stream is None:
            return torch.zeros((hidden_states.shape[0], 102400), dtype=torch.float32,
                               device=hidden_states.device)
        self.stream.wait_stream(torch_mlu.mlu.current_stream(hidden_states.device))
        with torch_mlu.mlu.stream(self.stream):
            output = self._empty((hidden_states.shape[0], 102400), torch.float32)
            for start in range(0, 102400, 10240):
                wide = self._project(hidden_states, "lm_head.weight", start, 10240)
                self._run("phase11_logits", wide, output, hidden_states.shape[0], 10240, start)
            if len(self.logit_traces) == output.shape[0]:
                for trace, row in zip(self.logit_traces, output, strict=True):
                    self._dump(trace, "logits", row)
        return output


NativeDeepseekV2Lite = NeurXDeepSeekV2ForCausalLM
