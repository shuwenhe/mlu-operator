"""Materialized attention CustomOp for the accepted native QK, Softmax and AV chain."""

import ctypes
import math

import torch
import torch_mlu
from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("checkpoint_native_mlu_materialized_attention")
class NativeMaterializedAttention(CustomOp):
    def __init__(self, library):
        super().__init__(enforce_enable=True)
        self.library = ctypes.CDLL(str(library))
        self.launchers = {}
        pointer, size = ctypes.c_void_p, ctypes.c_uint32
        for dtype, tag in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
            qk = getattr(self.library, f"phase9_launch_qk_{tag}")
            softmax = getattr(self.library, f"phase9_softmax_{tag}")
            av = getattr(self.library, f"phase9_av_{tag}")
            qk.argtypes = [pointer] * 4 + [size, ctypes.c_float] + [pointer] * 3 + [size] * 4
            softmax.argtypes = [pointer] * 3 + [size] + [pointer] * 4 + [size] * 3
            av.argtypes = [pointer] * 5 + [size] * 4
            for launch in (qk, softmax, av):
                launch.restype = ctypes.c_int
            self.launchers[dtype] = qk, softmax, av

    def forward_native(self, q, k, v, positions, cache_base, length, scale):
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError("Expected materialized [heads, tokens, width] tensors")
        heads, queries, width = q.shape
        capacity = k.shape[1]
        if (
            heads != 16 or width != 192 or not 1 <= queries <= 128
            or not 1 <= length <= capacity <= 131 or not 0 <= cache_base < length
            or k.shape != (heads, capacity, 192) or v.shape != (heads, capacity, 128)
            or positions.shape != (queries,) or positions.dtype != torch.int32
            or q.device.type != "mlu" or q.dtype not in self.launchers
            or k.dtype != q.dtype or v.dtype != q.dtype
            or not math.isfinite(ctypes.c_float(scale).value) or scale <= 0
            or any(t.device != q.device or not t.is_contiguous() for t in (q, k, v, positions))
        ):
            raise ValueError("Unsupported materialized MLU attention arguments")

        qk, softmax, av = self.launchers[q.dtype]
        with torch_mlu.mlu.device(q.device):
            stream = torch_mlu.mlu.current_stream(q.device)
            queue = stream.mlu_stream
            score_shape = (heads, queries, length)
            out_shape = (heads, queries, 128)

            def alloc(shape, dtype=torch.float32):
                return torch.empty(shape, dtype=dtype, device=q.device)

            raw, scaled, masked = alloc(score_shape), alloc(score_shape, q.dtype), alloc(score_shape)
            maximum, denom = alloc((heads, queries)), alloc((heads, queries))
            probs, low_probs = alloc(score_shape), alloc(score_shape, q.dtype)
            raw_av, result = alloc(out_shape), alloc(out_shape, q.dtype)
            try:
                status = qk(queue, q.data_ptr(), k.data_ptr(), positions.data_ptr(), cache_base,
                            scale, raw.data_ptr(), scaled.data_ptr(), masked.data_ptr(),
                            heads, queries, length, capacity)
                if status:
                    raise RuntimeError(f"QK CNRT launch failed with status {status}")
                status = softmax(queue, masked.data_ptr(), None, 0, maximum.data_ptr(),
                                 denom.data_ptr(), probs.data_ptr(), low_probs.data_ptr(),
                                 heads * queries, length, length)
                if status:
                    raise RuntimeError(f"Softmax CNRT launch failed with status {status}")
                status = av(queue, low_probs.data_ptr(), v.data_ptr(), raw_av.data_ptr(),
                            result.data_ptr(), heads, queries, length, capacity)
                if status:
                    raise RuntimeError(f"AV CNRT launch failed with status {status}")
            finally:
                # Retain scratch storage through the last queued consumer.
                stream.synchronize()
        return masked, low_probs, raw_av, result

    forward_oot = forward_native
