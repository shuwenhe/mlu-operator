"""Explicit vLLM CustomOp for the accepted BF16 native RMSNorm kernel."""

import ctypes
import math

import torch
import torch_mlu  # Register the MLU dispatch key before loading vLLM.
from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("checkpoint_native_mlu_rmsnorm")
class NativeRMSNorm(CustomOp):
    def __init__(self, library):
        super().__init__(enforce_enable=True)
        self.library = ctypes.CDLL(str(library))
        self.launch = self.library.phase82_native_bf16
        pointer = ctypes.c_void_p
        self.launch.argtypes = [pointer] * 6 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.launch.restype = ctypes.c_int

    def forward_native(self, x, weight, eps):
        if x.ndim < 1 or x.shape[-1] not in (512, 2048):
            raise ValueError("Native RMSNorm requires width 512 or 2048")
        width = x.shape[-1]
        rows = x.numel() // width
        if (
            x.device.type != "mlu"
            or weight.device != x.device
            or x.dtype != torch.bfloat16
            or weight.dtype != x.dtype
            or weight.shape != (width,)
            or not x.is_contiguous()
            or not weight.is_contiguous()
            or not 0 < rows <= 2**31 - 1
            or not math.isfinite(eps)
            or not 0 < ctypes.c_float(eps).value <= torch.finfo(torch.float32).max
        ):
            raise ValueError("Unsupported native MLU RMSNorm arguments")
        with torch_mlu.mlu.device(x.device):
            stream = torch_mlu.mlu.current_stream(x.device)
            output = torch.empty_like(x)
            status = self.launch(
                stream.mlu_stream, x.data_ptr(), weight.data_ptr(), output.data_ptr(),
                None, None, rows, width, eps,
            )
            if status:
                raise RuntimeError(f"Native MLU RMSNorm CNRT launch failed with status {status}")
            # Keep caller-owned input storage alive until this queue finishes reading it.
            stream.synchronize()
        return output

    forward_oot = forward_native


@CustomOp.register("checkpoint_native_mlu_router")
class NativeRouter(CustomOp):
    """DeepSeek-V2-Lite router logits using the pinned checkpoint FP32 order."""

    def __init__(self, library):
        super().__init__(enforce_enable=True)
        self.library = ctypes.CDLL(str(library))
        self.launch = self.library.phase115_router_bf16
        self.launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]
        self.launch.restype = ctypes.c_int

    def forward_native(self, x, weight):
        if (
            x.ndim != 2
            or not 0 < x.shape[0] <= 128
            or x.shape[1] != 2048
            or x.device.type != "mlu"
            or weight.device != x.device
            or x.dtype != torch.bfloat16
            or weight.dtype != x.dtype
            or weight.shape != (64, 2048)
            or not x.is_contiguous()
            or not weight.is_contiguous()
        ):
            raise ValueError("Unsupported native MLU router arguments")
        with torch_mlu.mlu.device(x.device):
            stream = torch_mlu.mlu.current_stream(x.device)
            output = torch.empty((x.shape[0], 64), dtype=torch.float32, device=x.device)
            status = self.launch(
                stream.mlu_stream, x.data_ptr(), weight.data_ptr(), output.data_ptr(), x.shape[0]
            )
            if status:
                raise RuntimeError(f"Native MLU router CNRT launch failed with status {status}")
            stream.synchronize()
        return output

    forward_oot = forward_native
