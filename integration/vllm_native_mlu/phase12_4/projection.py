"""Explicit ownership for accepted BF16 packed weights on a torch MLU queue."""

import ctypes

import torch
import torch_mlu


class PackedProjection:
    def __init__(self, library, weight):
        if weight.ndim != 2 or weight.dtype != torch.bfloat16 or weight.device.type != "mlu":
            raise ValueError("Expected a BF16 MLU checkpoint weight [N, K]")
        self.n, self.k = weight.shape
        self.device = weight.device
        self.library = ctypes.CDLL(str(library))
        pointer = ctypes.c_void_p
        self.handle = pointer()
        self.library.phase124_pack.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int,
                                               ctypes.POINTER(pointer)]
        self.library.phase124_project.argtypes = [pointer] * 4 + [ctypes.c_int]
        self.library.phase124_release.argtypes = [ctypes.POINTER(pointer)]
        for function in (self.library.phase124_pack, self.library.phase124_project,
                         self.library.phase124_release):
            function.restype = ctypes.c_int
        with torch_mlu.mlu.device(self.device):
            stream = torch_mlu.mlu.current_stream(self.device)
            kn = weight.transpose(0, 1).contiguous()
            try:
                status = self.library.phase124_pack(stream.mlu_stream, kn.data_ptr(), self.k,
                                                    self.n, ctypes.byref(self.handle))
                self._check(status, "pack")
                stream.synchronize()
            except Exception:
                if self.handle:
                    self.close()
                raise

    @staticmethod
    def _check(status, operation):
        if status:
            raise RuntimeError(f"Native projection {operation} returned CNRT status {status}")

    def __call__(self, x):
        if (
            not self.handle or x.ndim != 2 or not 1 <= x.shape[0] <= 128 or x.shape[1] != self.k
            or x.device != self.device or x.dtype != torch.bfloat16 or not x.is_contiguous()
        ):
            raise ValueError("Unsupported native projection input or closed weight")
        with torch_mlu.mlu.device(self.device):
            stream = torch_mlu.mlu.current_stream(self.device)
            output = x.new_empty((x.shape[0], self.n), dtype=torch.float32)
            self._check(self.library.phase124_project(stream.mlu_stream, x.data_ptr(),
                                                       output.data_ptr(), self.handle, x.shape[0]),
                        "launch")
            stream.synchronize()
        return output

    def close(self):
        if self.handle:
            with torch_mlu.mlu.device(self.device):
                self._check(self.library.phase124_release(ctypes.byref(self.handle)), "release")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
