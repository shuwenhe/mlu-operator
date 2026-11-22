"""Device-only data movement for scheduler-owned materialized KV pages."""

import ctypes

import torch
import torch_mlu


class PagedKV:
    def __init__(self, library, keys, values, block_size):
        if (
            keys.ndim != 3 or keys.shape[1:] != (16, 192)
            or values.shape != (keys.shape[0], 16, 128)
            or keys.device.type != "mlu" or keys.device != values.device
            or keys.dtype not in (torch.float16, torch.bfloat16) or keys.dtype != values.dtype
            or not keys.is_contiguous() or not values.is_contiguous()
            or not 0 < block_size <= keys.shape[0] < 2**31 or keys.shape[0] % block_size
        ):
            raise ValueError("Unsupported paged KV allocation")
        self.keys, self.values, self.block_size = keys, values, block_size
        self.library = ctypes.CDLL(str(library))
        self._launch = self.library.phase12_paged_kv
        self._launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4
        self._launch.restype = ctypes.c_int

    def write(self, keys, values, slots):
        """Scatter rows to distinct scheduler-owned int32 slots. Padding is skipped."""
        if keys.ndim != 3:
            raise ValueError("Expected head-major keys")
        self._transfer(keys, values, slots, keys.shape[1], True)

    def read(self, block_table, length):
        """Gather the committed logical prefix into the native attention layout."""
        if not 1 <= length <= 131:
            raise ValueError("Unsupported logical cache length")
        keys = self.keys.new_empty((16, length, 192))
        values = self.values.new_empty((16, length, 128))
        self._transfer(keys, values, block_table, length, False)
        return keys, values

    def _transfer(self, keys, values, mapping, tokens, write):
        count = tokens if write else (tokens + self.block_size - 1) // self.block_size
        if (
            not 1 <= tokens <= 131 or keys.shape != (16, tokens, 192)
            or values.shape != (16, tokens, 128) or mapping.shape != (count,)
            or mapping.dtype != torch.int32 or mapping.device != self.keys.device
            or not mapping.is_contiguous()
            or any(t.device != self.keys.device or t.dtype != self.keys.dtype or not t.is_contiguous()
                   for t in (keys, values))
        ):
            raise ValueError("Unsupported paged KV transfer arguments")
        with torch_mlu.mlu.device(self.keys.device):
            stream = torch_mlu.mlu.current_stream(self.keys.device)
            tensors = self.keys, self.values, keys, values, mapping
            code = self._launch(stream.mlu_stream, *(t.data_ptr() for t in tensors),
                                tokens, self.block_size, self.keys.shape[0], int(write))
            if code:
                raise RuntimeError(f"Native paged KV CNRT launch failed with status {code}")
            stream.synchronize()
