"""Zero-copy worker allocation for two-head native MLA."""
import torch
from vllm_mlu.v1.kv_cache_interface import MLUFullAttentionSpec

from ...phase12_5.cache import NativeBackend, NativeCacheLayer
from .contracts import HEADS, layout


class TP8Backend(NativeBackend):
    @staticmethod
    def get_name():
        return "NEURX_NATIVE_TP8_MATERIALIZED"

    @staticmethod
    def reshape_kv_cache(raw, spec, num_blocks, kernel_block_size, cache_dtype):
        slots, split, words = layout(num_blocks, spec.block_size)
        if (
            (spec.num_kv_heads,spec.head_size,spec.head_size_v)!=(HEADS,192,128)
            or spec.dtype != torch.bfloat16 or cache_dtype not in ("auto","bfloat16")
            or kernel_block_size != spec.block_size or spec.page_size_bytes*num_blocks != words*2
            or raw.dtype != torch.int8 or raw.device.type != "mlu" or raw.ndim != 1
            or not raw.is_contiguous() or raw.numel() != words*2 or raw.data_ptr()%2
        ):
            raise ValueError("Incompatible TP=8 worker cache allocation")
        keys, values = raw.view(torch.bfloat16).split((split,words-split))
        return keys.view(slots,HEADS,192), values.view(slots,HEADS,128)


class TP8CacheLayer(NativeCacheLayer):
    def get_attn_backend(self):
        return TP8Backend

    def get_kv_cache_spec(self, vllm_config):
        block = vllm_config.cache_config.block_size
        layout(1,block)
        return MLUFullAttentionSpec(block_size=block,num_kv_heads=HEADS,head_size=192,
                                    head_size_v=128,dtype=torch.bfloat16)
