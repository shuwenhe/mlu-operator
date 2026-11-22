"""Vendor worker adapter for the materialized native layout."""
import json
import os

import torch
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm_mlu.v1.kv_cache_interface import MLUFullAttentionSpec
from vllm_mlu.v1.attention.backends.mla.tmo_flashmla import TMOFlashMLABackend
from vllm_mlu.v1.attention.backends.mla.tmo_flashmla import TMOFlashMLAMetadataBuilder

from ..phase12_4.cache_spec import reshape_kv_cache


def _trace_enabled():
    return bool(os.environ.get("NATIVE_DEEPSEEK_TRACE"))


def _emit(event):
    if _trace_enabled():
        print(json.dumps(event), flush=True)


class NativeMetadataBuilder(TMOFlashMLAMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if common_prefix_len:
            raise ValueError("Native MLA currently requires exclusive pages")
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build=fast_build)
        _emit(
            {
                "scheduler_metadata_received": True,
                "common_prefix_len": int(common_prefix_len),
                "fast_build": bool(fast_build),
                "num_reqs": int(common_attn_metadata.num_reqs),
                "num_actual_tokens": int(common_attn_metadata.num_actual_tokens),
                "max_query_len": int(common_attn_metadata.max_query_len),
            }
        )
        return metadata


class NativeBackend(TMOFlashMLABackend):
    @staticmethod
    def get_name():
        return "NATIVE_MATERIALIZED_MLA"

    @staticmethod
    def get_builder_cls():
        return NativeMetadataBuilder

    @classmethod
    def indexes_kv_by_block_stride(cls):
        return False

    @staticmethod
    def reshape_kv_cache(raw, spec, num_blocks, kernel_block_size, cache_dtype):
        contract = FullAttentionSpec(
            block_size=spec.block_size, num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size, head_size_v=spec.head_size_v, dtype=spec.dtype,
        )
        if spec.page_size_bytes != contract.page_size_bytes:
            raise ValueError("Unexpected vendor cache padding")
        if raw.data_ptr() % 2:
            raise ValueError("Native cache base pointer is not BF16-aligned")
        keys, values = reshape_kv_cache(raw, contract, num_blocks, kernel_block_size, cache_dtype)
        key_storage = keys.untyped_storage().data_ptr()
        value_storage = values.untyped_storage().data_ptr()
        raw_storage = raw.untyped_storage().data_ptr()
        if key_storage != value_storage or key_storage != raw_storage:
            raise ValueError("Native cache reshape produced duplicate storage")
        expected_k_words = num_blocks * contract.block_size * contract.num_kv_heads * contract.head_size
        if keys.storage_offset() != 0 or values.storage_offset() != expected_k_words:
            raise ValueError("Native cache K/V offsets do not match the accepted layout")
        _emit(
            {
                "reshape_hook_invoked": True,
                "cache_allocation_bytes": int(raw.numel()),
                "cache_base_address": raw.data_ptr(),
                "k_view_address": keys.data_ptr(),
                "v_view_address": values.data_ptr(),
                "zero_copy_verified": True,
                "cache_spec": {
                    "block_size": int(contract.block_size),
                    "num_kv_heads": int(contract.num_kv_heads),
                    "head_size": int(contract.head_size),
                    "head_size_v": int(contract.head_size_v),
                    "dtype": str(contract.dtype),
                },
            }
        )
        return keys, values



class NativeCacheLayer(torch.nn.Module, AttentionLayerBase):
    def __init__(self, name, config):
        super().__init__()
        self.layer_name = name
        self.kv_cache = ()
        self.prefill_backend = None
        context = config.compilation_config.static_forward_context
        if name in context:
            raise ValueError(f"Duplicate native cache layer {name}")
        context[name] = self
        _emit(
            {
                "layer_id": name,
                "layer_class": type(self).__name__,
                "static_context_layer_count": len(context),
            }
        )

    def get_attn_backend(self):
        return NativeBackend

    def get_kv_cache_spec(self, vllm_config):
        return MLUFullAttentionSpec(block_size=vllm_config.cache_config.block_size,
                                 num_kv_heads=16, head_size=192, head_size_v=128,
                                 dtype=torch.bfloat16)
