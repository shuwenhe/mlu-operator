"""Materialized cache sizing and worker raw-buffer view for DeepSeek-V2-Lite."""

import torch
import torch_mlu
from vllm.v1.kv_cache_interface import FullAttentionSpec

HEADS = 16
KEY_WIDTH = 192
VALUE_WIDTH = 128
LAYERS = 27


def per_layer_specs(layer_names, block_size):
    """Return separate layer entries eligible for the same full-attention group."""
    names = tuple(layer_names)
    if len(names) != LAYERS or len(set(names)) != LAYERS or block_size <= 0:
        raise ValueError("DeepSeek-V2-Lite requires 27 distinct cache layers and a positive block size")
    return {
        name: FullAttentionSpec(block_size=block_size, num_kv_heads=HEADS,
                                head_size=KEY_WIDTH, head_size_v=VALUE_WIDTH, dtype=torch.bfloat16)
        for name in names
    }


def reshape_kv_cache(raw, spec, num_blocks, kernel_block_size, cache_dtype):
    """Implement the vendor worker reshape signature for an unquantized raw byte tensor.

    All physical K slots precede all physical V slots. The two device views
    share the single engine allocation, and use the same physical slot IDs.
    """
    if (
        type(spec) is not FullAttentionSpec
        or (spec.num_kv_heads, spec.head_size, spec.head_size_v) != (HEADS, KEY_WIDTH, VALUE_WIDTH)
        or spec.dtype != torch.bfloat16 or cache_dtype not in ("auto", "bfloat16")
        or num_blocks <= 0 or kernel_block_size != spec.block_size
        or raw.device.type != "mlu" or raw.dtype != torch.int8 or raw.ndim != 1
        or not raw.is_contiguous() or raw.numel() != num_blocks * spec.page_size_bytes
        or spec.page_size_bytes != spec.block_size * HEADS * (KEY_WIDTH + VALUE_WIDTH) * 2
        or raw.storage_offset() % 2
    ):
        raise ValueError("Incompatible native MLA cache allocation")
    slots = num_blocks * spec.block_size
    words = raw.view(torch.bfloat16)
    keys, values = words.split((slots * HEADS * KEY_WIDTH, slots * HEADS * VALUE_WIDTH))
    return keys.view(slots, HEADS, KEY_WIDTH), values.view(slots, HEADS, VALUE_WIDTH)
