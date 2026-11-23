"""Translate worker metadata for the native materialized-attention contract."""

from dataclasses import dataclass

import torch

from ..phase12.paged_kv import PagedKV


@dataclass(frozen=True)
class RequestSpan:
    request_id: str
    start: int
    end: int
    length: int
    blocks: int


def validate_batch(metadata, request_ids, positions, block_size, num_blocks, *, max_length=131):
    """Validate active metadata before any device cache writes.

    Only small scheduler metadata is read back. Cache buffers and activations
    remain on device. This initial path assumes exclusive blocks per request.
    """
    offsets = metadata.query_start_loc
    lengths = metadata.seq_lens
    table = metadata.block_table_tensor
    slots = metadata.slot_mapping
    active = len(request_ids)
    if (
        active == 0 or len(set(request_ids)) != active or block_size <= 0 or num_blocks <= 0
        or offsets.ndim != 1 or offsets.numel() < active + 1
        or lengths.ndim != 1 or lengths.numel() < active
        or table.ndim != 2 or table.shape[0] < active or table.dtype != torch.int32
        or slots.ndim != 1 or slots.dtype != torch.int32
        or positions.ndim != 1 or positions.dtype != torch.int32
        or any(x.device != positions.device for x in (table, slots))
    ):
        raise ValueError("Invalid scheduler metadata layout")
    starts = offsets[:active + 1].tolist()
    if starts[0] != 0 or starts[-1] > min(slots.numel(), positions.numel()):
        raise ValueError("Invalid scheduled token span")
    positions_cpu = positions[:starts[-1]].tolist()
    slots_cpu = slots[:starts[-1]].tolist()
    block_rows = table[:active].tolist()
    used = set()
    result = []
    for i, (request, length) in enumerate(zip(request_ids, lengths[:active].tolist())):
        start, end = starts[i:i + 2]
        count = end - start
        needed = (length + block_size - 1) // block_size
        blocks = block_rows[i][:needed]
        if not 0 < count <= length <= max_length or count > 128 or len(blocks) != needed:
            raise ValueError(f"{request} has invalid token or context length")
        owned = set(blocks)
        if len(owned) != needed or used.intersection(owned) or any(
            block < 0 or block >= num_blocks for block in blocks
        ):
            raise ValueError(f"{request} has invalid or shared physical block ownership")
        used.update(owned)
        position_range = range(length - count, length)
        expected_slots = [
            blocks[position // block_size] * block_size + position % block_size
            for position in position_range
        ]
        if positions_cpu[start:end] != list(position_range) or slots_cpu[start:end] != expected_slots:
            raise ValueError(f"{request} positions and physical slots disagree with its block table")
        result.append(RequestSpan(request, start, end, length, needed))
    return result


class ScheduledAttention:
    """Append projected K/V, gather each committed prefix, and enqueue native attention."""

    def __init__(self, cache: PagedKV, native_attention):
        self.cache = cache
        self.native_attention = native_attention

    def __call__(self, metadata, request_ids, positions, q, k, v, scale):
        spans = validate_batch(
            metadata, request_ids, positions, self.cache.block_size,
            self.cache.keys.shape[0] // self.cache.block_size,
        )
        tokens = spans[-1].end
        if (
            not 0 < scale or q.shape != (tokens, 16, 192)
            or k.shape != q.shape or v.shape != (tokens, 16, 128)
            or any(t.device != self.cache.keys.device or t.dtype != self.cache.keys.dtype
                   for t in (q, k, v))
            or positions.device != q.device
        ):
            raise ValueError("Projected QKV does not match the active batch")
        outputs = []
        for row, span in enumerate(spans):
            token_slice = slice(span.start, span.end)
            # Device-only layout materialization. Q/K/V projection remains a model responsibility.
            query, key, value = (
                t[token_slice].transpose(0, 1).contiguous() for t in (q, k, v)
            )
            self.cache.write(key, value, metadata.slot_mapping[token_slice].contiguous())
            keys, values = self.cache.read(
                metadata.block_table_tensor[row, :span.blocks].contiguous(), span.length
            )
            _, _, _, result = self.native_attention(
                query, keys, values, positions[token_slice].contiguous(), 0, span.length, scale
            )
            outputs.append(result.transpose(0, 1))
        return torch.cat(outputs, dim=0)
