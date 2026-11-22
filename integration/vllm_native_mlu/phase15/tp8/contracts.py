"""DeepSeek-V2-Lite TP8 checkpoint ownership and cache geometry."""
from ...phase13_3_2.mapping import shard

TP = 8
HEADS = 2
BLOCK_SIZE = 16


def spans(name, rank):
    return shard(name, rank, TP)


def layout(blocks, block_size):
    if blocks <= 0 or block_size != BLOCK_SIZE:
        raise ValueError("Expected positive count of 16-token pages")
    slots = blocks * block_size
    keys = slots * HEADS * 192
    return slots, keys, keys + slots * HEADS * 128


def expert_owner(expert):
    if not 0 <= expert < 64:
        raise ValueError(f"Invalid routed expert {expert}")
    return expert // 8
