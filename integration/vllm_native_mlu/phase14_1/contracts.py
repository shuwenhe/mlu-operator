"""Frozen TP=2 layout boundaries, independent of device runtime."""


def layout(blocks, block_size):
    if blocks <= 0 or block_size != 16:
        raise ValueError("Expected positive count of 16-token pages")
    slots = blocks * block_size
    keys = slots * 8 * 192
    return slots, keys, keys + slots * 8 * 128


def spans(name, rank):
    if rank not in (0, 1):
        raise ValueError("TP=2 rank required")
    axis, width = {
        "q_proj.weight": (0, 1536), "kv_b_proj.weight": (0, 2048),
        "o_proj.weight": (1, 1024), "kv_a_proj_with_mqa.weight": (None, 576),
        "kv_a_layernorm.weight": (None, 512),
    }[name]
    return axis, rank * width if axis is not None else 0, width
