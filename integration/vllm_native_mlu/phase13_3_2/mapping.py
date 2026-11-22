"""Audited checkpoint shapes and head-contiguous TP offsets."""

RULES = {
    "q_proj.weight": ((3072, 2048), 0),
    "kv_a_proj_with_mqa.weight": ((576, 2048), None),
    "kv_a_layernorm.weight": ((512,), None),
    "kv_b_proj.weight": ((4096, 512), 0),
    "o_proj.weight": ((2048, 2048), 1),
}


def shard(name, rank, tp):
    if tp not in (1, 2, 4, 8) or not 0 <= rank < tp:
        raise ValueError("Invalid MLA TP rank or world size")
    shape, axis = RULES[name]
    if axis is None:
        return None, 0, shape[0]
    width = shape[axis] // tp
    return axis, rank * width, width
