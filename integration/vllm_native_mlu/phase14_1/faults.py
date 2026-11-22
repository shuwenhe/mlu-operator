"""Fault injection for an isolated length-one engine acceptance process."""
from types import SimpleNamespace


FAULTS = ("invalid-block", "invalid-slot", "divergent", "missing-metadata")


def alter(metadata, kind, blocks):
    if kind not in FAULTS[:3]:
        raise ValueError("Invalid schedule fault kind")
    if metadata.num_reqs != 1 or metadata.seq_lens[0].item() != 1:
        raise ValueError("Only isolated length-one fault fixtures are supported")
    table, slots = metadata.block_table_tensor.clone(), metadata.slot_mapping.clone()
    if kind == "invalid-block":
        table[0,0] = blocks
    elif kind == "invalid-slot":
        slots[0] = -1
    else:
        if blocks < 2:
            raise ValueError("Rank divergence fixture requires two blocks")
        alternate = (int(table[0,0])+1)%blocks
        table[0,0] = alternate
        slots[0] = alternate*16
    return SimpleNamespace(num_reqs=metadata.num_reqs,seq_lens=metadata.seq_lens,
                           query_start_loc=metadata.query_start_loc,block_table_tensor=table,
                           slot_mapping=slots)
