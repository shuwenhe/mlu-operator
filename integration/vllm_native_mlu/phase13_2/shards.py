"""Logical checkpoint offsets, independent of the compute layout."""


def row_span(size, rank, tp):
    if tp not in (1, 2, 4, 8) or not 0 <= rank < tp or size <= 0 or size % tp:
        raise ValueError(f"Invalid shard size={size}, rank={rank}, tp={tp}")
    width = size // tp
    return rank * width, width


def column_spans(sections, rank, tp):
    if not sections:
        raise ValueError("Expected at least one projection section")
    spans = []
    start = 0
    for size in sections:
        offset, width = row_span(size, rank, tp)
        spans.append((start + offset, width))
        start += size
    return tuple(spans)
