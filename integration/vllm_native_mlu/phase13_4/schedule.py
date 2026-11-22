"""Exclusive logical page ownership shared by the ranks."""
import hashlib
import json
import struct


BLOCK_SIZE = 16
MAX_LENGTH = 128


def digest(value):
    text = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return list(struct.unpack("<8i", hashlib.sha256(text).digest()))


class Pages:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("Expected positive page capacity")
        self.free = set(range(capacity))
        self.requests = {}

    def extend(self, request, tokens):
        old, table = self.requests.get(request, (0, []))
        end = old + tokens
        need = (end+BLOCK_SIZE-1)//BLOCK_SIZE - len(table)
        if not 0 < tokens <= end <= MAX_LENGTH or need > len(self.free):
            raise ValueError("Invalid context length or insufficient pages")
        picked = sorted(self.free)[:need]
        self.free.difference_update(picked)
        table = table + picked
        self.requests[request] = (end, table)
        return [table[p//BLOCK_SIZE]*BLOCK_SIZE+p%BLOCK_SIZE for p in range(old,end)]

    def release(self, request):
        _, table = self.requests.pop(request)
        self.free.update(table)
        return table
