"""Read-only safetensors header and byte reconstruction audit."""
from contextlib import ExitStack
import csv
import hashlib
import json
import mmap
from pathlib import Path
import re

repo = Path(__file__).resolve().parents[3]
model = repo.parent / "model/DeepSeek-V2-Lite"
index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
rules = {
    "q_proj.weight": ((3072, 2048), 0),
    "kv_a_proj_with_mqa.weight": ((576, 2048), None),
    "kv_a_layernorm.weight": ((512,), None),
    "kv_b_proj.weight": ((4096, 512), 0),
    "o_proj.weight": ((2048, 2048), 1),
}
pattern = re.compile(r"model.layers.(\d+).self_attn\.(.+)")
names = {name for name in index if pattern.fullmatch(name)}
assert names == {f"model.layers.{layer}.self_attn.{name}"
                 for layer in range(27) for name in rules}, names
rows = []
with ExitStack() as stack:
    shards = {}
    for file in set(index[name] for name in names):
        handle = stack.enter_context((model / file).open("rb"))
        mm = stack.enter_context(mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
        offset = 8 + int.from_bytes(mm[:8], "little")
        shards[file] = (mm, offset, json.loads(mm[8:offset]))
    for name in sorted(names, key=lambda name: (int(pattern.fullmatch(name)[1]), name)):
        layer, param = pattern.fullmatch(name).groups()
        expected, axis = rules[param]
        file = index[name]
        mm, offset, headers = shards[file]
        header = headers[name]
        shape = tuple(header["shape"])
        assert shape == expected and header["dtype"] == "BF16", (name, header)
        begin, end = (offset + i for i in header["data_offsets"])
        original = mm[begin:end]
        width = shape[axis] // 8 if axis is not None else None
        sha = hashlib.sha256(original).hexdigest()
        if axis == 0:
            stride = 2 * shape[1]
            pieces = [mm[begin + r * width * stride:begin + (r + 1) * width * stride] for r in range(8)]
            recovered = b"".join(pieces)
        elif axis == 1:
            stride = shape[1] * 2
            pieces = [b"".join(mm[pos + r * width * 2:pos + (r + 1) * width * 2]
                               for pos in range(begin, end, stride)) for r in range(8)]
            recovered = b"".join(pieces[r][i:i + width * 2]
                                 for i in range(0, len(pieces[0]), width * 2) for r in range(8))
        else:
            pieces = [original] * 8
            recovered = pieces[0]
        assert recovered == original, name
        for rank, piece in enumerate(pieces):
            local = list(shape)
            if axis is not None:
                local[axis] = width
            rows.append(dict(layer=layer, parameter=name, dtype=header["dtype"], shard=file,
                             original_shape=json.dumps(shape), partition_axis=axis, rank=rank,
                             local_shape=json.dumps(local), offset=rank * width if width else 0,
                             replication=axis is None,
                             reconstruction="replica" if axis is None else f"concat_axis_{axis}",
                             data_byte_start=begin, data_byte_end=end, original_sha256=sha,
                             local_sha256=hashlib.sha256(piece).hexdigest(), result="PASS"))
with (repo / "artifacts/logs/phase13_3_1_checkpoint_inventory.csv").open("x", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(names)} tensors; {len(rows)} rank records; exact byte reconstruction PASS")
