"""Verify official loader shard records against checkpoint bytes without importing torch."""
from contextlib import ExitStack
import hashlib
import json
import mmap
from pathlib import Path
import sys

from contracts import spans
from inspect_run import records


def main(checkpoint, log):
    index=json.loads((checkpoint/"model.safetensors.index.json").read_text())["weight_map"]
    rows=[r for r in records(log) if r["phase14_1"]=="shard"]
    assert len(rows)==270 and len({(r["rank"],r["parameter"]) for r in rows})==270
    with ExitStack() as context:
        files={}
        for row in rows:
            name=row["parameter"]
            filename=index[name]
            if filename not in files:
                handle=context.enter_context((checkpoint/filename).open("rb"))
                raw=context.enter_context(mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ))
                start=8+int.from_bytes(raw[:8],"little")
                files[filename]=raw,start,json.loads(raw[8:start])
            raw,start,headers=files[filename]
            header=headers[name]
            assert header["dtype"]=="BF16"
            shape=header["shape"]
            low,high=(start+x for x in header["data_offsets"])
            axis,offset,width=spans(name.split(".self_attn.",1)[1],row["rank"])
            assert (axis,offset)==(row["axis"],row["offset"])
            want=shape.copy()
            if axis is not None:
                want[axis]=width
            assert want==row["shape"]
            if axis==0:
                stride=2*shape[1]
                data=raw[low+offset*stride:low+(offset+width)*stride]
            elif axis==1:
                stride=2*shape[1]
                data=b"".join(raw[i+2*offset:i+2*(offset+width)] for i in range(low,high,stride))
            else:
                data=raw[low:high]
            assert hashlib.sha256(data).hexdigest()==row["sha256"],(name,row["rank"])
    print(f"All {len(rows)} rank-local checkpoint slice records match byte offsets and SHA-256")


if __name__=="__main__":
    main(*map(Path,sys.argv[1:3]))
