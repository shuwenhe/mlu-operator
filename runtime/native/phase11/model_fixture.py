"""Prepare native model metadata and prompt IDs from the original local tokenizer."""
from pathlib import Path
import sys
import numpy as np
from tokenizers import Tokenizer
from audit import load_headers

model, output = map(Path, sys.argv[1:3])
output.mkdir(parents=True, exist_ok=True)
with (output/"tensors.txt").open("w") as manifest:
    for name, (shard, offset, shape) in sorted(load_headers(model).items()):
        print(name,shard.name,offset,shape[0],shape[1] if len(shape)==2 else 0,file=manifest)
tokenizer = Tokenizer.from_file(str(model/"tokenizer.json"))
prompt = sys.argv[3]
ids = tokenizer.encode(prompt,add_special_tokens=True).ids
np.asarray(ids,dtype="<u4").tofile(output/"prompt.bin")
(output/"prompt.txt").write_text(prompt+"\n")
print("prompt ids",ids)
