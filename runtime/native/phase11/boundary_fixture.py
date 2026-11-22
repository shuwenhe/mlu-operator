"""Prepare metadata and an independent eager reference for the model boundaries."""
from pathlib import Path
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

from audit import load_headers


def main(root, output):
    output.mkdir(parents=True, exist_ok=True)
    tensors = load_headers(root)
    shards = sorted({data[0] for data in tensors.values()})
    with (output/"checkpoint.txt").open("w") as info:
        print(len(shards), file=info)
        for path in shards:
            start = min(t[1] for t in tensors.values() if t[0] == path)
            print(path.name, start, path.stat().st_size-start, file=info)
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            path, offset, _ = tensors[name]
            base = min(t[1] for t in tensors.values() if t[0] == path)
            print(shards.index(path), offset-base, file=info)

    def tensor(name):
        path, offset, shape = tensors[name]
        mmap = np.memmap(path, dtype="<u2", mode="c", offset=offset, shape=tuple(shape))
        return torch.from_numpy(mmap).view(torch.bfloat16)

    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    torch.set_num_threads(2)
    ids = tok.encode("The capital of France is", add_special_tokens=True).ids
    assert ids and len(ids)<=128
    np.asarray(ids, dtype="<u4").tofile(output/"ids.bin")
    for count in (1,len(ids)):
        emb = tensor("model.embed_tokens.weight")[ids[:count]].contiguous()
        wide = emb.float()
        norm = (wide * (wide.square().mean(dim=-1, keepdim=True)+1e-6).rsqrt()).bfloat16()
        norm = norm * tensor("model.norm.weight")
        # The projection is an isolated test. Its input is normalized embeddings.
        logits = torch.nn.functional.linear(norm,tensor("lm_head.weight")).float()
        for name, value in (("embedding",emb),("normalized",norm),("logits",logits)):
            kind = torch.int16 if value.dtype == torch.bfloat16 else torch.float32
            value.contiguous().view(kind).numpy().tofile(output/f"{name}-{count}.bin")
        logits.argmax(-1).numpy().astype("<u4").tofile(output/f"argmax-{count}.bin")
        print("boundary_reference", count, logits.shape, logits.argmax(-1).tolist())


if __name__ == "__main__":
    main(*map(Path,sys.argv[1:]))
