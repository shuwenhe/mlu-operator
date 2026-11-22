"""Export only the actual dense Layer 0 weights in packed-GEMM input order."""
from pathlib import Path
import sys

import numpy as np

from audit import load_headers


def main(root, output):
    tensors = load_headers(root)
    output.mkdir(parents=True, exist_ok=True)
    names = {"gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight",
             "down": "mlp.down_proj.weight", "input_gamma": "input_layernorm.weight",
             "post_gamma": "post_attention_layernorm.weight"}
    for alias, suffix in names.items():
        shard, offset, shape = tensors["model.layers.0." + suffix]
        matrix = np.memmap(shard, mode="r", offset=offset, dtype="<u2", shape=tuple(shape))
        np.ascontiguousarray(matrix.T).tofile(output / f"{alias}.bin")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
