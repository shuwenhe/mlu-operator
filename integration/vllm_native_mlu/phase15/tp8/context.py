"""Prepare the native cache library and checkpoint-derived tables for extended TP8."""
import os
from pathlib import Path
import re
import subprocess

from ...phase12_5.prepare_rope import main as prepare_rope

MAX_CONTEXT = 161
PREFILL_CHUNK = 128
CACHE_PAGES = 16


def prepare_context(checkpoint, length, output):
    if length <= 128:
        return
    root = Path(output) / "native_context"
    root.mkdir()
    frozen = Path(os.environ["NATIVE_DEEPSEEK_ROPE"])
    prepare_rope(Path(checkpoint), root, max(131, length))
    for name in ("cos.bin", "sin.bin", "scale.txt"):
        if not (root / name).read_bytes().startswith((frozen / name).read_bytes()):
            raise ValueError(f"Extended YaRN changed the frozen prefix in {name}")
    native = Path(__file__).resolve().parents[2] / "phase12/paged_kv.mlu"
    source = root / "paged_two.mlu"
    text, substitutions = re.subn(r"constexpr int heads = 16;", "constexpr int heads = 2;",
                                  native.read_text())
    if substitutions != 1:
        raise ValueError("Unexpected paged KV head declaration")
    source.write_text(text)
    sdk = Path("/usr/local/neuware")
    obj = root / "paged_two.o"
    library = root / "libpaged_two.so"
    subprocess.run([str(sdk / "bin/cncc"), "--bang-mlu-arch=mtp_592", "-std=c++11",
                    "-O2", "-fPIC", f"-DNATIVE_KV_MAX_TOKENS={length}", "-c",
                    str(source), "-o", str(obj)], check=True)
    subprocess.run(["g++", "-shared", str(obj), "-L" + str(sdk / "lib64"),
                    "-Wl,-rpath," + str(sdk / "lib64"), "-lcnrt", "-o", str(library)], check=True)
    os.environ["NATIVE_DEEPSEEK_ROPE"] = str(root)
    os.environ["NEURX_PAGED_TWO"] = str(library)
