"""Render the greedy token IDs emitted by the native runner."""
from pathlib import Path
import sys
from tokenizers import Tokenizer

checkpoint, log = map(Path, sys.argv[1:])
tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
tokens = [int(line.split()[2]) for line in log.read_text().splitlines()
          if line.startswith("generated ")]
if not tokens:
    raise ValueError("The native log contains no generated tokens")
for token in tokens:
    if tokenizer.id_to_token(token) is None:
        raise ValueError(f"Generated padded vocabulary ID {token} has no tokenizer entry")
print(tokenizer.decode(tokens))
