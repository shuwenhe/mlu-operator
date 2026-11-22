"""Run one fault and one fresh-engine frozen generation in the same process."""
import argparse
import json
import os
from pathlib import Path
import sys

from .cycles import main as run_cycles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=("worker_loss", "collective_timeout", "invalid-block",
                                             "invalid-slot", "divergent", "missing-metadata"))
    args = parser.parse_args()
    root = Path(os.environ["NEURX_TP2_DUMP"])
    for name in ("fault", "recovered"):
        path = root / name
        path.mkdir()
        os.environ["NEURX_TP2_DUMP"] = str(path)
        if name == "fault":
            option = "--distributed-fault" if "_" in args.scenario else "--negative"
            sys.argv[1:] = [option, args.scenario]
        else:
            sys.argv[1:] = []
        print(json.dumps(dict(phase14_2="recovery_stage", stage=name,
                              scenario=args.scenario)), flush=True)
        run_cycles()


if __name__ == "__main__":
    main()
