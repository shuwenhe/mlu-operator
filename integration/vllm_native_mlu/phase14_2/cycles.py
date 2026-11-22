"""Acceptance harness for repeated Vendor TP=2 creation and release."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from ..phase14_1 import engine
from .executor import LifecycleExecutor


class InjectedInitializationFailure(Exception):
    pass


class InitializationFailureExecutor(LifecycleExecutor):
    def _post_init_executor(self):
        raise InjectedInitializationFailure("fault injected after Vendor workers and queues are ready")


def resources():
    descriptors = {}
    for fd in Path("/proc/self/fd").iterdir():
        try:
            descriptors[fd.name] = os.readlink(fd)
        except FileNotFoundError:
            # The directory scan itself may have supplied a transient descriptor.
            continue
    return {
        "fds": len(descriptors),
        "sockets": [value for value in descriptors.values() if value.startswith("socket:")],
        "kernel_threads": len(os.listdir("/proc/self/task")),
        "python_threads": [thread.name for thread in threading.enumerate()],
        "children": Path("/proc/thread-self/children").read_text().split(),
    }


def cnmon(path):
    output = subprocess.run(["cnmon"], check=True, capture_output=True, text=True).stdout
    path.write_text(output)
    if "No running processes found" not in output:
        raise RuntimeError(f"MLU processes present at cycle boundary. See {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--double-shutdown", action="store_true")
    parser.add_argument("--initialization-failure", action="store_true")
    parser.add_argument("--distributed-fault", choices=("worker_loss", "collective_timeout"))
    args, forwarded = parser.parse_known_args()
    if args.distributed_fault:
        from .distributed_faults import exercise

        engine.exercise_worker_loss = partial(exercise, kind=args.distributed_fault)
        forwarded.append("--worker-loss")
    sys.argv[1:] = forwarded
    make_llm = engine.LLM
    dump = Path(os.environ["NEURX_TP2_DUMP"])

    for cycle in range(args.cycles):
        before = resources()
        start = time.monotonic()
        created = []
        path = dump / str(cycle)
        path.mkdir()
        cnmon(path / "cnmon_before.log")
        os.environ["NEURX_TP2_DUMP"] = str(path)
        executor_cls = InitializationFailureExecutor if args.initialization_failure else LifecycleExecutor

        def create(*positional, **kwargs):
            llm = make_llm(*positional, distributed_executor_backend=executor_cls, **kwargs)
            created.append(llm)
            return llm

        engine.LLM = create
        try:
            engine.main()
        except InjectedInitializationFailure:
            if not args.initialization_failure:
                raise
        else:
            if args.initialization_failure:
                raise AssertionError("Initialization fault did not reach Vendor executor")
            if args.double_shutdown:
                created[0].llm_engine.engine_core.shutdown()
        finally:
            engine.LLM = make_llm
        exit_codes = None
        if created:
            core = created[0].llm_engine.engine_core.engine_core
            executor = core.model_executor
            exit_codes = executor.worker_exit_codes
            if executor._owned_sockets or not executor._released or exit_codes is None:
                raise RuntimeError("Vendor executor retained owned resources")
        created.clear()
        cnmon(path / "cnmon_after.log")
        print(json.dumps(dict(phase14_2="cycle", cycle=cycle, seconds=time.monotonic()-start,
                              before=before, after=resources(), double=args.double_shutdown,
                              initialization_failure=args.initialization_failure,
                              worker_exit_codes=exit_codes)), flush=True)


if __name__ == "__main__":
    main()
