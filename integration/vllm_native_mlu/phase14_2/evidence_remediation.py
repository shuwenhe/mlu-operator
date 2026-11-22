"""Phase 14.2 evidence-remediation harness for repeated TP=2 lifecycle."""
import argparse
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from ..phase14_1 import engine
from .executor import LifecycleExecutor


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _proc_snapshot(pid: int) -> dict[str, Any]:
    status_path = Path(f"/proc/{pid}/status")
    stat_path = Path(f"/proc/{pid}/stat")
    snapshot = {
        "observed": False,
        "pid": pid,
        "name": None,
        "state": None,
        "ppid": None,
        "threads": None,
        "start_time_ticks": None,
        "captured_at_utc": _utc_now(),
        "error": None,
    }
    try:
        status = {}
        for line in status_path.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                status[key.strip()] = value.strip()
        stat_parts = stat_path.read_text().split()
        snapshot.update({
            "observed": True,
            "name": status.get("Name"),
            "state": status.get("State"),
            "ppid": int(status["PPid"]) if "PPid" in status else None,
            "threads": int(status["Threads"]) if "Threads" in status else None,
            "start_time_ticks": int(stat_parts[21]) if len(stat_parts) >= 22 else None,
        })
    except BaseException as exc:  # pragma: no cover - runtime forensics only
        snapshot["error"] = repr(exc)
    return snapshot


def _thread_inventory(pid: int) -> dict[str, Any]:
    task_root = Path(f"/proc/{pid}/task")
    items = []
    for task in sorted(task_root.iterdir(), key=lambda p: int(p.name)):
        name = None
        comm = task / "comm"
        if comm.exists():
            name = comm.read_text().strip()
        items.append({"tid": int(task.name), "name": name})
    return {"count": len(items), "threads": items}


def _fd_type(target: str) -> str:
    if target.startswith("socket:["):
        return "socket"
    if target.startswith("pipe:["):
        return "pipe"
    if target.startswith("anon_inode:"):
        return "anon_inode"
    if target.startswith("eventfd"):
        return "eventfd"
    return "file_or_other"


def _fd_inventory(pid: int) -> dict[str, Any]:
    fd_root = Path(f"/proc/{pid}/fd")
    fdinfo_root = Path(f"/proc/{pid}/fdinfo")
    rows = []
    for fd_path in sorted(fd_root.iterdir(), key=lambda p: int(p.name)):
        fd = int(fd_path.name)
        target = None
        error = None
        try:
            target = os.readlink(fd_path)
        except BaseException as exc:
            error = repr(exc)
        fdinfo = {}
        info_file = fdinfo_root / fd_path.name
        if info_file.exists():
            for line in info_file.read_text().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fdinfo[key.strip()] = value.strip()
        rows.append({
            "fd": fd,
            "target": target,
            "fd_type": _fd_type(target or ""),
            "device": fdinfo.get("mnt_id"),
            "inode": fdinfo.get("ino"),
            "socket_identity": target if (target or "").startswith("socket:[") else None,
            "error": error,
        })
    return {"count": len(rows), "rows": rows}


def _children() -> list[int]:
    content = Path("/proc/thread-self/children").read_text().strip()
    if not content:
        return []
    return [int(x) for x in content.split()]


def _cnmon_capture(path: Path) -> dict[str, Any]:
    result = subprocess.run(["cnmon"], check=False, capture_output=True, text=True)
    output = result.stdout if result.stdout else result.stderr
    path.write_text(output)
    pid_matches = sorted({int(x) for x in re.findall(r"\bPID\s*[:=]?\s*(\d+)\b", output)})
    card_matches = sorted({int(x) for x in re.findall(r"\b(?:Card|Device|MLU)\s*[:#]?\s*(\d+)\b", output)})
    mem_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(MiB|GiB|MB|GB)", output)
    util_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", output)
    return {
        "command_exit_code": result.returncode,
        "no_running_processes": "No running processes found" in output,
        "device_ids": card_matches,
        "active_process_ids": pid_matches,
        "memory_samples": [{"value": float(v), "unit": u} for v, u in mem_matches],
        "utilization_samples_percent": [float(x) for x in util_matches],
        "metric_unavailable": not bool(mem_matches or util_matches),
        "raw_file": str(path),
    }


def _enginecore_lifecycle(created_llm) -> dict[str, Any]:
    core = created_llm.llm_engine.engine_core.engine_core
    lifecycle = {
        "architecture": {
            "module": type(core).__module__,
            "class": type(core).__name__,
        },
        "process_handle_attribute": None,
        "is_separate_process": False,
        "pid": None,
        "ppid": None,
        "start_time_ticks": None,
        "exitcode": None,
        "joined": None,
        "reaped": None,
        "notes": [],
    }
    proc_attr = None
    proc_obj = None
    for name in ("proc", "process", "_proc", "engine_proc"):
        value = getattr(core, name, None)
        if value is not None and hasattr(value, "pid") and hasattr(value, "is_alive"):
            proc_attr = name
            proc_obj = value
            break
    if proc_obj is None:
        lifecycle["notes"].append("No process handle on EngineCore object; treated as in-process/thread architecture.")
        return lifecycle
    lifecycle["process_handle_attribute"] = proc_attr
    lifecycle["is_separate_process"] = True
    lifecycle["pid"] = getattr(proc_obj, "pid", None)
    lifecycle["exitcode"] = getattr(proc_obj, "exitcode", None)
    if lifecycle["pid"] is not None:
        info = _proc_snapshot(lifecycle["pid"])
        lifecycle["ppid"] = info.get("ppid")
        lifecycle["start_time_ticks"] = info.get("start_time_ticks")
        lifecycle["reaped"] = not info.get("observed", False)
    try:
        proc_obj.join(timeout=0)
        lifecycle["joined"] = True
    except BaseException as exc:  # pragma: no cover - runtime forensics only
        lifecycle["joined"] = False
        lifecycle["notes"].append(f"join failed: {exc!r}")
    return lifecycle


def _record_cycle_event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10)
    args, forwarded = parser.parse_known_args()

    sys.argv[1:] = forwarded
    make_llm = engine.LLM
    dump = Path(os.environ["NEURX_TP2_DUMP"])
    parent_pid = os.getpid()
    lifecycle_log = dump.parent / "repeated_lifecycle.log"

    parent_identity = _proc_snapshot(parent_pid)
    _record_cycle_event(lifecycle_log, {"phase14_2": "parent_identity", **parent_identity})

    for cycle in range(args.cycles):
        cycle_dir = dump / str(cycle)
        cycle_dir.mkdir(parents=True)
        start = time.monotonic()
        generated = {"called": False, "parent_during_generate": None}
        created = []

        before = {
            "parent": _proc_snapshot(parent_pid),
            "children": _children(),
            "threads": _thread_inventory(parent_pid),
            "fds": _fd_inventory(parent_pid),
            "cnmon": _cnmon_capture(cycle_dir / "cnmon_before.log"),
        }

        os.environ["NEURX_TP2_DUMP"] = str(cycle_dir)

        def create(*positional, **kwargs):
            llm = make_llm(*positional, distributed_executor_backend=LifecycleExecutor, **kwargs)
            original_generate = llm.generate

            def wrapped_generate(*g_args, **g_kwargs):
                generated["called"] = True
                generated["parent_during_generate"] = _proc_snapshot(parent_pid)
                generated["children_during_generate"] = _children()
                return original_generate(*g_args, **g_kwargs)

            llm.generate = wrapped_generate
            created.append(llm)
            return llm

        engine.LLM = create
        failure = None
        try:
            engine.main()
        except BaseException as exc:
            failure = repr(exc)
            raise
        finally:
            engine.LLM = make_llm

        enginecore = None
        worker_lifecycle = []
        worker_exit_codes = None
        if created:
            enginecore = _enginecore_lifecycle(created[0])
            core = created[0].llm_engine.engine_core.engine_core
            executor = core.model_executor
            worker_lifecycle = list(getattr(executor, "worker_lifecycle", []))
            worker_exit_codes = getattr(executor, "worker_exit_codes", None)

        after = {
            "parent": _proc_snapshot(parent_pid),
            "children": _children(),
            "threads": _thread_inventory(parent_pid),
            "fds": _fd_inventory(parent_pid),
            "cnmon": _cnmon_capture(cycle_dir / "cnmon_after.log"),
        }
        event = {
            "phase14_2": "cycle",
            "cycle": cycle,
            "seconds": time.monotonic() - start,
            "parent_pid": parent_pid,
            "before": before,
            "during": generated,
            "after": after,
            "worker_exit_codes": worker_exit_codes,
            "worker_lifecycle": worker_lifecycle,
            "enginecore": enginecore,
            "failure": failure,
        }
        _record_cycle_event(lifecycle_log, event)
        print(json.dumps(event), flush=True)


if __name__ == "__main__":
    main()
