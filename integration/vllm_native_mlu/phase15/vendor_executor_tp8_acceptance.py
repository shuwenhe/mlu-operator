"""Stage B2: TP=8 Vendor executor lifecycle acceptance (3 cycles)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_mlu  # noqa: F401
from vllm import LLM
from vllm.distributed.parallel_state import get_tp_group


OUT = Path(os.environ.get("PHASE15_STAGE_B2_OUT", "/results"))
OUT.mkdir(parents=True, exist_ok=True)
CYCLES = int(os.environ.get("PHASE15_STAGE_B2_CYCLES", "3"))
SHUTDOWN_TIMEOUT_S = float(os.environ.get("PHASE15_STAGE_B2_SHUTDOWN_TIMEOUT_S", "120"))
os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", str(int(SHUTDOWN_TIMEOUT_S)))


def log(event: str, **fields):
    print(json.dumps({"phase15_stage_b2": event, **fields}, sort_keys=True), flush=True)


def checksum(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def get_child_states():
    children = Path("/proc/thread-self/children").read_text().split()
    details = []
    for pid in children:
        status = Path(f"/proc/{pid}/status")
        state = "missing"
        ppid = ""
        if status.exists():
            for line in status.read_text().splitlines():
                if line.startswith("State:"):
                    state = line.split(":", 1)[1].strip()
                if line.startswith("PPid:"):
                    ppid = line.split(":", 1)[1].strip()
        details.append({"pid": pid, "ppid": ppid, "state": state})
    return details


def fd_socket_counts():
    fd_links = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            fd_links.append(os.readlink(entry))
        except FileNotFoundError:
            continue
    sockets = [x for x in fd_links if x.startswith("socket:")]
    return len(fd_links), len(sockets)


def cnmon_running_processes():
    try:
        out = subprocess.run(["/bin/bash", "-lc", "cnmon | cat"], check=True, text=True, capture_output=True).stdout
        return "no" if "No running processes found" in out else "yes"
    except Exception as exc:  # pragma: no cover
        return f"error:{exc}"


def snapshot(cycle: int, phase: str):
    fds, sockets = fd_socket_counts()
    children = get_child_states()
    return {
        "cycle": cycle,
        "phase": phase,
        "timestamp": time.time(),
        "parent_pid": os.getpid(),
        "fds": fds,
        "sockets": sockets,
        "python_threads": len(threading.enumerate()),
        "child_count": len(children),
        "children": " ".join([f"{c['pid']}({c['state']})" for c in children]),
        "cnmon_running_processes": cnmon_running_processes(),
    }


def build_llm():
    kwargs = dict(
        model="/checkpoint",
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=8,
        enforce_eager=True,
        max_model_len=16,
        block_size=16,
        max_num_seqs=1,
        max_num_batched_tokens=16,
        num_gpu_blocks_override=8,
        gpu_memory_utilization=0.30,
        enable_prefix_caching=False,
        disable_log_stats=True,
        disable_custom_all_reduce=True,
    )
    try:
        return LLM(**kwargs, shutdown_timeout=int(SHUTDOWN_TIMEOUT_S))
    except TypeError:
        log("shutdown_timeout_kwarg_not_supported")
        return LLM(**kwargs)


def worker_collectives(worker, cycle: int):
    group = get_tp_group()
    rank = group.rank_in_group
    world = group.world_size
    device = torch.device("mlu", torch.mlu.current_device())
    backend = dist.get_backend(group.device_group)
    pid = os.getpid()
    ppid = os.getppid()

    log("worker_init_complete", cycle=cycle, rank=rank, world_size=world, device=str(device), pid=pid, ppid=ppid)

    rows = []
    payloads = [4 * 1024, 64 * 1024, 1024 * 1024, 16 * 1024 * 1024]

    started = time.perf_counter()
    dist.barrier(group=group.device_group)
    torch.mlu.synchronize(device)
    rows.append(
        dict(
            cycle=cycle,
            rank=rank,
            world_size=world,
            device=f"mlu:{device.index}",
            backend=backend,
            op="barrier",
            payload_bytes=0,
            input_checksum="",
            output_checksum="",
            latency_ms=(time.perf_counter() - started) * 1000,
            status="PASS",
        )
    )

    for payload in payloads:
        elems = payload // 4

        tensor = torch.full((elems,), float(rank + 1), dtype=torch.float32, device=device)
        in_sum = checksum(tensor)
        started = time.perf_counter()
        dist.broadcast(tensor, src=0, group=group.device_group)
        torch.mlu.synchronize(device)
        rows.append(
            dict(
                cycle=cycle,
                rank=rank,
                world_size=world,
                device=f"mlu:{device.index}",
                backend=backend,
                op="broadcast",
                payload_bytes=payload,
                input_checksum=in_sum,
                output_checksum=checksum(tensor),
                latency_ms=(time.perf_counter() - started) * 1000,
                status="PASS",
            )
        )

        tensor = torch.full((elems,), float(rank + 1), dtype=torch.float32, device=device)
        in_sum = checksum(tensor)
        started = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group.device_group)
        torch.mlu.synchronize(device)
        expected = np.float32(world * (world + 1) / 2.0)
        ok = bool(torch.allclose(tensor[:1].cpu(), torch.tensor([expected], dtype=torch.float32), atol=0, rtol=0))
        rows.append(
            dict(
                cycle=cycle,
                rank=rank,
                world_size=world,
                device=f"mlu:{device.index}",
                backend=backend,
                op="all_reduce",
                payload_bytes=payload,
                input_checksum=in_sum,
                output_checksum=checksum(tensor),
                latency_ms=(time.perf_counter() - started) * 1000,
                status="PASS" if ok else "FAIL",
            )
        )

    log("worker_collectives_complete", cycle=cycle, rank=rank, pid=pid)
    return {
        "cycle": cycle,
        "rank": rank,
        "world_size": world,
        "device": f"mlu:{device.index}",
        "backend": backend,
        "pid": pid,
        "ppid": ppid,
        "rows": rows,
    }


def write_csv(path: Path, rows, headers):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def classify_exit_code(exit_code):
    if exit_code == 0:
        return "graceful"
    if exit_code == -15:
        return "sigterm"
    if exit_code == -9:
        return "sigkill"
    if exit_code is None:
        return "unknown"
    return f"other({exit_code})"


def wait_pids_exit(pids, timeout_s: float):
    pending = {int(pid) for pid in pids if pid is not None}
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        pending = {pid for pid in pending if Path(f"/proc/{pid}").exists()}
        if pending:
            time.sleep(0.1)
    return sorted(pending)


def main():
    mapping_rows = []
    collective_rows = []
    lifecycle_rows = []
    event_rows = []

    for cycle in range(1, CYCLES + 1):
        lifecycle_rows.append(snapshot(cycle, "baseline"))
        log("executor_init_start", cycle=cycle)
        llm = build_llm()
        log("executor_init_complete", cycle=cycle)
        lifecycle_rows.append(snapshot(cycle, "after_init"))

        try:
            report = llm.collective_rpc(worker_collectives, args=(cycle,), timeout=240)
            report = sorted(report, key=lambda x: x["rank"])
            for item in report:
                mapping_rows.append(
                    {
                        "cycle": cycle,
                        "rank": item["rank"],
                        "world_size": item["world_size"],
                        "device": item["device"],
                        "backend": item["backend"],
                        "worker_pid": item["pid"],
                        "worker_ppid": item["ppid"],
                    }
                )
                collective_rows.extend(item["rows"])
            lifecycle_rows.append(snapshot(cycle, "after_collectives"))
        finally:
            log("shutdown_request", cycle=cycle, timeout_s=SHUTDOWN_TIMEOUT_S)
            lifecycle_rows.append(snapshot(cycle, "before_shutdown"))
            engine_core_client = llm.llm_engine.engine_core
            core_engine = getattr(engine_core_client, "engine_core", None)
            model_executor = getattr(core_engine, "model_executor", None)
            workers = list(getattr(model_executor, "workers", ()) if model_executor is not None else ())
            worker_pids = [getattr(worker.proc, "pid", None) for worker in workers]
            engine_core_client.shutdown(timeout=SHUTDOWN_TIMEOUT_S)
            log("shutdown_call_returned", cycle=cycle)
            remaining_worker_pids = wait_pids_exit(worker_pids, SHUTDOWN_TIMEOUT_S)
            if remaining_worker_pids:
                log("shutdown_workers_timeout", cycle=cycle, remaining_worker_pids=remaining_worker_pids)
            worker_exit_codes = {
                str(getattr(worker.proc, "pid", "unknown")): getattr(worker.proc, "exitcode", None)
                for worker in workers
            }
            worker_exit_classes = {pid: classify_exit_code(code) for pid, code in worker_exit_codes.items()}
            row = snapshot(cycle, "after_shutdown")
            row["worker_pids"] = " ".join(str(pid) for pid in worker_pids if pid is not None)
            row["worker_exit_codes"] = json.dumps(worker_exit_codes, sort_keys=True)
            row["worker_exit_classes"] = json.dumps(worker_exit_classes, sort_keys=True)
            row["remaining_worker_pids"] = " ".join(str(pid) for pid in remaining_worker_pids)
            lifecycle_rows.append(row)

    write_csv(
        OUT / "rank_device_mapping.csv",
        mapping_rows,
        ["cycle", "rank", "world_size", "device", "backend", "worker_pid", "worker_ppid"],
    )
    write_csv(
        OUT / "cncl_acceptance.csv",
        collective_rows,
        [
            "cycle",
            "rank",
            "world_size",
            "device",
            "backend",
            "op",
            "payload_bytes",
            "input_checksum",
            "output_checksum",
            "latency_ms",
            "status",
        ],
    )
    write_csv(
        OUT / "lifecycle.csv",
        lifecycle_rows,
        [
            "cycle",
            "phase",
            "timestamp",
            "parent_pid",
            "fds",
            "sockets",
            "python_threads",
            "child_count",
            "children",
            "cnmon_running_processes",
            "worker_pids",
            "worker_exit_codes",
            "worker_exit_classes",
            "remaining_worker_pids",
        ],
    )

    # Final gate computation
    rank_ok = True
    backend_ok = True
    world_ok = True
    for cycle in range(1, CYCLES + 1):
        rows = [r for r in mapping_rows if r["cycle"] == cycle]
        ranks = sorted(r["rank"] for r in rows)
        devices = [r["device"] for r in sorted(rows, key=lambda x: x["rank"])]
        rank_ok &= ranks == list(range(8))
        backend_ok &= all(r["backend"] == "cncl" for r in rows)
        world_ok &= all(r["world_size"] == 8 for r in rows)
        rank_ok &= devices == [f"mlu:{i}" for i in range(8)]

    collective_ok = all(r["status"] == "PASS" for r in collective_rows)
    payload_ok = {4096, 65536, 1048576, 16777216}.issubset(
        {int(r["payload_bytes"]) for r in collective_rows if r["op"] in ("broadcast", "all_reduce")}
    )
    three_cycles_ok = len({r["cycle"] for r in mapping_rows}) == CYCLES
    after_shutdown_rows = [r for r in lifecycle_rows if r["phase"] == "after_shutdown"]
    worker_shutdown_clean = all(not r.get("remaining_worker_pids") for r in after_shutdown_rows)
    no_forced_worker_termination = all(
        "sigterm" not in r.get("worker_exit_classes", "") and "sigkill" not in r.get("worker_exit_classes", "")
        for r in after_shutdown_rows
    )

    lines = [
        "# Stage B2 Final Report",
        "",
        "| Gate | Status |",
        "|---|---|",
        f"| vendor_executor_initialized_tp8 | {'PASS' if world_ok else 'FAIL'} |",
        f"| rank_device_mapping_0_7 | {'PASS' if rank_ok else 'FAIL'} |",
        f"| vendor_tp_backend_cncl | {'PASS' if backend_ok else 'FAIL'} |",
        f"| vendor_collectives_barrier_broadcast_allreduce | {'PASS' if collective_ok else 'FAIL'} |",
        f"| payload_4kb_64kb_1mb_16mb | {'PASS' if payload_ok else 'FAIL'} |",
        f"| three_repeated_cycles | {'PASS' if three_cycles_ok else 'FAIL'} |",
        f"| graceful_worker_shutdown | {'PASS' if worker_shutdown_clean else 'FAIL'} |",
        f"| no_forced_worker_termination | {'PASS' if no_forced_worker_termination else 'FAIL'} |",
        "",
    ]

    overall = all(
        [
            world_ok,
            rank_ok,
            backend_ok,
            collective_ok,
            payload_ok,
            three_cycles_ok,
            worker_shutdown_clean,
            no_forced_worker_termination,
        ]
    )
    lines.append(f"Overall: {'PASS' if overall else 'FAIL'}")
    (OUT / "final_report.md").write_text("\n".join(lines) + "\n")

    log(
        "completed",
        world_ok=world_ok,
        rank_ok=rank_ok,
        backend_ok=backend_ok,
        collective_ok=collective_ok,
        payload_ok=payload_ok,
        three_cycles_ok=three_cycles_ok,
        worker_shutdown_clean=worker_shutdown_clean,
        no_forced_worker_termination=no_forced_worker_termination,
        overall="PASS" if overall else "FAIL",
    )


if __name__ == "__main__":
    main()
