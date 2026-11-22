"""Acceptance-only CNCL faults on the production rank group."""
from datetime import timedelta
import json
import os
import re
from pathlib import Path
import time

from ..phase14_1.engine import SamplingParams, prompt


def enter_collective(worker, kind):
    import torch
    import torch.distributed as dist

    model = worker.model_runner.model
    group = model.group
    rank = group.rank_in_group
    status = [None]
    value = torch.ones(1, device=model.device)
    dist.barrier(group=group.cpu_group)
    print(json.dumps(dict(phase14_2="fault_enter", kind=kind, rank=rank,
                          backend=dist.get_backend(group.device_group))), flush=True)
    if rank == 1 and kind == "worker_loss":
        os._exit(71)
    if rank == 0:
        submitted = False
        started = time.monotonic()
        try:
            work = dist.all_reduce(value, group=group.device_group, async_op=True)
            submitted = True
            started = time.monotonic()
            completed = work.wait(timeout=timedelta(seconds=3))
            if not completed:
                status[0] = "CNCL Work.wait returned timeout"
        except (RuntimeError, TimeoutError) as exc:
            status[0] = str(exc)
        elapsed = time.monotonic()-started
        print(json.dumps(dict(phase14_2="rank0_result", kind=kind, status=status[0],
                              submitted=submitted, wait_seconds=elapsed)), flush=True)
        if kind == "collective_timeout":
            status[0] = dict(message=status[0], submitted=submitted, wait_seconds=elapsed)
    if kind == "collective_timeout":
        # A CPU control-plane exchange communicates failure without another CNCL call.
        dist.broadcast_object_list(status, src=0, group=group.cpu_group)
    return dict(rank=rank, status=status[0])


def exercise(llm, kind):
    request_id = f"phase14_2-{kind}"
    fixture = Path("/project/artifacts/build/phase12_5_4/native-dump-yarn-scale")
    ids, _ = prompt(fixture, 1)
    client = llm.llm_engine
    scheduler = client.engine_core.engine_core.scheduler
    client.add_request(request_id, {"prompt_token_ids": ids}, SamplingParams(temperature=0, max_tokens=1))
    if scheduler.get_num_unfinished_requests() != 1:
        raise AssertionError("Live Vendor scheduler has no pending fault request")
    started = time.monotonic()
    try:
        try:
            result = llm.collective_rpc(enter_collective, args=(kind,), timeout=12)
        except (RuntimeError, TimeoutError) as exc:
            if kind != "worker_loss":
                raise
            result = {"type": type(exc).__name__, "message": str(exc)}
        else:
            if kind != "collective_timeout":
                raise AssertionError("RPC survived losing its worker")
            if {row["rank"] for row in result} != {0, 1} or result[0]["status"] != result[1]["status"]:
                raise AssertionError("Collective timeout status did not agree")
            status = result[0]["status"]
            if not (status["submitted"] and 2.5 <= status["wait_seconds"] <= 10
                    and re.search(r"timed out|caught.*timeout|wait returned timeout",
                                  status["message"] or "", re.IGNORECASE)):
                raise AssertionError(f"CNCL Work did not report the requested timeout: {status}")
    finally:
        client.abort_request([request_id])
        if scheduler.get_num_unfinished_requests():
            raise AssertionError("Pending requests survived fault abort")
    print(json.dumps(dict(phase14_2="distributed_fault", kind=kind, result=result,
                          seconds=time.monotonic()-started, pending=0)), flush=True)
