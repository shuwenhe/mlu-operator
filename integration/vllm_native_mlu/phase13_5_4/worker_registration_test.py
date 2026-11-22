"""Validate architecture registration in main and spawned worker processes."""
from __future__ import annotations

import json
import multiprocessing as mp

from vllm import ModelRegistry

from ..phase12_5.run import ARCHITECTURE, register


def _inspect_once() -> dict[str, object]:
    register()
    entry = ModelRegistry.models.get(ARCHITECTURE)
    if entry is None:
        return {
            "registered": False,
            "error": "architecture_not_registered",
        }
    model_cls = entry.load_model_cls()
    return {
        "registered": True,
        "class_module": model_cls.__module__,
        "class_name": model_cls.__name__,
        "supported_arch_count": len(ModelRegistry.get_supported_archs()),
        "builtin_deepseek_present": "DeepseekV2ForCausalLM" in ModelRegistry.get_supported_archs(),
    }


def _worker(queue: mp.Queue) -> None:
    try:
        queue.put({"ok": True, "data": _inspect_once()})
    except Exception as exc:  # explicit diagnostic path for registration gate
        queue.put({"ok": False, "error": str(exc)})


def main() -> int:
    report: dict[str, object] = {
        "architecture": ARCHITECTURE,
        "main_process": _inspect_once(),
        "worker_processes": [],
    }

    # Repeat registration should not fail.
    register()
    register()

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(queue,)) for _ in range(2)]
    for proc in workers:
        proc.start()
    for proc in workers:
        proc.join(timeout=120)

    worker_results = []
    for _ in workers:
        worker_results.append(queue.get(timeout=30))
    report["worker_processes"] = worker_results

    worker_all_ok = all(item.get("ok") and item.get("data", {}).get("registered") for item in worker_results)
    main_ok = bool(report["main_process"].get("registered"))  # type: ignore[index]
    report["status"] = "PASS" if main_ok and worker_all_ok else "FAIL"

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
