"""Parent-owned socket cleanup for the pinned Vendor multiprocessing executor."""
from pathlib import Path
import threading

from vllm.v1.executor.multiproc_executor import MultiprocExecutor


class LifecycleExecutor(MultiprocExecutor):
    """Keep the Vendor executor and release its IPC sockets before cyclic GC."""

    def __init__(self, vllm_config, monitor_workers=True):
        self._socket_owner = threading.get_ident()
        self._shutdown_lock = threading.Lock()
        self._owned_sockets = {}
        self._monitor = None
        self._released = False
        self.worker_exit_codes = None
        self.worker_lifecycle = []
        self._worker_pre_shutdown = {}
        try:
            super().__init__(vllm_config, monitor_workers=monitor_workers)
            self._capture_sockets()
        except BaseException:
            self.shutdown()
            raise

    def start_worker_monitor(self, inline=False):
        if inline:
            return super().start_worker_monitor(inline=True)
        # Vendor reaches this hook after wait_for_ready, before queue handshakes.
        # Capture sockets on the owner before this official monitor can stop workers.
        self._capture_sockets()
        self._monitor = threading.Thread(
            target=super().start_worker_monitor, kwargs={"inline": True},
            name="MultiprocWorkerMonitor", daemon=True,
        )
        self._monitor.start()

    def _capture_sockets(self):
        queues = [getattr(self, "rpc_broadcast_mq", None)]
        queues.extend(getattr(self, "response_mqs", ()))
        for worker in getattr(self, "workers", ()):
            queues.append(worker.worker_response_mq)
        for queue in filter(None, queues):
            spin = queue._spin_condition
            sockets = (queue.local_socket, queue.remote_socket)
            if spin is not None:
                sockets += (spin.local_notify_socket, spin.read_cancel_socket,
                            spin.write_cancel_socket)
            for socket in filter(None, sockets):
                self._owned_sockets.setdefault(socket.context, set()).add(socket)

    def shutdown(self):
        with self._shutdown_lock:
            if self._released:
                return
            self._capture_sockets()
            workers = tuple(getattr(self, "workers", ()))
            self._worker_pre_shutdown = {
                self._safe_worker_pid(worker): self._worker_info(worker) for worker in workers
            }
            super().shutdown()
            # The liveness callback can request teardown while the owner is in
            # an RPC. Only the owner can close its communication sockets.
            if threading.get_ident() != self._socket_owner:
                return
        if self._monitor is not None:
            self._monitor.join(timeout=10)
            if self._monitor.is_alive():
                raise RuntimeError("Vendor worker monitor did not exit after worker shutdown")
        with self._shutdown_lock:
            for context, sockets in self._owned_sockets.items():
                for socket in sockets:
                    socket.close(linger=0)
                context.term()
            self._owned_sockets.clear()
            workers = tuple(getattr(self, "workers", ()))
            self.worker_lifecycle = []
            for worker in workers:
                current = self._worker_info(worker)
                before = self._worker_pre_shutdown.get(current["pid"])
                if before is not None:
                    current["pre_shutdown"] = before
                self.worker_lifecycle.append(current)
            self.worker_exit_codes = [worker.proc.exitcode for worker in workers]
            if None in self.worker_exit_codes:
                raise RuntimeError("Vendor left a live worker after shutdown")
            for worker in workers:
                worker_pid = self._safe_worker_pid(worker)
                close_error = None
                try:
                    worker.proc.close()
                except BaseException as exc:  # pragma: no cover - defensive evidence path
                    close_error = repr(exc)
                    raise
                finally:
                    for item in self.worker_lifecycle:
                        if item["pid"] == worker_pid:
                            item["close_error"] = close_error
                            item["close_succeeded"] = close_error is None
            finalizer = getattr(self, "_finalizer", None)
            if finalizer is not None:
                finalizer.detach()
            self._released = True

    @staticmethod
    def _worker_info(worker):
        proc = worker.proc
        pid = LifecycleExecutor._safe_proc_pid(proc)
        info = {
            "pid": pid,
            "name": getattr(proc, "name", ""),
            "exitcode": getattr(proc, "exitcode", None),
            "is_alive": bool(proc.is_alive()),
            "close_succeeded": None,
            "close_error": None,
            "rank": getattr(worker, "rank", None),
            "parent_pid": None,
            "start_time_ticks": None,
        }
        if pid is None:
            return info
        status = Path(f"/proc/{pid}/status")
        if status.exists():
            for line in status.read_text().splitlines():
                if line.startswith("PPid:"):
                    info["parent_pid"] = int(line.split(":", 1)[1].strip())
                    break
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            parts = stat.read_text().split()
            if len(parts) >= 22:
                info["start_time_ticks"] = int(parts[21])
        return info

    @staticmethod
    def _safe_proc_pid(proc):
        try:
            return getattr(proc, "pid", None)
        except ValueError:
            return None

    @staticmethod
    def _safe_worker_pid(worker):
        return LifecycleExecutor._safe_proc_pid(worker.proc)
