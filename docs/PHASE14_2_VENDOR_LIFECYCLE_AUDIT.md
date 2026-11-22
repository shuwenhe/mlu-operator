# Phase 14.2 — Vendor lifecycle audit

The frozen image contains Vendor `vllm_mlu 0.25.1+mlu0.16.0.pt212`, `torch_mlu 1.34.1+torch2.12.1`, and PyZMQ 27.1.0. The installed source snapshots and package manifest are in the Phase 14.2 evidence directory.

## Same-process EngineCore

`InprocClient.shutdown(timeout)` calls `EngineCore.shutdown()` on the caller's thread. It does not forward or enforce the timeout. EngineCore clears structured-output state, shuts down the executor, shuts down the scheduler, then runs Vendor distributed and allocator cleanup. That cleanup calls `gc.collect()` after destroying the process groups.

The executor owns its broadcast queue and references each rank's response queue. Each `MessageQueue` creates its own PyZMQ context. The queue retains its data sockets and `SpinCondition` retains its notify and reader cancellation sockets. Context socket tracking uses weak references. The Vendor queue's `shutdown()` sets a flag and wakes blocked readers. It has no explicit socket-close or context-terminate operation. The executor signals child processes over their death pipes, waits for them, calls queue shutdown and clears its queue references. Each worker handles the death event, shuts down its local queues and destroys its process groups.

`Context.destroy()` takes the sockets discoverable through its tracking set, closes them and calls the synchronous `term()`. Destruction must run after socket users have finished. Worker-exit and monitor-thread ordering matter when deciding where the owning process can safely close its sockets.

## Independent EngineCore

The multiprocess frontend client owns its own IPC endpoints and finalizer and waits on a separate EngineCore process through its process manager. It has a separate timeout and resource cleanup sequence. The executor and its shared-memory queue contexts still reside with the EngineCore. Frontend shutdown completion therefore does not establish orderly executor queue termination. The Phase 14.1 separate-process fault attempt illustrates this distinction.
