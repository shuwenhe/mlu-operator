# Queue and workspace acceptance

[Workspace](../runtime/native/workspace.cpp) provisions separately allocated low-precision activation and FP32 scratch per slot. Each slot belongs to one worker queue, rejects re-entrance while active, and retains input/output allocation pins, packed-weight leases and temporary CNNL resources through completion. The dispatch layer accepts only whole owned allocation bases with sufficient capacity and contiguous row-major strides. Buffer reclamation fails while any active request holds a pin. CNNL device dispatch uses the caller's queue and synchronizes before returning from its operator. Native device dispatch normally enqueues. Native RoPE additionally drains its queue before a host validation read of the position indices.

**Hardware PASS, R3.** Two queues and distinct request slots were created. A fault injected into `cnrtQueueSync` left the request pending and preserved the raw error. Recovery reran real synchronization before release and reuse. The native GEMM dispatch test confirmed early output release is rejected. The connected decoder fixture submitted the two requests before synchronizing either. No actual driver corruption or queue hang was induced.

[R3/R4 transcript](../artifacts/logs/runtime_r34_hardware.log) · [Final suite](../artifacts/logs/runtime_final_hardware.log)
