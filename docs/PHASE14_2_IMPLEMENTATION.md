# Phase 14.2 — Lifecycle correction

The new `LifecycleExecutor` is selected through the existing Vendor `distributed_executor_backend` argument. It uses Vendor worker creation, request execution, failure monitoring and shutdown. It captures strong references to the existing parent-owned queue sockets before the Vendor monitor can release the queue references.

The lifecycle executor calls Vendor shutdown first. Once Vendor has signalled and waited for workers, the parent owner thread closes its sockets with zero linger and terminates their contexts. The monitor thread may request Vendor worker teardown and wake pending readers. It defers socket closure to the owner thread. A lock serializes shutdown across the two callers. A subsequent owner shutdown has no sockets left to close.

The Phase 14.1 launcher and model sources are preserved. The Phase 14.2 replay supplies the lifecycle executor to the same frozen inference driver. The launcher uses the same image and original configuration.

Select the fix with `LLM(..., distributed_executor_backend=LifecycleExecutor)`. The original default executor remains available for exact reproduction. Keep Engine creation, request stepping, and explicit client shutdown on its owner thread. The Vendor worker monitor continues to start after worker readiness and before the queue handshakes.

The first hardware retry closed the ZeroMQ sockets and exited, but repeated engines left four descriptors per round. The saved parent descriptor list shows two pipe pairs per cycle. The executor now waits for its Vendor liveness monitor to finish, records both worker exit codes, closes the exited Python process handles, and detaches the bound Vendor exit finalizer. The failure probe's forced container cleanup was used to end that failed experiment and is outside accepted shutdown evidence.
