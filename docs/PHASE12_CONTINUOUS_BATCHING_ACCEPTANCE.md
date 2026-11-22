# Phase 12 continuous batching

**NOT RUN.** The paged KV device probe uses independent logical prefixes and explicit page remapping. vLLM model registration and cache dispatch are required to exercise the vendor scheduler with native execution. The requested batch sizes 1, 2 and 4, mixed prefill and decode, cancellation, backpressure and scheduler-driven block reclamation remain untested. [Batching hardware status](../artifacts/logs/phase12_batching_hardware.log)
