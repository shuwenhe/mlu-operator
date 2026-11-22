# Phase 14.2 — Distributed failure recovery

The hardware recovery harness runs a fault followed by a frozen request in a newly constructed Vendor TP=2 Engine. It compares both workers’ logits and the exact greedy tokens using the original acceptance driver. The native MLA, Ordered O and KV cache code paths are unchanged.

The worker-loss and collective-timeout fixtures first add a pending request to the actual Vendor scheduler. Both workers enter a CPU control barrier and record their device-group backend. The worker-loss fixture exits rank 1 and checks the bounded error at the Vendor RPC client. The collective-timeout fixture withholds rank 1 from a real CNCL work operation and verifies an identical timeout status on both ranks over the existing CPU control group. Each fixture aborts its pending request and checks that the scheduler reports no unfinished work. The independent recovery cycle proves whether the next engine can run after the failed group was torn down.

The gate report distinguishes fixtures that actually reached their target path from launch or harness errors.
