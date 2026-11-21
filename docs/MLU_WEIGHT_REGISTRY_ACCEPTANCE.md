# MLU packed-weight registry acceptance

The registry key comprises revision, tensor identity, shard, byte offset, logical layout, dtype, canonical device PCI identity, K and N. The runtime fixture uses the checkpoint index SHA-256 as its local revision key, with actual shard names and payload offsets. It does not claim a full checkpoint content hash. Published weights are immutable views. Each lookup returns an aliasing reference-counted lease. Invalidation rejects subsequent lookups and retains old entries while leases exist.

Host packing is synchronous. Device packing submits to a worker-owned queue and is initially invisible to readers. Explicit publication synchronizes that queue before exposing the handle. The caller keeps the source live through publication and must also keep it live if submission fails after allocation. Pending invalidated entries synchronize their packing queue before collection. Request dispatch pins leases until its queue completes. The registry may reclaim only invalidated entries whose leases have ended. Multiple revisions can coexist. Duplicate exact keys are rejected.

**Hardware PASS, R2.** Host packing, asynchronous device packing, hidden pre-publication entry, publication, two-revision visibility, invalidation with a live lease, collection after lease release, and zero registry bytes were exercised. The dispatch hardware test additionally invalidated a weight after GEMM submission and confirmed that it was reclaimed only after request completion. Source buffers, queue and packed allocations were released in the test.

[R1/R2 transcript](../artifacts/logs/runtime_r12_hardware.log) · [Final suite](../artifacts/logs/runtime_final_hardware.log)
