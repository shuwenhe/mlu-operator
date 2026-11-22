# Phase 12 paged KV cache

The native adapter consumes scheduler-format int32 physical slot mappings on writes and physical block IDs on reads. Its page buffers are token-major `[physical slots, 16, width]`, with 192 key and 128 value words per head. Gather produces the head-major logical prefix used by the accepted native attention chain. The BANG C kernel copies low-precision bits without arithmetic, and the wrapper launches it on the current MLU queue. It synchronizes before returning so buffers remain live through the transfer.

The scheduler is responsible for unique live write slots, valid block ownership, committed lengths and the ordering of reclamation. Padding writes with negative slots are skipped. Out-of-range indices are guarded in the kernel. The current adapter supports logical lengths through 131, the existing standalone position-table limit.

**Device probe PASS.** BF16 and FP16 passed bitwise readback for noncontiguous physical blocks, a one-token append, two independent logical prefixes, ignored padding and explicit block reassignment on MLU590-M9 card 0.

**Scheduler integration BLOCKED.** The standalone hardware harness supplies metadata and simulates reassignment. The vendor model has yet to allocate this cache format and connect its actual block tables and slot mappings at each layer. Preemption, recomputation and framework-driven reclamation remain untested.

[Device evidence and reproduction commands](../artifacts/logs/phase12_kv_cache_hardware.log)
