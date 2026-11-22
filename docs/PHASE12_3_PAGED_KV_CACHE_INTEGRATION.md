# Scheduler-facing paged KV adapter

The adapter reuses the Phase 12 BANG C scatter and gather library. For each scheduled request it writes the current projected K/V into the requested physical slots and gathers the committed prefix in the layout consumed by native attention. Cache allocation and reclamation belong to the scheduler. The current native component test allocates buffers and supplies block IDs itself. Engine-owned native cache allocation and reclamation are **BLOCKED** on the model and cache specification binding.

Each transfer uses the active CNRT queue and synchronizes before its buffers can be reused. Cross-request ownership is checked across the active batch. Ownership across scheduling steps depends on the engine integration. Prefix sharing and preemption are outside the current adapter contract. [Hardware component log](../artifacts/logs/phase12_3_native_mla_hardware.log)

The engine cache specification must account for 10,240 bytes per token per layer in BF16 or FP16, before allocator alignment. This is `16 * (192 + 128) * 2` bytes. A block of size `B` therefore needs `B * 10,240` bytes per layer, with explicit key and value views. Worker positions must be narrowed to int32 after checking the supported range before passing them to the native launcher.
