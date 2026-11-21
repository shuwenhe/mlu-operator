# Phase 9.4: KV Cache Integration and Acceptance

## Objective
Integrate native attention QK/Softmax/AV kernels with MLU KV cache lifecycle management.

## Scope

**Operations:**
1. Prefill: Append full KV batch to cache
2. Decode: Append single token KV, increment cursor
3. Attention: QK/Softmax/AV using cache[0:cursor+batch]
4. Rollback: Restore cache on error
5. Concurrency: Two independent requests with disjoint caches

## Test Configuration

**Two concurrent requests (A and B):**
- Each request: 2-token prefill + 1-token decode
- Interleaved submission before sync
- Separate cache slots, independent cursors

**Matrix:**
| Request | Phase | Tokens | Cache read | Cache write |
|---------|-------|--------|-----------|------------|
| A | Prefill | 2 | — | cache[0:2] |
| B | Prefill | 2 | — | cache[0:2] |
| A | Decode | 1 | cache[0:3] | cache[3:4] |
| B | Decode | 1 | cache[0:3] | cache[3:4] |

**Dtypes:** BF16 and FP16

## Implementation Status

### Cache Infrastructure
✓ Existing: `Workspace` class with per-slot cache allocation
✓ Existing: `Registry` for weight management
✓ Existing: Queue synchronization in `Worker`

### Attention Integration
⏳ **Phase 9.1–9.3:** Native kernels must complete first
⏳ Dispatch methods: `attention_qk`, `attention_softmax`, `attention_av`

### Cache Testing
- Written in Phase 9 full layer test (r6_native_attention_test.cpp)
- Validates cache append, read, and state preservation

## Key Specifications

### Cache Lifecycle (per request)

1. **Allocation:** `slot = workspace.add(queue_id, size_low, size_fp32)`
   - `size_low` for cached BF16/FP16 KV
   - `size_fp32` for FP32 workspace (attention internals)

2. **Prefill (batch=2):**
   ```
   workspace.start(slot)
   dispatch.attention_qk(..., q[0:2], k_cache[0:2])
   dispatch.attention_softmax(...)
   dispatch.attention_av(..., v_cache[0:2])
   workspace.finish(slot)
   cursor = 2
   ```

3. **Decode (batch=1):**
   ```
   workspace.start(slot)
   dispatch.attention_qk(..., q[2], k_cache[0:3])  // read all previous KV
   dispatch.attention_softmax(...)
   dispatch.attention_av(..., v_cache[0:3])
   workspace.finish(slot)
   cursor = 3
   ```

4. **Cleanup:**
   - `workspace.finish(slot)` before next request
   - Explicit rollback on error (not automatic)

### Two Concurrent Requests

```
Queue A: submit prefill_A → (no sync)
Queue B: submit prefill_B → (no sync)
         ↓
         sync both
Queue A: submit decode_A → (no sync)
Queue B: submit decode_B → (no sync)
         ↓
         sync both
         ↓
         compare results
```

## Validation Strategy

1. **Cache state:** Verify cache contents match reference (host-side check)
2. **Numerical results:** Compare AV output per request against checkpoint
3. **Concurrency:** Ensure Q-A and Q-B don't corrupt each other's cache
4. **Synchronization:** Explicit sync before reusing device buffers
5. **Cleanup:** No leaked device memory, all slots released

## Acceptance Criteria

| Criterion | Requirement | Status |
|-----------|-------------|--------|
| Native QK kernel | Phase 9.1 acceptance | ⏳ |
| Native Softmax kernel | Phase 9.2 acceptance | ⏳ |
| Native AV kernel | Phase 9.3 acceptance | ⏳ |
| Prefill cache append | Correct cache state | ⏳ |
| Decode cache read | Full history accessible | ⏳ |
| Two requests (BF16) | Numerical acceptance | ⏳ |
| Two requests (FP16) | Numerical acceptance | ⏳ |
| Cache cleanup | Zero leaks after teardown | ⏳ |
| Explicit sync required | No implicit backend retry | ⏳ |

## Known Issues / Risks

1. **Queue synchronization:** Need explicit `worker.sync(queue_id)` between QK/Softmax/AV stages, not just between requests
2. **Cache sizing:** Need to ensure workspace pre-allocates enough for full sequence (prefill + decode)
3. **Interleaved requests:** If submissions don't use separate queues, may serialize instead of running truly concurrent

## Dependency

- Requires Phase 9.1, 9.2, 9.3 to complete
- Feeds into Phase 9.5 (full Layer 0 test)

## Deliverables

- `runtime/native/phase9/attention_harness.cpp` — Full integration test
- `docs/PHASE9_KV_CACHE_ACCEPTANCE.md` — This document with results
- Traces: QK/Softmax/AV intermediate values per request

## Next Steps

1. Complete Phase 9.1–9.3 kernels
2. Implement integration test harness
3. Run on MLU590-M9 with both requests
4. Validate cache state before/after each operation
5. Update this document with results
6. Proceed to Phase 9.5 if passed

---

**Status:** Design phase complete. Awaiting kernel implementations.

**Last updated:** 2026-11-21
