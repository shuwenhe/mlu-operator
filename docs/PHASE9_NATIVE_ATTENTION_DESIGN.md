# Phase 9: Native MLU Attention and KV Cache

## Objective
Replace the Phase 8.2 CPU checkpoint attention reference with native MLU execution while preserving all other accepted components.

## Frozen Components (Do Not Modify)
- AVX2-compatible GEMM (native matrix kernel)
- Phase 8.2 accepted RMSNorm (cascade sum, exact sqrt/rsqrt)
- Native YaRN RoPE
- Worker and Weight Registry
- Existing Layer 0 fixtures and test infrastructure

## Numerical Contract
- Tolerance: `1e-5 + 1e-5 * abs(reference)`
- Reference: checkpoint PyTorch with FP32 intermediates, BF16/FP16 model boundaries
- No silent fallback; native kernel must execute or fail explicitly
- Bitwise parity required where Phase 8.2 specified it

## Architecture Overview

### Attention Computation Pipeline

**Input Pipeline:**
1. Q: `[B, 16, S, 192]` (from Q projection, BF16/FP16)
2. K/V in cache: `[B, 16, L, 192]` / `[B, 16, L, 128]` respectively
3. Mask: `[B, 1, S, L]` (additive causal mask, computed or passed)

**QK Phase:**
- Scale: `192^(-1/2) * yarn_mscale(40, 0.707)^2`
- Compute: `Q @ K^T` → scores `[B, 16, S, L]`
- Apply scale (with BF16 rounding per Phase 8.2)
- Apply causal mask (-INF for future positions)
- Result: scaled masked scores

**Softmax Phase:**
- Find row max over L (per [B, head, S])
- Exponentiate with max subtraction
- Accumulate denominator (FP32)
- Normalize
- Convert probabilities to activation dtype (BF16/FP16)

**AV Phase:**
- Compute: `prob @ V` → attention `[B, 16, S, 128]`
- Accumulate in FP32
- Convert to activation dtype on output

**Cache Management:**
- Prefill: Materialize full KV in cache
- Decode: Append single token KV, read from cache
- Rollback: Support cache state restoration
- Two concurrent requests: disjoint cache slots

### Hardware Mapping

**MLU Cluster Considerations:**
- MLU590-M9 has 16×4 clusters with local memory
- Native BANG C kernels vs. CNNL library calls
- Need synchronization between QK→Softmax→AV stages

**Data Layout:**
- Q: Row-major `[B×16, S, 192]` or packed
- K/V: Row-major `[B×16, L, 192]` / `[B×16, L, 128]`
- Scores: `[B×16, S, L]` FP32 intermediate, BF16/FP16 output
- Probs: Row-major, FP32 → BF16/FP16

## Implementation Phases

### Phase 9.1: Native QK
**Deliverable:** Native MLU QK matching checkpoint accumulation

**Operations:**
1. Materialize Q and K tensors (or use row pointers)
2. Per-row QK computation: `∑[p∈0..192) Q[p] * K[p]`
3. Scale by `scale_factor` (with BF16 intermediate rounding)
4. Store raw QK, scaled QK, and masked QK

**Validation:**
- Compare FP32 intermediates (pre-scale QK values)
- Compare BF16-rounded scaled values
- Compare after causal mask application
- Verify tensor strides and contiguity

**Acceptance Criteria:**
- All QK/scaled/masked values within tolerance
- Same row-major order and stride as checkpoint reference
- Handle both regular and boundary cases (S=1 decode)

### Phase 9.2: Native Softmax
**Deliverable:** Native MLU softmax preserving FP32 precision

**Operations:**
1. Find row maximum (FP32)
2. Exponentiate (exp(masked_qk - max), FP32)
3. Accumulate denominator (sum of exps, FP32)
4. Normalize: `prob = exp(masked_qk - max) / denom`
5. Convert to BF16/FP16 for output

**Validation:**
- Compare FP32 max values per row
- Compare FP32 exponential intermediates
- Compare FP32 denominator
- Compare final normalized probabilities

**Acceptance Criteria:**
- Every intermediate within tolerance
- Output dtype matches model (BF16/FP16)
- Denominator NaN/INF handling matches reference

### Phase 9.3: Native AV
**Deliverable:** Native MLU attention-weighted sum

**Operations:**
1. Load BF16/FP16 probabilities
2. Per row: `∑[j∈0..L) prob[j] * V[j, :]` (FP32 accumulation)
3. Convert FP32 result to BF16/FP16

**Validation:**
- Compare FP32 accumulation per output dimension
- Compare BF16/FP16 output values
- Handle irregular sequence lengths (S=1, S<L)

**Acceptance Criteria:**
- All output values within tolerance
- Reduction order consistent with reference
- Same strides as checkpoint AV output

### Phase 9.4: MLA and KV Cache
**Deliverable:** Integrated KV cache with prefill/decode support

**Operations:**
1. **Prefill (batch):** Append full KV batch to cache
2. **Decode (token-at-a-time):** Append single KV token
3. **Attention read:** Access cache[0:cursor+batch] for QK/AV
4. **Rollback:** Restore previous cache state on error
5. **Cache lifecycle:** Per-request disjoint cache slots

**Validation:**
- Two independent requests with separate caches
- Interleaved prefill/decode ordering
- Cache append/read consistency
- Rollback preserves state

**Acceptance Criteria:**
- Final attention output matches checkpoint
- Both FP16 and BF16 test cases pass
- Zero cache leaks after teardown

### Phase 9.5: Full Layer 0 Native Execution
**Deliverable:** Remove CPU attention, run all nine hardware cases

**Components:**
1. Input RMSNorm (native, Phase 8.2)
2. Q/KV projections (native GEMM)
3. KV normalization and projection (native GEMM)
4. RoPE (native, Phase 8.2)
5. **Native Attention** (new, all phases above)
6. Output projection (native GEMM)
7. Post-attention RMSNorm (native, Phase 8.2)
8. FFN projections (native GEMM)
9. Post-layer RMSNorm (native, Phase 8.2)

**Validation:**
- Nine original test cases (various prefill lengths, FP16/BF16)
- Two interleaved requests
- Two repetitions per case
- Numerical acceptance report

**Acceptance Criteria:**
- All 1,421,312 final output values within tolerance
- Zero mismatches at checkpoints (input/output norms, projections, etc.)
- Explicit pass/fail, no silent fallback

## Performance Validation

**After numerical acceptance:**

**Measurements:**
1. QK latency (ms)
2. Softmax latency (ms)
3. AV latency (ms)
4. Full attention latency (ms)
5. Prefill throughput (tokens/s)
6. Decode TPOT (ms/token)
7. Peak device memory (MiB)

**Comparison baseline:** Phase 8.2 CPU attention execution time

**Acceptance threshold:** Demonstrate speedup over CPU path or justify (e.g., memory bandwidth bound)

## Deliverables

1. **docs/PHASE9_NATIVE_ATTENTION_DESIGN.md** (this document)
2. **docs/PHASE9_QK_ACCEPTANCE.md** — QK validation results
3. **docs/PHASE9_SOFTMAX_ACCEPTANCE.md** — Softmax validation results
4. **docs/PHASE9_AV_ACCEPTANCE.md** — AV validation results
5. **docs/PHASE9_KV_CACHE_ACCEPTANCE.md** — Cache lifecycle validation
6. **docs/PHASE9_LAYER0_HARDWARE_ACCEPTANCE.md** — Full layer numerical acceptance
7. **runtime/native/phase9/attention_qk.mlu** — QK kernel
8. **runtime/native/phase9/attention_softmax.mlu** — Softmax kernel
9. **runtime/native/phase9/attention_av.mlu** — AV kernel
10. **runtime/native/phase9/attention_harness.cpp** — C++ dispatch and test harness
11. **Updated dispatch.h/dispatch.cpp** — Add attention method to Dispatch class

## Testing Strategy

**Unit tests (phase 9.1–9.3):**
- Checkpoint fixture (1 head, variable S/L)
- Both dtypes (BF16, FP16)
- Edge cases (S=1, L=1, S=L)
- Prefill lengths: 1, 7, 32, 128

**Integration tests (phase 9.4–9.5):**
- Full Layer 0 (all 16 heads, real weights)
- Two interleaved requests
- Prefill + decode token-at-a-time
- Both dtypes
- Two repetitions per case
- Numerical and cleanup validation

**Acceptance criteria:** All comparisons within `1e-5 + 1e-5 * abs(reference)`

## Risk Mitigation

1. **Numerical instability:** Accumulate in FP32, avoid intermediate rounding except at model boundaries
2. **Device resource exhaustion:** Explicit cache ownership, synchronization before reuse, rollback on failure
3. **Performance regression:** Profile each kernel individually before integration
4. **Scheduler ambiguity:** Use explicit queue synchronization (not implicit)

## Rollback Plan

If native attention fails numerical acceptance:
1. Keep Phase 8.2 checkpoint CPU attention path as fallback
2. Document root cause (precision, algorithm, hardware limitation)
3. Proposal for Phase 10: adjusted accumulation or new kernel design

---

**Status:** Design phase complete. Ready for Phase 9.1 (QK kernel) implementation.
