# Phase 9 Implementation Roadmap

## Status Overview

**Current:** Phase 8.2 passed all components except native attention — CPU reference is currently used.

**Target:** Phase 9.5 completion with all nine test cases passing using native MLU attention kernels.

## Phase 9.1: QK Kernel Implementation ← **ACTIVE**

### Objectives
1. Implement native MLU QK computation matching checkpoint semantics
2. Validate against Phase 8.2 reference checkpoint
3. Establish baseline for subsequent phases

### Deliverables
- `runtime/native/phase9/attention.h` — Kernel declarations
- `runtime/native/phase9/attention.cpp` — CPU reference implementations
- `runtime/native/phase9/qk_test.cpp` — Unit test harness
- `docs/PHASE9_QK_ACCEPTANCE.md` — Acceptance report

### Key Specifications
- **Input:** Q `[rows, seq, 192]`, K `[rows, cache_len, 192]` (BF16/FP16)
- **Scale:** `192^(-1/2) * yarn_mscale(40, 0.707)^2`
- **Rounding:** Intermediate scale value rounded to BF16/FP16 per Phase 8.2
- **Mask:** Causal mask: `if (l > s + cache_base) → -INF`
- **Output:** 
  - `qk_out` — FP32 raw scores
  - `scaled_out` — Low-precision scaled scores
  - `masked_out` — FP32 scores with causal mask applied

### Acceptance Criteria
✓ All FP32 raw QK values within `1e-5 + 1e-5 * abs(ref)`  
✓ All scaled values within tolerance after low-precision rounding  
✓ Causal mask applied correctly (all -INF at masked positions)  
✓ Strides and row-major order preserved  
✓ Both BF16 and FP16 test cases pass  

### Test Matrix
| Dtype | Seq | Cache | Test Count |
|-------|-----|-------|-----------|
| BF16  | 1   | 1     | 1         |
| BF16  | 7   | 7     | 1         |
| BF16  | 32  | 32    | 1         |
| BF16  | 128 | 128   | 1         |
| FP16  | 1   | 1     | 1         |
| FP16  | 7   | 7     | 1         |
| FP16  | 32  | 32    | 1         |
| FP16  | 128 | 128   | 1         |
| **Total** | | | **8** |

### Validation Script
```bash
cd /app/shuwen/mlu-operator
for dtype in bf16 fp16; do
    for seq in 1 7 32 128; do
        g++ -std=c++17 -O2 \
            runtime/native/phase9/qk_test.cpp \
            -o /tmp/qk_test_${dtype}_${seq}
        /tmp/qk_test_${dtype}_${seq} $dtype $seq $seq >> phase9_qk_results.csv
    done
done
python3 runtime/native/phase9/validate_qk.py phase9_qk_results.csv
```

## Phase 9.2: Softmax Kernel Implementation

### Objectives
1. Implement row-wise softmax preserving FP32 precision
2. Validate numerically against checkpoint PyTorch

### Deliverables
- `runtime/native/phase9/attention.cpp` — Softmax CPU reference
- `runtime/native/phase9/softmax_test.cpp` — Unit test harness
- `docs/PHASE9_SOFTMAX_ACCEPTANCE.md` — Acceptance report

### Key Specifications
- **Input:** Masked QK `[rows, seq, cache_len]` (FP32 with -INF at masked positions)
- **Algorithm:**
  1. Find row max (FP32)
  2. Exponentiate: `exp(masked - max)`
  3. Sum for denominator (FP32)
  4. Normalize: `prob = exp(...) / sum`
  5. Convert to BF16/FP16

### Acceptance Criteria
✓ Row max values match within FP32 tolerance  
✓ Exponential values within tolerance  
✓ Denominators (sum of exps) within tolerance  
✓ Normalized probabilities within tolerance  
✓ Low-precision output within tolerance  
✓ No NaN/INF propagation errors  

## Phase 9.3: AV Kernel Implementation

### Objectives
1. Implement attention-weighted value summation
2. Verify FP32 accumulation and output conversion

### Deliverables
- `runtime/native/phase9/av_test.cpp` — Unit test harness
- `docs/PHASE9_AV_ACCEPTANCE.md` — Acceptance report

### Key Specifications
- **Input:** Probs `[rows, seq, cache_len]` (BF16/FP16), V `[rows, cache_len, 128]` (BF16/FP16)
- **Algorithm:** Per output dimension: `∑[l] probs[l] * V[l]` (FP32 accumulation)
- **Output:** Converted to BF16/FP16

### Acceptance Criteria
✓ FP32 accumulators within tolerance  
✓ Output values after low-precision conversion within tolerance  
✓ Reduction order preserved  

## Phase 9.4: MLA and KV Cache

### Objectives
1. Integrate native attention with KV cache lifecycle
2. Support prefill and token-at-a-time decode
3. Validate two independent concurrent requests

### Deliverables
- `runtime/native/phase9/attention_harness.cpp` — Full integration test
- `docs/PHASE9_KV_CACHE_ACCEPTANCE.md` — Cache lifecycle report

### Key Specifications
- **Cache allocation:** Per-request disjoint slots, explicit ownership
- **Prefill:** Append full batch KV to cache[cursor:cursor+batch]
- **Decode:** Append single token KV, increment cursor
- **Read:** Attention uses cache[0:cursor+batch]
- **Rollback:** Restore cache state on error
- **Synchronization:** Explicit queue sync between QK→Softmax→AV

### Test Cases
1. Request A prefill (2 tokens) + decode (1 token)
2. Request B prefill (2 tokens) + decode (1 token)
3. Interleaved submission before sync
4. Independent cache validation

## Phase 9.5: Full Layer 0 Native Execution

### Objectives
1. Remove CPU attention fallback
2. Run all nine original test cases
3. Validate full numerical acceptance

### Deliverables
- `runtime/native/r6_native_attention_test.cpp` — Updated r6_test.cpp
- `docs/PHASE9_LAYER0_HARDWARE_ACCEPTANCE.md` — Acceptance report

### Test Matrix
| Case | Dtype | Prefill | Decode | Heads | Reps |
|------|-------|---------|--------|-------|------|
| 1    | BF16  | 1       | 1      | 16    | 2    |
| 2    | BF16  | 7       | 1      | 16    | 2    |
| 3    | BF16  | 32      | 1      | 16    | 2    |
| 4    | BF16  | 128     | 1      | 16    | 2    |
| 5    | BF16  | 2       | 1      | 16    | 2    |
| 6    | FP16  | 1       | 1      | 16    | 2    |
| 7    | FP16  | 7       | 1      | 16    | 2    |
| 8    | FP16  | 32      | 1      | 16    | 2    |
| 9    | FP16  | 128     | 1      | 16    | 2    |

### Acceptance Criteria
✓ All checkpoints (input norm, output norm, attention, etc.) within tolerance  
✓ 1,421,312 final output values compared, 0 mismatches  
✓ Two interleaved requests execute without interference  
✓ Zero tracked device memory leaks after teardown  
✓ Both repetitions produce identical results  

### Validation Command
```bash
cd /app/shuwen/mlu-operator
./runtime/native/phase9/build_full_layer.sh
python3 runtime/native/phase9/validate_full_layer.py artifacts/logs/phase9_layer0_accuracy.csv
```

## Phase 9.6: Performance Validation (Optional)

### Measurements
- QK latency (ms)
- Softmax latency (ms)
- AV latency (ms)
- Full attention latency (ms)
- Prefill throughput (tokens/s)
- Decode TPOT (ms/token)
- Peak device memory (MiB)

### Baselines (from Phase 8.2 CPU attention)
- Attention latency: TBD (CPU reference)
- Target: Demonstrate speedup or justify memory bandwidth bound

## Rollback Strategy

**If Phase 9.5 numerical acceptance fails:**

1. Identify root cause (accumulation order, rounding boundary, algorithm)
2. Create detailed diagnostic (preserve failed intermediates)
3. Document in `PHASE9_ROOT_CAUSE.md`
4. Keep Phase 8.2 CPU reference as fallback
5. Proposal for Phase 10 (adjusted algorithm, alternate kernel design, or relaxed tolerance with justification)

## Progress Tracking

| Phase | Task | Status | Date | Notes |
|-------|------|--------|------|-------|
| 9.1 | QK kernel | PENDING | | Awaiting BANG C implementation |
| 9.2 | Softmax kernel | PENDING | | Awaiting BANG C implementation |
| 9.3 | AV kernel | PENDING | | Awaiting BANG C implementation |
| 9.4 | Cache integration | PENDING | | Depends on 9.1–9.3 |
| 9.5 | Full Layer 0 | PENDING | | Depends on 9.4 |
| 9.6 | Performance | PENDING | | After 9.5 numerical acceptance |

---

**Note:** This roadmap is frozen after Phase 9 design acceptance. Any specification changes require explicit amendment.
