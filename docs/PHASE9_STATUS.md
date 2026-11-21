# Phase 9 Native MLU Attention — Implementation Status

## Overview

**Phase 9** replaces the Phase 8.2 CPU checkpoint attention reference with native MLU execution. The design and unit test infrastructure is complete; BANG C kernel implementation is pending.

## Completion Status

| Component | Status | Location |
|-----------|--------|----------|
| **Design documentation** | ✓ COMPLETE | docs/PHASE9_NATIVE_ATTENTION_DESIGN.md |
| **Implementation roadmap** | ✓ COMPLETE | docs/PHASE9_IMPLEMENTATION_ROADMAP.md |
| **QK spec & tests** | ✓ COMPLETE | docs/PHASE9_QK_ACCEPTANCE.md, runtime/native/phase9/qk_test.cpp |
| **Softmax spec & tests** | ✓ COMPLETE | docs/PHASE9_SOFTMAX_ACCEPTANCE.md, runtime/native/phase9/softmax_test.cpp |
| **AV spec & tests** | ✓ COMPLETE | docs/PHASE9_AV_ACCEPTANCE.md, runtime/native/phase9/av_test.cpp |
| **Cache integration spec** | ✓ COMPLETE | docs/PHASE9_KV_CACHE_ACCEPTANCE.md |
| **Layer 0 acceptance spec** | ✓ COMPLETE | docs/PHASE9_LAYER0_HARDWARE_ACCEPTANCE.md |
| **C++ kernel headers** | ✓ COMPLETE | runtime/native/phase9/attention.h |
| **CPU reference implementations** | ✓ COMPLETE | runtime/native/phase9/attention.cpp |
| **Dispatch integration** | ✓ COMPLETE | runtime/native/dispatch.h, dispatch.cpp |
| **BANG C QK kernel** | ⏳ PENDING | runtime/native/phase9/attention_qk.mlu |
| **BANG C Softmax kernel** | ⏳ PENDING | runtime/native/phase9/attention_softmax.mlu |
| **BANG C AV kernel** | ⏳ PENDING | runtime/native/phase9/attention_av.mlu |
| **Full layer test harness** | ⏳ PENDING | runtime/native/r6_native_attention_test.cpp |

## Architectural Summary

### Three Native MLU Kernels

**1. Query-Key Multiplication (QK)**
```
Input:  Q [B×16, S, 192] BF16/FP16
        K [B×16, L, 192] BF16/FP16
Output: scores [B×16, S, L] FP32 raw + BF16/FP16 scaled + FP32 masked
```
- Accumulate Q·K^T in FP32
- Scale with Phase 8.2-style rounding
- Apply causal mask (-INF at future positions)

**2. Row-wise Softmax**
```
Input:  masked [B×16, S, L] FP32
Output: probs [B×16, S, L] FP32 normalized + BF16/FP16 converted
        max [B×16, S] FP32
        denom [B×16, S] FP32
```
- Find row max, subtract for stability
- Exponentiate, accumulate denominator
- Normalize and convert to low precision

**3. Attention-Value Summation (AV)**
```
Input:  probs [B×16, S, L] BF16/FP16
        V [B×16, L, 128] BF16/FP16
Output: attention [B×16, S, 128] FP32 + BF16/FP16 converted
```
- Accumulate prob·V in FP32
- Convert to low precision at output

### Numerical Properties

| Property | Value |
|----------|-------|
| Q/K dim | 192 |
| V dim | 128 |
| Scale factor | 192^(-1/2) × yarn_mscale(40,0.707)^2 ≈ 0.072169 |
| Accumulation precision | FP32 for all three kernels |
| Output precision | BF16 or FP16 (model-dependent) |
| Tolerance | 1e-5 + 1e-5 × abs(reference) |
| Causal mask | -INF at l > s+cache_base |

## Frozen Components (Do Not Modify)

```
✓ Input RMSNorm (Phase 8.2 accepted, cascade sum + exact sqrt/rsqrt)
✓ GEMM kernels (Phase 8.1 accepted, AVX2 compatible)
✓ YaRN RoPE (Phase 8.2 accepted)
✓ Post-attention/layer RMSNorm (Phase 8.2 accepted)
✓ Worker, Memory, Workspace (registry & queue management)
✓ All existing test fixtures and reference implementations
```

## Unit Test Infrastructure

### Phase 9.1: QK Tests
- File: `runtime/native/phase9/qk_test.cpp`
- Coverage: 8 test cases (BF16×4 + FP16×4)
- Matrix: seq={1,7,32,128}, cache_len=seq
- Comparison stages: qk_fp32, scaled, masked
- Expected total elements: 104,748

### Phase 9.2: Softmax Tests
- File: `runtime/native/phase9/softmax_test.cpp`
- Coverage: 8 test cases (BF16×4 + FP16×4)
- Matrix: seq={1,7,32,128}, cache_len=seq
- Comparison stages: max, denom, probs_fp32, probs_lowp
- Expected total elements: 205,420

### Phase 9.3: AV Tests
- File: `runtime/native/phase9/av_test.cpp`
- Coverage: 8 test cases (BF16×4 + FP16×4)
- Matrix: seq={1,7,32,128}, cache_len=seq, value_dim=128
- Comparison stages: av_fp32, av_lowp
- Expected total elements: 85,968

**Total unit test comparisons: 396,136 elements across all three kernels**

## Integration Test Infrastructure

### Phase 9.4: Cache Integration
- Validates prefill/decode cache lifecycle
- Tests two concurrent requests
- Verifies cache state consistency

### Phase 9.5: Full Layer 0
- Nine test cases from Phase 8.2
- All 16 heads, real model weights
- Two interleaved requests, two repetitions
- **1,421,312 final output values** compared
- Expected mismatches: ~400 (from Phase 8 FFN regressions)

## Build & Test Procedure

### Step 1: Compile Unit Tests
```bash
cd /app/shuwen/mlu-operator

# QK test
g++ -std=c++17 -O2 -fPIC \
    runtime/native/phase9/qk_test.cpp \
    -o artifacts/build/phase9_qk_test

# Softmax test
g++ -std=c++17 -O2 -fPIC \
    runtime/native/phase9/softmax_test.cpp \
    -o artifacts/build/phase9_softmax_test

# AV test
g++ -std=c++17 -O2 -fPIC \
    runtime/native/phase9/av_test.cpp \
    -o artifacts/build/phase9_av_test
```

### Step 2: Run Unit Tests (CPU Reference)
```bash
# QK tests
for dtype in bf16 fp16; do
    for seq in 1 7 32 128; do
        ./artifacts/build/phase9_qk_test $dtype $seq $seq >> artifacts/logs/phase9_qk_results.csv
    done
done

# Softmax tests
for dtype in bf16 fp16; do
    for seq in 1 7 32 128; do
        ./artifacts/build/phase9_softmax_test $dtype $seq $seq >> artifacts/logs/phase9_softmax_results.csv
    done
done

# AV tests
for dtype in bf16 fp16; do
    for seq in 1 7 32 128; do
        ./artifacts/build/phase9_av_test $dtype $seq $seq >> artifacts/logs/phase9_av_results.csv
    done
done
```

### Step 3: Validate CPU References
```bash
python3 runtime/native/phase9/validate_qk.py artifacts/logs/phase9_qk_results.csv
python3 runtime/native/phase9/validate_softmax.py artifacts/logs/phase9_softmax_results.csv
python3 runtime/native/phase9/validate_av.py artifacts/logs/phase9_av_results.csv
```

### Step 4: Implement BANG C Kernels
- Create `runtime/native/phase9/attention_qk.mlu`
- Create `runtime/native/phase9/attention_softmax.mlu`
- Create `runtime/native/phase9/attention_av.mlu`
- Link to dispatch methods

### Step 5: Build Full Layer Test
```bash
./runtime/native/phase9/build_full_layer.sh
```

### Step 6: Run Hardware Tests
```bash
./artifacts/build/r6_native_attention_test \
    --authorized-reference-mla /path/to/fixture/dir
```

### Step 7: Validate Hardware Results
```bash
python3 runtime/native/phase9/validate_layer0.py \
    artifacts/logs/phase9_layer0_accuracy.csv
```

## Documentation Deliverables

| File | Purpose |
|------|---------|
| docs/PHASE9_NATIVE_ATTENTION_DESIGN.md | Architecture, frozen components, pipeline overview |
| docs/PHASE9_IMPLEMENTATION_ROADMAP.md | Detailed task breakdown, test matrix, validation scripts |
| docs/PHASE9_QK_ACCEPTANCE.md | QK specification, unit tests, acceptance criteria |
| docs/PHASE9_SOFTMAX_ACCEPTANCE.md | Softmax specification, unit tests, numerical details |
| docs/PHASE9_AV_ACCEPTANCE.md | AV specification, unit tests, reduction order specs |
| docs/PHASE9_KV_CACHE_ACCEPTANCE.md | Cache lifecycle, two-request concurrency, cleanup validation |
| docs/PHASE9_LAYER0_HARDWARE_ACCEPTANCE.md | Full layer numerical acceptance, nine test cases, results |

## Key Implementation Decisions

### Accumulation Strategy
- **FP32 intermediates** for all operations (QK, Softmax, AV)
- **Low-precision inputs/outputs** at model boundaries
- **BF16 rounding** in scale step follows Phase 8.2 pattern

### Cache Management
- **Per-request allocation** via Workspace
- **Explicit ownership** (no automatic backend retry)
- **Synchronization** between QK/Softmax/AV stages
- **Rollback support** for error recovery

### Concurrency Model
- **Two independent queues** (one per request)
- **Disjoint cache slots** (no sharing)
- **Explicit sync** before comparing results
- **Post-teardown validation** (zero leaked bytes)

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Numerical instability | FP32 accumulation, avoid rounding except at boundaries |
| Device resource exhaustion | Explicit cache ownership, synchronization, cleanup validation |
| Precision loss in BF16 | Phase 8.2-style rounding in scale step |
| Interleaved request issues | Disjoint queues, separate caches, explicit sync |
| Silent fallback | Dispatch methods require explicit implementation; no implicit retry |

## Success Criteria

**Phase 9 is complete when:**
1. ✓ All CPU unit tests pass (396,136 elements within tolerance)
2. ✓ BANG C kernels implemented and linked to dispatch
3. ✓ Hardware full layer test passes (1,421,312 final values within tolerance)
4. ✓ All attention stages (QK/Softmax/AV) pass individually
5. ✓ Two concurrent requests run without interference
6. ✓ Zero resource leaks after teardown
7. ✓ Documentation complete with actual results
8. (Optional) Performance metrics show improvement over CPU attention

## Rollback / Failure Plan

**If numerical acceptance fails:**
1. Preserve failed case data and intermediates
2. Create detailed diagnostic report (PHASE9_ROOT_CAUSE.md)
3. Identify fixable issues (e.g., accumulation order) vs. fundamental limitations (e.g., hardware precision)
4. Keep Phase 8.2 CPU reference as fallback
5. Proposal for Phase 10 (alternative algorithm, adjusted tolerance, etc.)

## Timeline Estimates

| Phase | Task | Est. Duration | Dependency |
|-------|------|---------------|-----------|
| 9.1 | QK kernel | 2–3 days | None |
| 9.2 | Softmax kernel | 2–3 days | 9.1 passed |
| 9.3 | AV kernel | 2–3 days | 9.2 passed |
| 9.4 | Cache integration | 1–2 days | 9.1–9.3 passed |
| 9.5 | Full layer test | 1–2 days | 9.4 passed |
| 9.6 | Performance validation | 1 day | 9.5 passed |

**Total:** ~2–3 weeks from kernel implementation to Phase 9 completion

---

## Quick Links

- **Main design:** [PHASE9_NATIVE_ATTENTION_DESIGN.md](docs/PHASE9_NATIVE_ATTENTION_DESIGN.md)
- **Roadmap:** [PHASE9_IMPLEMENTATION_ROADMAP.md](docs/PHASE9_IMPLEMENTATION_ROADMAP.md)
- **QK tests:** [runtime/native/phase9/qk_test.cpp](runtime/native/phase9/qk_test.cpp)
- **Softmax tests:** [runtime/native/phase9/softmax_test.cpp](runtime/native/phase9/softmax_test.cpp)
- **AV tests:** [runtime/native/phase9/av_test.cpp](runtime/native/phase9/av_test.cpp)
- **C++ headers:** [runtime/native/phase9/attention.h](runtime/native/phase9/attention.h)
- **Dispatch integration:** [runtime/native/dispatch.h](runtime/native/dispatch.h)

---

**Phase 9 implementation kickoff ready.**

**Next action:** Implement BANG C kernels (QK, Softmax, AV).

**Last updated:** 2026-11-21
