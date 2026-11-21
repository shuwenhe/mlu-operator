# Phase 9.1: Native QK Acceptance

## Objective
Validate native MLU Query-Key multiplication kernel against checkpoint reference.

## Specification

**Computation:**
- Per row [b, s]: QK[b,s,l] = ∑[p∈0..192) Q[b,s,p] × K[b,l,p] (FP32)
- Scale: 192^(-1/2) × yarn_mscale(40, 0.707)^2 ≈ 0.072169...
- Rounding: Scaled value rounded to BF16/FP16 per Phase 8.2
- Causal mask: if l > s + cache_base → set to -INF

**Outputs:**
1. `qk_out` — Raw FP32 scores (pre-scale)
2. `scaled_out` — Low-precision scaled scores
3. `masked_out` — FP32 scores with causal mask applied

**Numerical contract:** `1e-5 + 1e-5 * abs(reference)`

## Test Configuration

| Test | Dtype | Seq | Cache | Rows | Total Comparisons |
|------|-------|-----|-------|------|-------------------|
| QK-BF16-1   | BF16 | 1   | 1   | 1   | 3×1 = 3 |
| QK-BF16-7   | BF16 | 7   | 7   | 1   | 3×49 = 147 |
| QK-BF16-32  | BF16 | 32  | 32  | 1   | 3×1024 = 3072 |
| QK-BF16-128 | BF16 | 128 | 128 | 1   | 3×16384 = 49,152 |
| QK-FP16-1   | FP16 | 1   | 1   | 1   | 3×1 = 3 |
| QK-FP16-7   | FP16 | 7   | 7   | 1   | 3×49 = 147 |
| QK-FP16-32  | FP16 | 32  | 32  | 1   | 3×1024 = 3072 |
| QK-FP16-128 | FP16 | 128 | 128 | 1   | 3×16384 = 49,152 |
| **Total**   | | | | | **104,748** |

## Implementation Status

### CPU Reference (Serial)
✓ Implemented in `runtime/native/phase9/attention.cpp`
- Class: `AttentionQK::compute_serial()`
- Input validation and dimension checks
- BF16/FP16 encoding/decoding utilities

### Unit Tests
✓ Created in `runtime/native/phase9/qk_test.cpp`
- Synthetic deterministic input data
- Three-stage comparison (qk_fp32, scaled, masked)
- Tolerance checking with detailed error reporting

### BANG C Kernel
⏳ **PENDING** — Requires MLU/BANG C implementation
- Phase 9.1 deliverable
- Target: Match CPU reference within tolerance
- Hardware target: MLU590-M9

## Build Instructions

```bash
cd /app/shuwen/mlu-operator

# Build CPU reference test
g++ -std=c++17 -O2 -fPIC \
    -I./common/runtime \
    -I./operators/rmsnorm/native \
    runtime/native/phase9/qk_test.cpp \
    -o artifacts/build/phase9_qk_test

# Run test suite
for dtype in bf16 fp16; do
    for seq in 1 7 32 128; do
        ./artifacts/build/phase9_qk_test $dtype $seq $seq >> artifacts/logs/phase9_qk_results.csv
    done
done
```

## Expected Results Format

```
stage,elements,max_error,mean_error,mismatches
qk_fp32,<N>,<max>,<mean>,<mismatches>
qk_scaled,<N>,<max>,<mean>,<mismatches>
qk_masked,<N>,<max>,<mean>,<mismatches>
...
```

## Acceptance Criteria

| Criterion | Requirement | Status |
|-----------|-------------|--------|
| CPU reference implementation | Fully functional | ✓ |
| Unit test harness | Compilation and runs without crashes | ⏳ |
| All QK FP32 values within tolerance | 100% pass rate | ⏳ |
| All scaled values within tolerance | 100% pass rate | ⏳ |
| All masked values within tolerance | 100% pass rate | ⏳ |
| BF16 test cases (all lengths) | 100% pass rate | ⏳ |
| FP16 test cases (all lengths) | 100% pass rate | ⏳ |
| No silent NaN/INF propagation | Explicit -INF at masked positions | ⏳ |
| Strides and row-major order preserved | Same layout as reference | ⏳ |

## Validation Script

```python
# runtime/native/phase9/validate_qk.py
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])

for stage in ['qk_fp32', 'qk_scaled', 'qk_masked']:
    subset = df[df['stage'] == stage]
    if subset.empty:
        print(f"{stage}: MISSING")
        sys.exit(1)
    
    max_mismatches = subset['mismatches'].max()
    if max_mismatches > 0:
        print(f"{stage}: FAIL ({max_mismatches} mismatches)")
        sys.exit(1)
    
    print(f"{stage}: PASS")

print("\nPhase 9.1 QK: PASS")
sys.exit(0)
```

## Known Issues / Risks

1. **BF16 rounding:** Intermediate scaled value must be rounded to BF16 before converting to output dtype. Order of operations matters for numerical agreement.
2. **Causal mask:** Must use exact -INF (not -very-large), and downstream softmax must handle -INF correctly.
3. **Cache base offset:** Causal limit = cache_base + position[s] + 1; ensure boundary conditions correct at prefill vs. decode.

## Next Steps

1. Implement BANG C kernel for QK computation
2. Run test suite on MLU590-M9 hardware
3. If failures, run diagnostic with intermediate value tracing
4. Update this document with actual results
5. Proceed to Phase 9.2 (Softmax) if accepted

---

**Status:** Design phase complete. Ready for BANG C kernel implementation.

**Last updated:** 2026-11-21
