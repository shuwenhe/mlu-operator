# Phase 9.2: Native Softmax Acceptance

## Objective
Validate native MLU row-wise softmax with FP32 precision against checkpoint reference.

## Specification

**Algorithm (per row [b,s]):**
1. Find row max: `max = max{masked[b,s,l] | l < cache_len}`
2. Exponentiate: `exp[l] = exp(masked[b,s,l] - max)`
3. Denominator: `denom = ∑[l] exp[l]` (FP32)
4. Normalize: `prob[l] = exp[l] / denom`
5. Convert to low-precision: `prob_low[l] = round(prob[l])`

**Inputs:**
- `masked` — [rows, seq, cache_len] FP32 with -INF at masked positions
- `seq_len` — Actual cache length (for causal masking boundary)

**Outputs:**
1. `maxes` — [rows, seq] FP32 row maximums
2. `denom` — [rows, seq] FP32 denominators (∑ exp)
3. `probs` — [rows, seq, cache_len] FP32 normalized probabilities
4. `probs_low` — [rows, seq, cache_len] BF16/FP16 converted probabilities

**Numerical contract:** `1e-5 + 1e-5 * abs(reference)`

**Special handling:**
- -INF at masked positions: exponentiate to 0.0, don't contribute to denom
- Zero denominator: set all probs to 0.0
- Output dtype: BF16 or FP16 per model requirement

## Test Configuration

| Test | Dtype | Seq | Cache | Total Elements |
|------|-------|-----|-------|----------------|
| SM-BF16-1   | BF16 | 1   | 1   | 2 (max/denom) + 3×1 (probs) = 5 |
| SM-BF16-7   | BF16 | 7   | 7   | 2×7 + 3×49 = 161 |
| SM-BF16-32  | BF16 | 32  | 32  | 2×32 + 3×1024 = 3136 |
| SM-BF16-128 | BF16 | 128 | 128 | 2×128 + 3×16384 = 49,408 |
| SM-FP16-1   | FP16 | 1   | 1   | 5 |
| SM-FP16-7   | FP16 | 7   | 7   | 161 |
| SM-FP16-32  | FP16 | 32  | 32  | 3136 |
| SM-FP16-128 | FP16 | 128 | 128 | 49,408 |
| **Total**   | | | | **205,420** |

## Implementation Status

### CPU Reference (Serial)
✓ Implemented in `runtime/native/phase9/attention.cpp`
- Class: `AttentionSoftmax::compute_serial()`
- Numerically stable max subtraction
- Proper handling of -INF positions

### Unit Tests
✓ Created in `runtime/native/phase9/softmax_test.cpp`
- Synthetic causal-masked input data
- Four-stage comparison (max, denom, probs_fp32, probs_lowp)
- Detailed error reporting per stage

### BANG C Kernel
⏳ **PENDING** — Requires MLU/BANG C implementation

## Critical Implementation Details

### Numerical Stability
- **Max subtraction:** Essential for numerical stability with large scores
- **FP32 precision:** Denominator accumulation must use FP32, not lower
- **Zero handling:** When denom ≈ 0, output zeros (all probs masked)

### -INF Handling
- Input may have -INF from causal mask
- `exp(-INF) = 0.0` exactly
- Must not propagate -INF to output probs (should be 0.0)

### Output Conversion
- After normalization (probs_fp32), convert each value to BF16/FP16
- Use checkpoint rounding (matches encode_bf16 / encode_fp16)

## Build Instructions

```bash
cd /app/shuwen/mlu-operator

# Build CPU reference test
g++ -std=c++17 -O2 -fPIC \
    -I./common/runtime \
    runtime/native/phase9/softmax_test.cpp \
    -o artifacts/build/phase9_softmax_test

# Run test suite
for dtype in bf16 fp16; do
    for seq in 1 7 32 128; do
        ./artifacts/build/phase9_softmax_test $dtype $seq $seq >> artifacts/logs/phase9_softmax_results.csv
    done
done
```

## Expected Results Format

```
stage,elements,max_error,mean_error,mismatches
softmax_max,<N>,<max>,<mean>,<mismatches>
softmax_denom,<N>,<max>,<mean>,<mismatches>
softmax_probs_fp32,<N>,<max>,<mean>,<mismatches>
softmax_probs_lowp,<N>,<max>,<mean>,<mismatches>
...
```

## Acceptance Criteria

| Criterion | Requirement | Status |
|-----------|-------------|--------|
| CPU reference implementation | Fully functional | ✓ |
| Row max values within tolerance | 100% pass | ⏳ |
| Denominators within tolerance | 100% pass | ⏳ |
| FP32 normalized probs within tolerance | 100% pass | ⏳ |
| Low-precision probs within tolerance | 100% pass | ⏳ |
| BF16 test cases (all lengths) | 100% pass | ⏳ |
| FP16 test cases (all lengths) | 100% pass | ⏳ |
| Proper -INF handling | No NaN propagation | ⏳ |
| Output sum-to-1 property | ∑[l] prob[l] ≈ 1.0 | ⏳ |

## Validation Script

```python
# runtime/native/phase9/validate_softmax.py
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])

for stage in ['softmax_max', 'softmax_denom', 'softmax_probs_fp32', 'softmax_probs_lowp']:
    subset = df[df['stage'] == stage]
    if subset.empty:
        print(f"{stage}: MISSING")
        sys.exit(1)
    
    max_mismatches = subset['mismatches'].max()
    if max_mismatches > 0:
        print(f"{stage}: FAIL ({max_mismatches} mismatches)")
        sys.exit(1)
    
    print(f"{stage}: PASS")

print("\nPhase 9.2 Softmax: PASS")
sys.exit(0)
```

## Known Issues / Risks

1. **Max value with all -INF rows:** When entire row is -INF (shouldn't happen with valid causal mask), max is -INF. Result probs should all be 0.0.
2. **Small denominator:** Rounding errors accumulate. With normalized probabilities, some epsilon error is expected.
3. **FP16 precision:** With smaller exponent range, may underflow or overflow compared to BF16. Test boundary cases.

## Dependency

- **Requires Phase 9.1 to complete first:** QK produces masked input for softmax
- **Feeds to Phase 9.3:** Softmax output (probs) is input to AV

## Next Steps

1. Implement BANG C kernel for softmax
2. Run test suite on MLU590-M9 hardware
3. If failures, preserve failed intermediates for debugging
4. Verify output probabilities sum to ~1.0 per row
5. Update this document with actual results
6. Proceed to Phase 9.3 (AV) if accepted

---

**Status:** Design phase complete. Ready for BANG C kernel implementation.

**Last updated:** 2026-11-21
