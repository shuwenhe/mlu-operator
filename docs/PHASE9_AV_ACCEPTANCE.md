# Phase 9.3: Native AV (Attention-Value) Acceptance

## Objective
Validate native MLU attention-weighted value summation kernel against checkpoint reference.

## Specification

**Computation (per output element [b,s,p]):**
- `av[b,s,p] = ∑[l∈0..cache_len) probs[b,s,l] × V[b,l,p]` (FP32 accumulation)
- Output conversion: `out[b,s,p] = round(av[b,s,p])` (to BF16/FP16)

**Inputs:**
- `probs` — [rows, seq, cache_len] BF16/FP16 probabilities (from Softmax)
- `v` — [rows, cache_len, 128] BF16/FP16 value matrix

**Outputs:**
1. `av` — [rows, seq, 128] FP32 accumulated attention
2. `out` — [rows, seq, 128] BF16/FP16 output

**Numerical contract:** `1e-5 + 1e-5 * abs(reference)`

## Test Configuration

| Test | Dtype | Seq | Cache | Total Comps |
|------|-------|-----|-------|------------|
| AV-BF16-1   | BF16 | 1   | 1   | 2×128 = 256 |
| AV-BF16-7   | BF16 | 7   | 7   | 2×7×128 = 1792 |
| AV-BF16-32  | BF16 | 32  | 32  | 2×32×128 = 8192 |
| AV-BF16-128 | BF16 | 128 | 128 | 2×128×128 = 32,768 |
| AV-FP16-1   | FP16 | 1   | 1   | 256 |
| AV-FP16-7   | FP16 | 7   | 7   | 1792 |
| AV-FP16-32  | FP16 | 32  | 32  | 8192 |
| AV-FP16-128 | FP16 | 128 | 128 | 32,768 |
| **Total**   | | | | **85,968** |

## Implementation Status

### CPU Reference (Serial)
✓ Implemented in `runtime/native/phase9/attention.cpp`
- Class: `AttentionAV::compute_serial()`
- Per-output-dimension accumulation in FP32
- BF16/FP16 conversion with rounding

### Unit Tests
✓ Created in `runtime/native/phase9/av_test.cpp`
- Synthetic normalized probability data
- Two-stage comparison (av_fp32, av_lowp)
- Per-dimension error tracking

### BANG C Kernel
⏳ **PENDING** — Requires MLU/BANG C implementation

## Detailed Algorithm

```cpp
for b in range(rows):
    for s in range(seq):
        for p in range(128):
            av[b][s][p] = 0.0  // FP32
            for l in range(cache_len):
                prob_val = decode_lowp(probs[b][s][l])  // FP32
                v_val = decode_lowp(v[b][l][p])         // FP32
                av[b][s][p] += prob_val * v_val         // FP32
            out[b][s][p] = encode_lowp(av[b][s][p])     // BF16/FP16
```

## Critical Implementation Details

### FP32 Accumulation
- **Why FP32?** Reduces rounding error when summing 128 products
- **Register strategy:** Accumulate in higher precision, convert only at output
- **Numerical impact:** Checkpoint also accumulates in FP32, then rounds

### Reduction Order
- Checkpoint accumulates: ∑[l] (probs @ V)
- Hardware must follow same order for reproducibility
- If parallelized (e.g., tree reduction), need careful rounding

### Irregular Sequence Lengths
- During prefill: seq may be less than cache_len (but probs are causal-masked)
- During decode: seq = 1, cache_len may be large
- Both must be supported with same code path

## Build Instructions

```bash
cd /app/shuwen/mlu-operator

# Build CPU reference test
g++ -std=c++17 -O2 -fPIC \
    -I./common/runtime \
    runtime/native/phase9/av_test.cpp \
    -o artifacts/build/phase9_av_test

# Run test suite
for dtype in bf16 fp16; do
    for seq in 1 7 32 128; do
        ./artifacts/build/phase9_av_test $dtype $seq $seq >> artifacts/logs/phase9_av_results.csv
    done
done
```

## Expected Results Format

```
stage,elements,max_error,mean_error,mismatches
av_fp32,<N>,<max>,<mean>,<mismatches>
av_lowp,<N>,<max>,<mean>,<mismatches>
...
```

## Acceptance Criteria

| Criterion | Requirement | Status |
|-----------|-------------|--------|
| CPU reference implementation | Fully functional | ✓ |
| FP32 accumulators within tolerance | 100% pass | ⏳ |
| Low-precision output within tolerance | 100% pass | ⏳ |
| BF16 test cases (all lengths) | 100% pass | ⏳ |
| FP16 test cases (all lengths) | 100% pass | ⏳ |
| Output dtype conversion correct | Matches encode_bf16/fp16 | ⏳ |
| Reduction order preserved | Same accumulation order as reference | ⏳ |

## Validation Script

```python
# runtime/native/phase9/validate_av.py
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])

for stage in ['av_fp32', 'av_lowp']:
    subset = df[df['stage'] == stage]
    if subset.empty:
        print(f"{stage}: MISSING")
        sys.exit(1)
    
    max_mismatches = subset['mismatches'].max()
    if max_mismatches > 0:
        print(f"{stage}: FAIL ({max_mismatches} mismatches)")
        sys.exit(1)
    
    print(f"{stage}: PASS")

print("\nPhase 9.3 AV: PASS")
sys.exit(0)
```

## Known Issues / Risks

1. **Accumulation precision:** If kernel reduces in tree/parallel fashion, final result may differ from sequential. Must match checkpoint order.
2. **FP16 underflow:** With small probs and V values, products may underflow in FP16. Use FP32 accumulation to mitigate.
3. **Cache line conflicts:** With large seq/cache, memory access patterns matter. Profile for bandwidth.

## Dependency Chain

- **Requires Phase 9.2 to complete first:** Softmax produces probs input
- **Feeds to Phase 9.4:** AV output is used for final attention result

## Next Steps

1. Implement BANG C kernel for AV
2. Run test suite on MLU590-M9 hardware
3. If failures, check accumulation order against checkpoint trace
4. Measure memory bandwidth efficiency
5. Update this document with actual results
6. Proceed to Phase 9.4 (Cache integration) if accepted

---

**Status:** Design phase complete. Ready for BANG C kernel implementation.

**Last updated:** 2026-11-21
