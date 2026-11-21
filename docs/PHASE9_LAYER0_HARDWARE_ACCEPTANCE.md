# Phase 9.5: Full Layer 0 Native Attention Acceptance

## Objective
Complete replacement of CPU reference attention with native MLU kernels across all nine original test cases.

## Specification

**Layer 0 Pipeline:**
1. Input RMSNorm (native, Phase 8.2)
2. Q projection (native GEMM)
3. KV-A projection + normalization + KV-B projection (native GEMM chain)
4. RoPE (native, Phase 8.2)
5. **Attention (native QK + Softmax + AV)** ← NEW
6. Output projection (native GEMM)
7. Post-attention RMSNorm (native, Phase 8.2)
8. FFN projections (native GEMM)
9. Post-layer RMSNorm (native, Phase 8.2)

**Key change:** Step 5 now uses native MLU attention instead of CPU checkpoint reference.

## Test Matrix

| Case | Dtype | Prefill | Decode | Cache | Heads | Reps | Total Elements |
|------|-------|---------|--------|-------|-------|------|----------------|
| 1    | BF16  | 1       | 1      | 2     | 16    | 2    | 2×24,576 |
| 2    | BF16  | 7       | 1      | 8     | 16    | 2    | 2×24,576 |
| 3    | BF16  | 32      | 1      | 33    | 16    | 2    | 2×24,576 |
| 4    | BF16  | 128     | 1      | 129   | 16    | 2    | 2×24,576 |
| 5    | BF16  | 2       | 1      | 3     | 16    | 2    | 2×24,576 |
| 6    | FP16  | 1       | 1      | 2     | 16    | 2    | 2×24,576 |
| 7    | FP16  | 7       | 1      | 8     | 16    | 2    | 2×24,576 |
| 8    | FP16  | 32      | 1      | 33    | 16    | 2    | 2×24,576 |
| 9    | FP16  | 128     | 1      | 129   | 16    | 2    | 2×24,576 |
| **Total** | | | | | | | **1,421,312** |

**Parameters:**
- Batch size: 1 (decode only)
- Sequence length: prefill + 1 (decode token)
- Cache length: prefill + 1
- Two concurrent requests with separate caches
- Two repetitions per test case

## Implementation Status

### Core Components (Frozen)
✓ Input RMSNorm (Phase 8.2 accepted)
✓ GEMM kernels (Phase 8.1 accepted, AVX2 compatible)
✓ RoPE (Phase 8.2 accepted)
✓ Post-attention/layer RMSNorm (Phase 8.2 accepted)
✓ Weight Registry, Worker, Memory Manager

### Native Attention (New for Phase 9)
⏳ QK kernel (Phase 9.1)
⏳ Softmax kernel (Phase 9.2)
⏳ AV kernel (Phase 9.3)
⏳ Cache integration (Phase 9.4)
⏳ Dispatch methods (attention_qk, attention_softmax, attention_av)

### Test Harness
⏳ `runtime/native/r6_native_attention_test.cpp` — Updated r6_test.cpp with native attention

## Checkpoint Comparison Stages

| Stage | Expected | Description |
|-------|----------|-------------|
| input_norm | 36 | 2 requests × 2 reps × 9 cases ÷ baseline chunks |
| projected_q | 36 | Native Q projection output |
| projected_kv_cache | 36 | Native KV cache (via GEMM chain) |
| attention_qk | 36 | Native QK: raw, scaled, masked |
| attention_softmax | 36 | Native Softmax: max, denom, probs |
| attention_output | 36 | Native AV output (pre-projection) |
| attention_residual | 36 | Attention output + residual input (BF16) |
| post_norm | 36 | Post-attention RMSNorm |
| ffn_gate | 36 | Gate projection (independent, known failures from Phase 8) |
| ffn_up | 36 | Up projection |
| ffn_down | 36 | Down projection |
| final_output | 36 | Layer output after residual |

## Numerical Contract

**Tolerance:** `1e-5 + 1e-5 * abs(reference)`

**Reference:** Checkpoint PyTorch with FP32 intermediates for all GEMM/attention operations, BF16/FP16 at model boundaries.

**Bitwise parity:** None required (Phase 8.2 also allowed tolerance)

## Validation Strategy

### Stage-by-stage comparison
1. Each checkpoint stage compared independently
2. All values must pass threshold
3. No silent fallback or implicit retry

### Request concurrency
- Request A and B must not interfere
- Independent cache validation per request

### Numerical consistency
- Both repetitions must produce identical results
- Deterministic input seeding

### Resource cleanup
- Zero device memory leaked
- All weights released to registry
- Both request caches freed

## Expected Failure Points (from Phase 8)

**Known issues to watch:**

1. **FFN projections:** Phase 8.2 showed failures in gate/up/down. These are UNCHANGED in Phase 9, so same failures expected.
   - Expected: Gate 1,754 mismatches, Up 2,412, Down 1,948
   - **Action:** Document as Phase 8 regression, not Phase 9 failure

2. **Residual accumulation:** 4 mismatches in Phase 8.2 at attention residual
   - Root cause: BF16 midpoint rounding behavior
   - **Action:** Reproduce same mismatches; if different, investigate

3. **Final output:** 404 mismatches propagated from FFN failures
   - **Action:** This is expected; Phase 9 target is to match Phase 8 numerical behavior

## Test Command

```bash
cd /app/shuwen/mlu-operator

# Build test harness with native attention
./runtime/native/phase9/build_full_layer.sh

# Run all nine test cases
./artifacts/build/r6_native_attention_test \
    --authorized-reference-mla /path/to/fixture/dir

# Compare results
python3 runtime/native/phase9/validate_layer0.py \
    artifacts/logs/phase9_layer0_accuracy.csv
```

## Acceptance Report Contents

The final `PHASE9_LAYER0_HARDWARE_ACCEPTANCE.md` will contain:

1. **Execution summary:** All nine cases executed, no crashes
2. **Device coverage:** MLU590-M9 PCI addresses, driver/CNCC versions
3. **Numerical results:** CSV with stage comparisons
4. **FFN disclaimer:** Note that gate/up/down failures are Phase 8 regressions
5. **Attention metrics:** QK, Softmax, AV stage results
6. **Cache lifecycle:** Verification of prefill/decode/cleanup
7. **Resource tracking:** Peak memory, cleanup validation
8. **Bitwise parity:** None required; tolerance-based acceptance

## Acceptance Criteria

| Criterion | Requirement | Status |
|-----------|-------------|--------|
| All kernels implemented | Phase 9.1–9.3 complete | ⏳ |
| Cache integration working | Phase 9.4 passed | ⏳ |
| Nine test cases executed | No crashes, all complete | ⏳ |
| Attention stages numerical | Within tolerance across all cases | ⏳ |
| FFN stages (known failures) | Documented as Phase 8 regressions | ⏳ |
| Final output acceptable | Mismatches match Phase 8 baseline | ⏳ |
| Two concurrent requests | No cross-contamination | ⏳ |
| Two repetitions identical | Deterministic results | ⏳ |
| Resource cleanup | Zero tracked bytes remaining | ⏳ |
| No silent fallback | Explicit pass/fail only | ⏳ |

## Known Issues / Risks

1. **FFN regressions:** Phase 8.2 showed ~400 final mismatches. Phase 9 may reproduce or improve, but target is "match Phase 8", not "fix Phase 8".
2. **Dispatch integration:** Dispatch methods are stubs; must implement actual MLU kernel calls.
3. **Concurrency subtleties:** Queue synchronization placement is critical; too few syncs → cache corruption, too many → serialization.

## Dependency Chain

- Requires Phase 9.1, 9.2, 9.3, 9.4 to complete
- Marks Phase 9 implementation complete (but not performance validation)

## Deliverables

1. `docs/PHASE9_LAYER0_HARDWARE_ACCEPTANCE.md` — Acceptance report with full results
2. `runtime/native/r6_native_attention_test.cpp` — Updated r6_test.cpp
3. Build/validation scripts in `runtime/native/phase9/`
4. Checkpoint accuracy CSV: `artifacts/logs/phase9_layer0_accuracy.csv`
5. Hardware execution log: `artifacts/logs/phase9_layer0_hardware.log`

## Next Steps

1. Complete Phase 9.1–9.4 implementation
2. Build and run r6_native_attention_test on hardware
3. Compare all stages to checkpoint reference
4. Document any new failures or improvements
5. If numerical acceptance met:
   - Proceed to Phase 9.6 (performance validation)
   - Measure QK, Softmax, AV latencies
   - Measure prefill throughput, decode TPOT
6. If numerical acceptance failed:
   - Create root cause analysis
   - Identify fixable vs. fundamental limitations
   - Proposal for Phase 10

---

**Status:** Design phase complete. Awaiting Phase 9.1–9.4 implementations.

**Last updated:** 2026-11-21
