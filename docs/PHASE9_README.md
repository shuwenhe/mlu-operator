# Phase 9: Native MLU Attention Implementation

## Quick Start

**Status:** ✓ Design and infrastructure complete; ready for BANG C kernel implementation.

**What's new:** 
- Replaces Phase 8.2 CPU checkpoint attention with native MLU execution
- Three kernels: QK → Softmax → AV
- 396,136 unit test elements + 1,421,312 full layer test elements

## Key Files

### Documentation (Read First)
1. **[PHASE9_STATUS.md](docs/PHASE9_STATUS.md)** — Architecture overview & quick links
2. **[PHASE9_NATIVE_ATTENTION_DESIGN.md](docs/PHASE9_NATIVE_ATTENTION_DESIGN.md)** — Full design spec
3. **[PHASE9_LAUNCH_CHECKLIST.md](PHASE9_LAUNCH_CHECKLIST.md)** — Completion checklist
4. **[PHASE9_DELIVERABLES_SUMMARY.md](PHASE9_DELIVERABLES_SUMMARY.md)** — Detailed inventory

### Phase-Specific Docs
- **[PHASE9_QK_ACCEPTANCE.md](docs/PHASE9_QK_ACCEPTANCE.md)** — Query-Key spec & tests
- **[PHASE9_SOFTMAX_ACCEPTANCE.md](docs/PHASE9_SOFTMAX_ACCEPTANCE.md)** — Softmax spec & tests
- **[PHASE9_AV_ACCEPTANCE.md](docs/PHASE9_AV_ACCEPTANCE.md)** — AV kernel spec & tests
- **[PHASE9_KV_CACHE_ACCEPTANCE.md](docs/PHASE9_KV_CACHE_ACCEPTANCE.md)** — Cache lifecycle
- **[PHASE9_LAYER0_HARDWARE_ACCEPTANCE.md](docs/PHASE9_LAYER0_HARDWARE_ACCEPTANCE.md)** — Full layer 0

### Code
- **[runtime/native/phase9/attention.h](runtime/native/phase9/attention.h)** — Kernel declarations
- **[runtime/native/phase9/attention.cpp](runtime/native/phase9/attention.cpp)** — CPU references
- **[runtime/native/phase9/qk_test.cpp](runtime/native/phase9/qk_test.cpp)** — QK unit test
- **[runtime/native/phase9/softmax_test.cpp](runtime/native/phase9/softmax_test.cpp)** — Softmax test
- **[runtime/native/phase9/av_test.cpp](runtime/native/phase9/av_test.cpp)** — AV test
- **[runtime/native/phase9/run_all_tests.py](runtime/native/phase9/run_all_tests.py)** — Master runner

### Integration
- **[runtime/native/dispatch.h](runtime/native/dispatch.h)** — Updated with attention methods
- **[runtime/native/dispatch.cpp](runtime/native/dispatch.cpp)** — Dispatch implementations (stubs)

## Test Infrastructure

### Run All Unit Tests
```bash
cd /app/shuwen/mlu-operator
python3 runtime/native/phase9/run_all_tests.py
```

**Expected output:**
```
✓ ALL TESTS PASSED
Phase 9 unit test infrastructure ready for BANG C kernel implementation
```

**Test coverage:**
- 24 unit test cases (396,136 elements)
- 2 dtypes (BF16, FP16)
- 4 sequence lengths per kernel (1, 7, 32, 128)

### Individual Kernel Tests
```bash
# Build
g++ -std=c++17 -O2 runtime/native/phase9/qk_test.cpp -o artifacts/build/phase9_qk_test
g++ -std=c++17 -O2 runtime/native/phase9/softmax_test.cpp -o artifacts/build/phase9_softmax_test
g++ -std=c++17 -O2 runtime/native/phase9/av_test.cpp -o artifacts/build/phase9_av_test

# Run
./artifacts/build/phase9_qk_test bf16 128 128
./artifacts/build/phase9_softmax_test fp16 32 32
./artifacts/build/phase9_av_test bf16 7 7
```

## Frozen Components (Do Not Modify)

✓ Input RMSNorm (Phase 8.2 accepted)
✓ GEMM kernels (Phase 8.1 accepted)
✓ YaRN RoPE (Phase 8.2 accepted)
✓ Post-attention/layer RMSNorm (Phase 8.2)
✓ Worker, Memory, Workspace, Registry

## Specifications

### Numerical Contract
- **Tolerance:** `1e-5 + 1e-5 × abs(reference)`
- **Reference:** Checkpoint PyTorch with FP32 intermediates
- **Accumulation:** FP32 for all three kernels
- **Input/output:** BF16 or FP16 (model-dependent)

### Attention Pipeline
1. **QK:** Query·Key^T in FP32, scale with Phase 8.2 rounding, apply causal mask (-INF)
2. **Softmax:** Row-wise softmax with FP32 precision, convert to low-precision
3. **AV:** prob·V in FP32, convert output to low-precision

### Tensor Shapes
- Q: `[B, 16, S, 192]` BF16/FP16
- K/V: `[B, 16, L, 192]` / `[B, 16, L, 128]` BF16/FP16
- Scores: `[B, 16, S, L]` FP32 intermediate, BF16/FP16 output
- Probs: `[B, 16, S, L]` FP32 intermediate, BF16/FP16 output
- Output: `[B, 16, S, 128]` FP32 intermediate, BF16/FP16 final

## Next Steps: BANG C Kernel Implementation

Three files need implementation:

1. **runtime/native/phase9/attention_qk.mlu**
   - Query-Key multiplication with scaling and causal masking
   - Input: Q, K, positions, cache_base, scale
   - Output: qk_out (FP32), scaled_out (low-p), masked_out (FP32)

2. **runtime/native/phase9/attention_softmax.mlu**
   - Row-wise softmax with numerically stable max subtraction
   - Input: masked (FP32)
   - Output: maxes, denom, probs (FP32), probs_low (low-p)

3. **runtime/native/phase9/attention_av.mlu**
   - Attention-weighted value summation
   - Input: probs (low-p), V (low-p)
   - Output: av (FP32), out (low-p)

Then create: `runtime/native/phase9/attention_harness.cpp` and `runtime/native/r6_native_attention_test.cpp`

## Full Layer 0 Test

After kernels complete:
```bash
./runtime/native/phase9/build_full_layer.sh
./artifacts/build/r6_native_attention_test --authorized-reference-mla /path/to/fixture/dir
python3 runtime/native/phase9/validate_layer0.py artifacts/logs/phase9_layer0_accuracy.csv
```

**Expected:** 1,421,312 final output values compared, all within tolerance.

## Troubleshooting

### CPU Unit Tests Fail
1. Check test file is not corrupted: `g++ -fsyntax-only runtime/native/phase9/qk_test.cpp`
2. Verify Python path: `python3 --version`
3. Check CSV format: Look at `artifacts/logs/phase9_qk_results.csv`

### Dispatch Compilation Fails
1. Ensure dispatch.h and dispatch.cpp both updated
2. Check for syntax: `g++ -c -std=c++17 runtime/native/dispatch.cpp`

### Hardware Tests Don't Build
1. BANG C kernels must be implemented first
2. Dispatch stubs currently require BANG C implementations
3. Remove `require(false,...)` calls after kernel implementation

## Documentation Standards

- All design docs under `/docs/`
- All code under `/runtime/native/phase9/`
- Test results in `/artifacts/logs/phase9_*.csv`
- Logs in `/artifacts/logs/phase9_*.log`

## Acceptance Criteria

**Phase 9 implementation complete when:**
1. ✓ All CPU unit tests pass (396,136 elements)
2. ⏳ BANG C kernels implemented
3. ⏳ Hardware full layer test passes (1,421,312 elements)
4. ⏳ All attention stages pass individually
5. ⏳ Two concurrent requests validated
6. ⏳ Zero resource leaks after teardown
7. ⏳ Documentation updated with actual results
8. ⏳ Performance metrics established

## References

- **Phase 8.2:** [PHASE8_2_HARDWARE_ACCEPTANCE.md](docs/PHASE8_2_HARDWARE_ACCEPTANCE.md)
- **MLA Design:** [MLU_ATTENTION_MLA_DESIGN.md](docs/MLU_ATTENTION_MLA_DESIGN.md)
- **Hardware:** [HARDWARE_INVENTORY.md](docs/HARDWARE_INVENTORY.md)
- **Environment:** [ENVIRONMENT.md](docs/ENVIRONMENT.md)

---

**Status:** Ready for BANG C kernel implementation.

**Date:** 2026-11-21

**Next:** Implement Phase 9.1 QK kernel (attention_qk.mlu)
