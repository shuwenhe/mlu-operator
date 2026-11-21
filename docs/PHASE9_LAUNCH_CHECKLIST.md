# Phase 9 Launch Checklist

**Date:** 2026-11-21  
**Status:** Ready for BANG C kernel implementation

## Design & Specification ✓

- [x] Phase 9 architecture documented (PHASE9_NATIVE_ATTENTION_DESIGN.md)
- [x] Implementation roadmap created (PHASE9_IMPLEMENTATION_ROADMAP.md)
- [x] All frozen components identified (RMSNorm, GEMM, RoPE)
- [x] Numerical contract specified (1e-5 + 1e-5 × abs(ref))
- [x] Cache lifecycle documented (PHASE9_KV_CACHE_ACCEPTANCE.md)
- [x] Full layer 0 test matrix defined (9 cases, 1,421,312 final values)

## Unit Test Infrastructure ✓

- [x] QK test harness (qk_test.cpp, 8 test cases, 104,748 elements)
- [x] Softmax test harness (softmax_test.cpp, 8 test cases, 205,420 elements)
- [x] AV test harness (av_test.cpp, 8 test cases, 85,968 elements)
- [x] CPU reference implementations (attention.cpp)
- [x] Validation scripts (validate_qk.py, validate_softmax.py, validate_av.py)
- [x] Master test runner (run_all_tests.py)

## C++ Integration ✓

- [x] Dispatch header updated (dispatch.h with attention methods)
- [x] Dispatch stub implementations added (dispatch.cpp)
- [x] Attention kernel headers (attention.h with 6 function signatures)
- [x] BF16/FP16 encoding/decoding utilities implemented
- [x] Serial CPU implementations for reference (all three kernels)

## Documentation Deliverables ✓

| Document | Status | Purpose |
|----------|--------|---------|
| PHASE9_NATIVE_ATTENTION_DESIGN.md | ✓ | Architecture, frozen components, phases overview |
| PHASE9_IMPLEMENTATION_ROADMAP.md | ✓ | Detailed roadmap, test matrices, scripts |
| PHASE9_QK_ACCEPTANCE.md | ✓ | QK kernel spec and unit tests |
| PHASE9_SOFTMAX_ACCEPTANCE.md | ✓ | Softmax kernel spec and unit tests |
| PHASE9_AV_ACCEPTANCE.md | ✓ | AV kernel spec and unit tests |
| PHASE9_KV_CACHE_ACCEPTANCE.md | ✓ | Cache lifecycle, concurrent requests |
| PHASE9_LAYER0_HARDWARE_ACCEPTANCE.md | ✓ | Full layer 0 acceptance criteria |
| PHASE9_STATUS.md | ✓ | Implementation status, quick links |

## File Inventory ✓

```
runtime/native/phase9/
├── attention.h                 # Kernel declarations (6 functions)
├── attention.cpp               # CPU reference implementations (3 classes)
├── qk_test.cpp                 # QK unit test (8 cases)
├── softmax_test.cpp            # Softmax unit test (8 cases)
├── av_test.cpp                 # AV unit test (8 cases)
├── run_all_tests.py            # Master test runner
├── validate_qk.py              # QK validation
├── validate_softmax.py          # Softmax validation
├── validate_av.py              # AV validation
├── attention_qk.mlu            # ⏳ PENDING: BANG C kernel
├── attention_softmax.mlu        # ⏳ PENDING: BANG C kernel
├── attention_av.mlu            # ⏳ PENDING: BANG C kernel
├── attention_harness.cpp       # ⏳ PENDING: Integration test
└── build_full_layer.sh         # ⏳ PENDING: Build script

runtime/native/
├── dispatch.h                  # ✓ Updated with attention methods
├── dispatch.cpp                # ✓ Updated with attention stubs
└── r6_native_attention_test.cpp # ⏳ PENDING: Updated r6_test.cpp

docs/
├── PHASE9_NATIVE_ATTENTION_DESIGN.md ✓
├── PHASE9_IMPLEMENTATION_ROADMAP.md ✓
├── PHASE9_QK_ACCEPTANCE.md ✓
├── PHASE9_SOFTMAX_ACCEPTANCE.md ✓
├── PHASE9_AV_ACCEPTANCE.md ✓
├── PHASE9_KV_CACHE_ACCEPTANCE.md ✓
├── PHASE9_LAYER0_HARDWARE_ACCEPTANCE.md ✓
└── PHASE9_STATUS.md ✓
```

## Numerical Test Coverage ✓

| Component | Elements | Test Cases | Dtypes | Coverage |
|-----------|----------|-----------|--------|----------|
| QK kernel | 104,748 | 8 | BF16, FP16 | seq={1,7,32,128} |
| Softmax | 205,420 | 8 | BF16, FP16 | seq={1,7,32,128} |
| AV kernel | 85,968 | 8 | BF16, FP16 | seq={1,7,32,128} |
| **Unit test total** | **396,136** | **24** | | |
| Full Layer 0 | 1,421,312 | 9 | BF16, FP16 | 9 cases |
| **Grand total** | **1,817,448** | **33** | | |

## CPU Reference Validation Status

**Run before BANG C implementation:**
```bash
cd /app/shuwen/mlu-operator
python3 runtime/native/phase9/run_all_tests.py
```

Expected output:
```
✓ ALL TESTS PASSED
Phase 9 unit test infrastructure ready for BANG C kernel implementation
```

## BANG C Kernel Implementation (Next Phase)

### QK Kernel (attention_qk.mlu)
- [ ] Implement Q·K^T accumulation (FP32)
- [ ] Scale with BF16/FP16 rounding
- [ ] Apply causal mask (-INF at future positions)
- [ ] Link to dispatch.cpp::Dispatch::attention_qk

### Softmax Kernel (attention_softmax.mlu)
- [ ] Find row max (FP32)
- [ ] Exponentiate with numerically stable max subtraction
- [ ] Accumulate denominator (FP32)
- [ ] Normalize and convert to low precision
- [ ] Link to dispatch.cpp::Dispatch::attention_softmax

### AV Kernel (attention_av.mlu)
- [ ] Accumulate prob·V in FP32
- [ ] Convert output to low precision
- [ ] Link to dispatch.cpp::Dispatch::attention_av

### Full Layer Test (r6_native_attention_test.cpp)
- [ ] Update r6_test.cpp to call native attention dispatch methods
- [ ] Remove CPU checkpoint attention reference
- [ ] Validate all nine test cases
- [ ] Generate PHASE9_LAYER0_HARDWARE_ACCEPTANCE.md

### Build & Integration
- [ ] Create build_full_layer.sh script
- [ ] Link BANG C object files to test harness
- [ ] Compile and run on MLU590-M9 hardware
- [ ] Generate acceptance reports

## Success Criteria

**Phase 9 implementation complete when:**

1. **CPU reference tests pass:** `python3 run_all_tests.py` exits 0
2. **BANG C kernels implemented:** All three .mlu files compiled and linked
3. **Unit tests pass on hardware:** Each kernel tested individually
4. **Integration test passes:** Full layer 0 test with native attention only
5. **Numerical acceptance achieved:** 1,421,312 final values within tolerance
6. **Cache lifecycle validated:** Two concurrent requests, proper cleanup
7. **Documentation complete:** All acceptance reports updated with actual results
8. **Performance baseline established:** Measure QK/Softmax/AV latencies

## Handoff to Next Phase

**After Phase 9.5 completion:**

- Phase 9.6: Performance optimization (optional)
  - Measure QK/Softmax/AV latencies
  - Measure prefill throughput, decode TPOT
  - Compare vs. Phase 8.2 CPU attention
  - Identify bottlenecks

- Phase 10 (if needed):
  - If numerical acceptance failed: Root cause analysis
  - If performance insufficient: Optimization strategies
  - If all passed: Production deployment planning

## References

- Phase 8.2 completion: [PHASE8_2_HARDWARE_ACCEPTANCE.md](docs/PHASE8_2_HARDWARE_ACCEPTANCE.md)
- Current implementation: [PHASE9_STATUS.md](docs/PHASE9_STATUS.md)
- Detailed roadmap: [PHASE9_IMPLEMENTATION_ROADMAP.md](docs/PHASE9_IMPLEMENTATION_ROADMAP.md)
- MLU info: [HARDWARE_INVENTORY.md](docs/HARDWARE_INVENTORY.md)

## Sign-Off

- **Design:** ✓ Complete
- **Specification:** ✓ Complete
- **Unit test infrastructure:** ✓ Complete
- **C++ integration:** ✓ Complete
- **Documentation:** ✓ Complete

**Ready for BANG C kernel implementation.**

---

**Prepared by:** Copilot SDK  
**Date:** 2026-11-21  
**Next step:** Implement BANG C kernels (Phase 9.1 QK)
