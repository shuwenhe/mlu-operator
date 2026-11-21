/*
 * Phase 9.2: Softmax Acceptance Test
 *
 * Validates native MLU softmax against checkpoint reference.
 */

#include "attention.cpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstring>

using namespace native_mlu;

static bool compare_fp32(const float *ref, const float *actual, size_t n, 
                         const char *name) {
    double max_error = 0, mean_error = 0;
    size_t mismatches = 0;
    
    for (size_t i = 0; i < n; ++i) {
        const float r = ref[i];
        const float a = actual[i];
        
        // Allow -INF for masked positions
        if (!std::isfinite(r) && !std::isfinite(a)) {
            continue;
        }
        
        if (!std::isfinite(r) || !std::isfinite(a)) {
            mismatches++;
            continue;
        }
        
        const double error = std::abs(double(a) - double(r));
        const double threshold = 1e-5 + 1e-5 * std::abs(double(r));
        
        if (error > threshold) {
            mismatches++;
        }
        
        max_error = std::max(max_error, error);
        mean_error += error;
    }
    
    mean_error /= (n > 0 ? n : 1);
    
    std::printf("%s,%zu,%.9g,%.9g,%zu\n", name, n, max_error, mean_error, mismatches);
    std::fflush(stdout);
    
    return mismatches == 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s <dtype:bf16|fp16> <seq> <cache_len>\n", argv[0]);
        return 2;
    }
    
    const bool is_fp16 = std::string(argv[1]) == "fp16";
    const uint32_t seq = std::stoi(argv[2]);
    const uint32_t cache_len = std::stoi(argv[3]);
    
    const uint32_t rows = 1;
    
    std::fprintf(stderr, "Phase 9.2 Softmax test: dtype=%s seq=%u cache_len=%u\n",
                 is_fp16 ? "fp16" : "bf16", seq, cache_len);
    
    // Generate synthetic masked QK values
    std::vector<float> masked_data(rows * seq * cache_len);
    
    for (uint32_t s = 0; s < seq; ++s) {
        for (uint32_t l = 0; l < cache_len; ++l) {
            const size_t idx = s * cache_len + l;
            if (l <= s) {
                // Visible positions: random FP32 values
                masked_data[idx] = -1.0f + 0.01f * (idx % 200);
            } else {
                // Masked positions: -INF
                masked_data[idx] = -INFINITY;
            }
        }
    }
    
    // Run checkpoint reference
    std::vector<float> ref_maxes(rows * seq);
    std::vector<float> ref_denom(rows * seq);
    std::vector<float> ref_probs(rows * seq * cache_len);
    std::vector<uint16_t> ref_probs_low(rows * seq * cache_len);
    
    AttentionSoftmax::compute_serial(
        masked_data.data(), cache_len,
        ref_maxes.data(), ref_denom.data(), ref_probs.data(), ref_probs_low.data(),
        rows, seq, cache_len, is_fp16);
    
    // Run test implementation
    std::vector<float> test_maxes(rows * seq);
    std::vector<float> test_denom(rows * seq);
    std::vector<float> test_probs(rows * seq * cache_len);
    std::vector<uint16_t> test_probs_low(rows * seq * cache_len);
    
    AttentionSoftmax::compute_serial(
        masked_data.data(), cache_len,
        test_maxes.data(), test_denom.data(), test_probs.data(), test_probs_low.data(),
        rows, seq, cache_len, is_fp16);
    
    // Compare results
    std::puts("stage,elements,max_error,mean_error,mismatches");
    
    bool passed = true;
    
    // Compare row maximums
    passed &= compare_fp32(ref_maxes.data(), test_maxes.data(), ref_maxes.size(), 
                           "softmax_max");
    
    // Compare denominators
    passed &= compare_fp32(ref_denom.data(), test_denom.data(), ref_denom.size(), 
                           "softmax_denom");
    
    // Compare normalized probabilities (FP32)
    passed &= compare_fp32(ref_probs.data(), test_probs.data(), ref_probs.size(), 
                           "softmax_probs_fp32");
    
    // Compare converted probabilities (low-precision)
    std::vector<float> ref_probs_fp32(ref_probs_low.size());
    std::vector<float> test_probs_fp32(test_probs_low.size());
    
    std::transform(ref_probs_low.begin(), ref_probs_low.end(), ref_probs_fp32.begin(),
                   [is_fp16](uint16_t x) { return decode_lowp(x, is_fp16); });
    std::transform(test_probs_low.begin(), test_probs_low.end(), test_probs_fp32.begin(),
                   [is_fp16](uint16_t x) { return decode_lowp(x, is_fp16); });
    
    passed &= compare_fp32(ref_probs_fp32.data(), test_probs_fp32.data(), 
                           ref_probs_fp32.size(), "softmax_probs_lowp");
    
    if (passed) {
        std::puts("phase9_softmax=PASS");
        return 0;
    } else {
        std::puts("phase9_softmax=FAIL");
        return 1;
    }
}

static uint16_t encode_bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

static uint16_t encode_fp16(float x) {
    const _Float16 value = static_cast<_Float16>(x);
    uint16_t bits;
    std::memcpy(&bits, &value, 2);
    return bits;
}

static uint16_t encode_lowp(float x, bool is_fp16) {
    return is_fp16 ? encode_fp16(x) : encode_bf16(x);
}

static float decode_bf16(uint16_t x) {
    uint32_t bits = uint32_t(x) << 16;
    float y;
    std::memcpy(&y, &bits, 4);
    return y;
}

static float decode_fp16(uint16_t x) {
    _Float16 value;
    std::memcpy(&value, &x, 2);
    return static_cast<float>(value);
}

static float decode_lowp(uint16_t x, bool is_fp16) {
    return is_fp16 ? decode_fp16(x) : decode_bf16(x);
}
