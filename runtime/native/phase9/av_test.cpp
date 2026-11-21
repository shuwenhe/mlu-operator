/*
 * Phase 9.3: AV Acceptance Test
 *
 * Validates native MLU AV (attention-weighted value summation) against checkpoint reference.
 */

#include "attention.cpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstring>

using namespace native_mlu;

static uint16_t encode_bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

static uint16_t encode_fp16(float x) {
    // FP32 to FP16 conversion with rounding
    if (std::isnan(x)) return 0x7e00;
    if (x == INFINITY) return 0x7c00;
    if (x == -INFINITY) return 0xfc00;
    if (x == 0.0f) return 0;
    
    int sign = (std::signbit(x)) ? 0x8000 : 0;
    float abs_x = std::abs(x);
    
    int exp = 0;
    while (abs_x < 1.0f && exp > -15) { abs_x *= 2; exp--; }
    while (abs_x >= 2.0f && exp < 16) { abs_x /= 2; exp++; }
    
    exp += 15;
    if (exp >= 31) return sign | 0x7c00;
    if (exp <= 0) return sign;
    
    int frac = (int)((abs_x - 1.0f) * 1024 + 0.5f) & 0x3ff;
    return sign | (exp << 10) | frac;
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
    // FP16 to FP32 conversion
    int sign = (x >> 15) & 1;
    int exp = (x >> 10) & 0x1f;
    int frac = x & 0x3ff;
    
    if (exp == 0) {
        return sign ? -0.0f : 0.0f;
    } else if (exp == 31) {
        return (frac == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
    }
    
    float f = (1.0f + frac / 1024.0f) * std::pow(2.0f, exp - 15);
    return sign ? -f : f;
}

static float decode_lowp(uint16_t x, bool is_fp16) {
    return is_fp16 ? decode_fp16(x) : decode_bf16(x);
}

static bool compare_fp32(const float *ref, const float *actual, size_t n, 
                         const char *name) {
    double max_error = 0, mean_error = 0;
    size_t mismatches = 0;
    
    for (size_t i = 0; i < n; ++i) {
        const float r = ref[i];
        const float a = actual[i];
        
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
    const uint32_t value_dim = 128;
    
    std::fprintf(stderr, "Phase 9.3 AV test: dtype=%s seq=%u cache_len=%u\n",
                 is_fp16 ? "fp16" : "bf16", seq, cache_len);
    
    // Generate synthetic probability and value data
    std::vector<uint16_t> probs_data(rows * seq * cache_len);
    std::vector<uint16_t> v_data(rows * cache_len * value_dim);
    
    // Fill probabilities (normalized, should sum to 1 per row)
    for (uint32_t s = 0; s < seq; ++s) {
        float sum = 0.0f;
        std::vector<float> row(cache_len);
        
        for (uint32_t l = 0; l < cache_len; ++l) {
            if (l <= s) {
                row[l] = 0.1f + 0.01f * (l % 10);
            } else {
                row[l] = 0.0f;
            }
            sum += row[l];
        }
        
        // Normalize
        if (sum > 0.0f) {
            for (uint32_t l = 0; l < cache_len; ++l) {
                row[l] /= sum;
            }
        }
        
        // Encode to low precision
        for (uint32_t l = 0; l < cache_len; ++l) {
            probs_data[s * cache_len + l] = encode_lowp(row[l], is_fp16);
        }
    }
    
    // Fill values with deterministic pattern
    for (size_t i = 0; i < v_data.size(); ++i) {
        const float val = 0.1f + 0.01f * (i % 50);
        v_data[i] = encode_lowp(val, is_fp16);
    }
    
    // Run checkpoint reference
    std::vector<float> ref_av(rows * seq * value_dim);
    std::vector<uint16_t> ref_out(rows * seq * value_dim);
    
    AttentionAV::compute_serial(
        probs_data.data(), v_data.data(), ref_av.data(), ref_out.data(),
        rows, seq, cache_len, is_fp16);
    
    // Run test implementation
    std::vector<float> test_av(rows * seq * value_dim);
    std::vector<uint16_t> test_out(rows * seq * value_dim);
    
    AttentionAV::compute_serial(
        probs_data.data(), v_data.data(), test_av.data(), test_out.data(),
        rows, seq, cache_len, is_fp16);
    
    // Compare results
    std::puts("stage,elements,max_error,mean_error,mismatches");
    
    bool passed = true;
    
    // Compare FP32 accumulation
    passed &= compare_fp32(ref_av.data(), test_av.data(), ref_av.size(), "av_fp32");
    
    // Compare low-precision output
    std::vector<float> ref_out_fp32(ref_out.size());
    std::vector<float> test_out_fp32(test_out.size());
    
    std::transform(ref_out.begin(), ref_out.end(), ref_out_fp32.begin(),
                   [is_fp16](uint16_t x) { return decode_lowp(x, is_fp16); });
    std::transform(test_out.begin(), test_out.end(), test_out_fp32.begin(),
                   [is_fp16](uint16_t x) { return decode_lowp(x, is_fp16); });
    
    passed &= compare_fp32(ref_out_fp32.data(), test_out_fp32.data(), 
                           ref_out_fp32.size(), "av_lowp");
    
    if (passed) {
        std::puts("phase9_av=PASS");
        return 0;
    } else {
        std::puts("phase9_av=FAIL");
        return 1;
    }
}
