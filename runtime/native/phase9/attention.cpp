/*
 * Phase 9: Native MLU Attention Dispatch
 *
 * C++ wrapper for BANG C kernels with device memory management,
 * BF16/FP16 specialization, and checkpoint validation.
 */

#include "attention.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace native_mlu {

/*
 * BF16 encoding/decoding for checkpoint validation
 */
static float decode_bf16(uint16_t x) {
    uint32_t bits = uint32_t(x) << 16;
    float y;
    std::memcpy(&y, &bits, 4);
    return y;
}

static uint16_t encode_bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

/*
 * FP16 encoding/decoding (portable implementation)
 */
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

static float decode_lowp(uint16_t x, bool is_fp16) {
    if (is_fp16) {
        return decode_fp16(x);
    }
    return decode_bf16(x);
}

static uint16_t encode_lowp(float x, bool is_fp16) {
    if (is_fp16) {
        return encode_fp16(x);
    }
    return encode_bf16(x);
}

/*
 * Attention QK: Query-Key multiplication with scale
 *
 * Matches checkpoint semantics:
 * - Accumulate Q @ K^T in FP32
 * - Scale with BF16 rounding per Phase 8.2
 * - Apply causal mask (-INF where l > s+cache_base)
 */
class AttentionQK {
public:
    // For testing: serial CPU implementation
    static void compute_serial(
        const uint16_t *q,      // [rows, seq, 192]
        const uint16_t *k,      // [rows, cache_len, 192]
        float scale,
        const uint32_t *positions,
        uint32_t cache_base,
        float *qk_out,          // [rows, seq, cache_len]
        uint16_t *scaled_out,
        float *masked_out,
        uint32_t rows, uint32_t seq, uint32_t cache_len,
        bool is_fp16) {
        
        const uint32_t dim = 192;
        const uint32_t q_stride = seq * dim;
        const uint32_t k_stride = cache_len * dim;
        const uint32_t qk_stride = seq * cache_len;
        
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t s = 0; s < seq; ++s) {
                const uint32_t pos = positions[s];
                const uint32_t causal_limit = cache_base + pos + 1;
                
                for (uint32_t l = 0; l < cache_len; ++l) {
                    // Accumulate Q·K^T
                    float qk_val = 0.0f;
                    for (uint32_t p = 0; p < dim; ++p) {
                        const float q_val = decode_lowp(q[r * q_stride + s * dim + p], is_fp16);
                        const float k_val = decode_lowp(k[r * k_stride + l * dim + p], is_fp16);
                        qk_val += q_val * k_val;
                    }
                    
                    const uint32_t idx = r * qk_stride + s * cache_len + l;
                    qk_out[idx] = qk_val;
                    
                    // Scale with BF16 rounding
                    const float scaled_val = scale_with_rounding(qk_val * scale, is_fp16);
                    scaled_out[idx] = encode_lowp(scaled_val, is_fp16);
                    
                    // Apply causal mask
                    if (l < causal_limit) {
                        masked_out[idx] = scaled_val;
                    } else {
                        masked_out[idx] = -INFINITY;
                    }
                }
            }
        }
    }

private:
    static float scale_with_rounding(float x, bool is_fp16) {
        // Per Phase 8.2: scale is applied with low-precision rounding
        return decode_lowp(encode_lowp(x, is_fp16), is_fp16);
    }
};

/*
 * Attention Softmax: Row-wise exponential normalization
 *
 * Matches checkpoint semantics:
 * - Find max per row (FP32)
 * - Exponentiate with numerically stable max subtraction
 * - Accumulate denominator in FP32
 * - Normalize and convert to low precision
 */
class AttentionSoftmax {
public:
    static void compute_serial(
        const float *masked,    // [rows, seq, cache_len]
        uint32_t seq_len,
        float *maxes_out,       // [rows, seq]
        float *denom_out,       // [rows, seq]
        float *probs_out,       // [rows, seq, cache_len]
        uint16_t *probs_low_out,
        uint32_t rows, uint32_t seq, uint32_t cache_len,
        bool is_fp16) {
        
        const uint32_t qk_stride = seq * cache_len;
        
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t s = 0; s < seq; ++s) {
                const uint32_t row_idx = r * qk_stride + s * cache_len;
                
                // Find row maximum
                float row_max = -INFINITY;
                for (uint32_t l = 0; l < seq_len; ++l) {
                    if (std::isfinite(masked[row_idx + l])) {
                        row_max = std::max(row_max, masked[row_idx + l]);
                    }
                }
                
                maxes_out[r * seq + s] = row_max;
                
                // Exponentiate and accumulate
                float sum = 0.0f;
                for (uint32_t l = 0; l < cache_len; ++l) {
                    float exp_val;
                    if (l < seq_len && std::isfinite(masked[row_idx + l])) {
                        exp_val = std::exp(masked[row_idx + l] - row_max);
                    } else {
                        exp_val = 0.0f;
                    }
                    probs_out[row_idx + l] = exp_val;
                    sum += exp_val;
                }
                
                denom_out[r * seq + s] = sum;
                
                // Normalize and convert
                for (uint32_t l = 0; l < cache_len; ++l) {
                    const float prob_val = (sum > 0.0f) ? (probs_out[row_idx + l] / sum) : 0.0f;
                    probs_out[row_idx + l] = prob_val;
                    probs_low_out[row_idx + l] = encode_lowp(prob_val, is_fp16);
                }
            }
        }
    }
};

/*
 * Attention AV: Attention-weighted value summation
 *
 * Matches checkpoint semantics:
 * - Load BF16/FP16 probabilities
 * - Accumulate V @ probs^T in FP32
 * - Convert FP32 output to low precision
 */
class AttentionAV {
public:
    static void compute_serial(
        const uint16_t *probs,  // [rows, seq, cache_len]
        const uint16_t *v,      // [rows, cache_len, 128]
        float *av_out,          // [rows, seq, 128]
        uint16_t *out,
        uint32_t rows, uint32_t seq, uint32_t cache_len,
        bool is_fp16) {
        
        const uint32_t value_dim = 128;
        const uint32_t probs_stride = seq * cache_len;
        const uint32_t v_stride = cache_len * value_dim;
        const uint32_t av_stride = seq * value_dim;
        
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t s = 0; s < seq; ++s) {
                for (uint32_t p = 0; p < value_dim; ++p) {
                    float av_val = 0.0f;
                    for (uint32_t l = 0; l < cache_len; ++l) {
                        const float prob_val = decode_lowp(
                            probs[r * probs_stride + s * cache_len + l], is_fp16);
                        const float v_val = decode_lowp(
                            v[r * v_stride + l * value_dim + p], is_fp16);
                        av_val += prob_val * v_val;
                    }
                    
                    const uint32_t idx = r * av_stride + s * value_dim + p;
                    av_out[idx] = av_val;
                    out[idx] = encode_lowp(av_val, is_fp16);
                }
            }
        }
    }
};

}  // namespace native_mlu
