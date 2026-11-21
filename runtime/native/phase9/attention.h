/*
 * Native MLU Attention Kernels — Phase 9
 *
 * Three independent operations on MLU:
 * 1. QK: Query-Key multiplication with scaling and masking
 * 2. Softmax: Row-wise exponential normalization
 * 3. AV: Attention-weighted value summation
 *
 * All kernels preserve checkpoint semantics:
 * - FP32 intermediate precision except model boundaries
 * - BF16/FP16 input/output with rounding per Phase 8.2
 * - Causal attention mask (-INF for future)
 * - YaRN scale factor: 192^(-1/2) * yarn_mscale(40, 0.707)^2
 */

#ifndef NATIVE_MLU_ATTENTION_H
#define NATIVE_MLU_ATTENTION_H

#include <cstdint>

/*
 * Query-Key multiplication and scaling
 *
 * Input:
 *   - q: [B*heads, S, 192] BF16/FP16 row-major
 *   - k: [B*heads, L, 192] BF16/FP16 row-major
 *   - scale: 192^(-1/2) * yarn_mscale^2 (FP32)
 *   - positions: [S] token position array (for causal mask)
 *   - cache_base: cache read position (for prefill vs. decode)
 *
 * Output:
 *   - qk: [B*heads, S, L] FP32 raw scores (before scaling)
 *   - scaled: [B*heads, S, L] BF16/FP16 scaled scores
 *   - masked: [B*heads, S, L] FP32 scores with -INF at masked positions
 *
 * Per row [b, s]:
 *   1. Accumulate: qk[b,s,l] = ∑[p] q[b,s,p] * k[b,l,p]  (FP32)
 *   2. Scale: rounded(qk * scale)  (matches BF16 rounding)
 *   3. Mask: if l > positions[s], set to -INF
 */
extern "C" uint32_t phase9_attention_qk_bf16(
    uint16_t *q_data,         /* input Q: [B*heads, S, 192] */
    uint16_t *k_data,         /* input K: [B*heads, L, 192] */
    float scale,              /* 192^(-1/2) * yarn_mscale^2 */
    const uint32_t *positions,/* [S] positions */
    uint32_t cache_base,      /* causal mask offset */
    float *qk_out,            /* output: [B*heads, S, L] FP32 raw */
    uint16_t *scaled_out,     /* output: [B*heads, S, L] BF16 */
    float *masked_out,        /* output: [B*heads, S, L] FP32 masked */
    uint32_t rows, uint32_t seq, uint32_t cache_len, uint32_t dim);

extern "C" uint32_t phase9_attention_qk_fp16(
    uint16_t *q_data, uint16_t *k_data, float scale,
    const uint32_t *positions, uint32_t cache_base,
    float *qk_out, uint16_t *scaled_out, float *masked_out,
    uint32_t rows, uint32_t seq, uint32_t cache_len, uint32_t dim);

/*
 * Row-wise softmax with FP32 precision
 *
 * Input:
 *   - masked: [B*heads, S, L] FP32 scores with causal mask
 *   - seq_len: actual cache length for each request (for padding)
 *
 * Output:
 *   - maxes: [B*heads, S] FP32 row maximums
 *   - denom: [B*heads, S] FP32 denominators (∑ exp)
 *   - probs: [B*heads, S, L] FP32 normalized probabilities
 *   - probs_low: [B*heads, S, L] BF16/FP16 converted probs
 *
 * Per row [b, s]:
 *   1. Find max over L
 *   2. Exponentiate: exp(masked[b,s,l] - max)
 *   3. Sum: denom = ∑ exp values
 *   4. Normalize: probs[b,s,l] = exp(...) / denom
 *   5. Convert to BF16/FP16 for output
 */
extern "C" uint32_t phase9_attention_softmax_bf16(
    float *masked_data,       /* input: [B*heads, S, L] FP32 */
    uint32_t seq_len,         /* cache length for causal masking */
    float *maxes_out,         /* output: [B*heads, S] FP32 max */
    float *denom_out,         /* output: [B*heads, S] FP32 denom */
    float *probs_out,         /* output: [B*heads, S, L] FP32 normalized */
    uint16_t *probs_low_out,  /* output: [B*heads, S, L] BF16 */
    uint32_t rows, uint32_t seq, uint32_t cache_len);

extern "C" uint32_t phase9_attention_softmax_fp16(
    float *masked_data, uint32_t seq_len,
    float *maxes_out, float *denom_out, float *probs_out, uint16_t *probs_low_out,
    uint32_t rows, uint32_t seq, uint32_t cache_len);

/*
 * Attention-weighted value summation
 *
 * Input:
 *   - probs_low: [B*heads, S, L] BF16/FP16 probabilities
 *   - v: [B*heads, L, 128] BF16/FP16 values
 *
 * Output:
 *   - av: [B*heads, S, 128] FP32 accumulated attention
 *   - out: [B*heads, S, 128] BF16/FP16 output
 *
 * Per row [b, s]:
 *   av[b,s,p] = ∑[l] probs[b,s,l] * v[b,l,p]  (FP32 accumulation)
 *   out[b,s,p] = round(av[b,s,p])  (to BF16/FP16)
 */
extern "C" uint32_t phase9_attention_av_bf16(
    uint16_t *probs_data,     /* input: [B*heads, S, L] BF16 */
    uint16_t *v_data,         /* input: [B*heads, L, 128] BF16 */
    float *av_out,            /* output: [B*heads, S, 128] FP32 */
    uint16_t *out,            /* output: [B*heads, S, 128] BF16 */
    uint32_t rows, uint32_t seq, uint32_t cache_len);

extern "C" uint32_t phase9_attention_av_fp16(
    uint16_t *probs_data, uint16_t *v_data,
    float *av_out, uint16_t *out,
    uint32_t rows, uint32_t seq, uint32_t cache_len);

#endif /* NATIVE_MLU_ATTENTION_H */
