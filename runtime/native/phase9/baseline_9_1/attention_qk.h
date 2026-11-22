#ifndef NATIVE_MLU_ATTENTION_QK_H
#define NATIVE_MLU_ATTENTION_QK_H

#include <cnrt.h>
#include <cstdint>

// All pointers address caller-owned device memory. Q is [heads, seq, 192],
// K is [heads, capacity, 192], and each output is [heads, seq, length].
// positions is [seq], shared across heads of one request. Key index j is
// visible when j <= cache_base + positions[s]. Raw holds the FP32 dot;
// scaled holds the two-rounding eager score; masked widens scaled and
// replaces future scores with negative infinity. Buffers must be disjoint
// and remain valid until the caller's queue completes.
extern "C" cnrtRet_t phase9_launch_qk_bf16(
    cnrtQueue_t queue, const uint16_t *q, const uint16_t *k,
    const uint32_t *positions, uint32_t cache_base, float scale,
    float *raw, uint16_t *scaled, float *masked,
    uint32_t heads, uint32_t seq, uint32_t length, uint32_t capacity);

extern "C" cnrtRet_t phase9_launch_qk_fp16(
    cnrtQueue_t queue, const uint16_t *q, const uint16_t *k,
    const uint32_t *positions, uint32_t cache_base, float scale,
    float *raw, uint16_t *scaled, float *masked,
    uint32_t heads, uint32_t seq, uint32_t length, uint32_t capacity);

#endif
