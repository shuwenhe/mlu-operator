#ifndef PHASE9_NATIVE_AV_H
#define PHASE9_NATIVE_AV_H

#include <cnrt.h>
#include <cstdint>

// Caller-owned, disjoint device buffers. Probabilities are [heads, seq, length],
// values are [heads, capacity, 128], and outputs are [heads, seq, 128].
// The probabilities already encode masking. The caller owns queue completion.
extern "C" cnrtRet_t phase9_av_bf16(cnrtQueue_t, const uint16_t *, const uint16_t *,
    float *, uint16_t *, uint32_t heads, uint32_t seq, uint32_t length, uint32_t capacity);
extern "C" cnrtRet_t phase9_av_fp16(cnrtQueue_t, const uint16_t *, const uint16_t *,
    float *, uint16_t *, uint32_t heads, uint32_t seq, uint32_t length, uint32_t capacity);
#endif
