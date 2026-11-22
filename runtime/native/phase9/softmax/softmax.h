#ifndef PHASE9_NATIVE_SOFTMAX_H
#define PHASE9_NATIVE_SOFTMAX_H

#include <cnrt.h>
#include <cstdint>

// Contiguous device buffers. Scores are widened checkpoint low-precision logits.
// Optional additive mask uses the same low dtype as the output, with mask_rows
// repeating across heads (1, query length, or the entire flattened row count).
// Existing -inf scores are causal masks. Columns >= valid are also masked.
// Fully masked rows produce NaNs as in eager torch.softmax. Queue ownership and
// buffer lifetimes belong to the caller. Buffers must be mutually disjoint.
extern "C" cnrtRet_t phase9_softmax_bf16(
    cnrtQueue_t, const float *scores, const uint16_t *mask, uint32_t mask_rows,
    float *maxes, float *denom, float *probs, uint16_t *low,
    uint32_t rows, uint32_t cols, uint32_t valid);
extern "C" cnrtRet_t phase9_softmax_fp16(
    cnrtQueue_t, const float *scores, const uint16_t *mask, uint32_t mask_rows,
    float *maxes, float *denom, float *probs, uint16_t *low,
    uint32_t rows, uint32_t cols, uint32_t valid);
#endif
