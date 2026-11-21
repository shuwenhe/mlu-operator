#ifndef MLU_NATIVE_GEMM_COMMON_H
#define MLU_NATIVE_GEMM_COMMON_H
#include <cnrt.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/* Device buffers. On successful enqueue, synchronize queue before reading C.
 * Buffers contain packed row-major matrices and must not overlap.
 */
cnrtRet_t mlu_bang_gemm_fp16(cnrtQueue_t queue, const uint16_t *a,
    const uint16_t *b, float *c, int m, int n, int k);
cnrtRet_t mlu_bang_gemm_bf16(cnrtQueue_t queue, const uint16_t *a,
    const uint16_t *b, float *c, int m, int n, int k);
#ifdef __cplusplus
}
#endif
#endif
