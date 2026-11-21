#ifndef MLU_NATIVE_RMSNORM_H
#define MLU_NATIVE_RMSNORM_H
#include <cnrt.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Packed, nonoverlapping device buffers. Gamma has width elements.
 * Optional diagnostics have rows*width elements. Synchronize after enqueue.
 */
cnrtRet_t bang_rmsnorm_fp16(cnrtQueue_t, const uint16_t *x, const uint16_t *gamma,
    uint16_t *out, float *normalized, uint16_t *rounded, int rows, int width, float eps);
cnrtRet_t bang_rmsnorm_bf16(cnrtQueue_t, const uint16_t *x, const uint16_t *gamma,
    uint16_t *out, float *normalized, uint16_t *rounded, int rows, int width, float eps);
#ifdef __cplusplus
}
#endif
#endif
