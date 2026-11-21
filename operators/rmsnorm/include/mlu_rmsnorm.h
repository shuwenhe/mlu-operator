#ifndef MLU_OPERATOR_RMSNORM_H
#define MLU_OPERATOR_RMSNORM_H

#include <stdint.h>

#include "mlu_types.h"

/* Packed X[rows,width], gamma[width] and output[rows,width].
 * Device selection persists on the calling thread.
 */
mlu_status mlu_rmsnorm(int device, mlu_dtype dtype, int rows, int width,
                      float epsilon, const uint16_t *x, const float *gamma,
                      float *output);

#endif
