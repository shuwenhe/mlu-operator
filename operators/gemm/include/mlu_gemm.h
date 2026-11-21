#ifndef MLU_OPERATOR_MLU_NATIVE_GEMM_H
#define MLU_OPERATOR_MLU_NATIVE_GEMM_H

#include <stddef.h>
#include <stdint.h>

#include "mlu_types.h"

typedef struct {
    double h2d_us;
    double kernel_us;
    double d2h_us;
    size_t workspace_bytes;
    size_t device_bytes;
} mlu_gemm_metrics;

/* Packed row-major A[M,K], B[K,N] and output[M,N].
 * A and B contain raw 16-bit floats. Kernel time is the mean of repeated launches.
 * A call owns its queue and device allocations. It must run on a dedicated worker
 * thread because CNRT device selection is thread-local and persists after return.
 */
mlu_status mlu_gemm(int device, mlu_dtype dtype,
                                int m, int n, int k, const uint16_t *a,
                                const uint16_t *b, float *output, int repeats,
                                mlu_gemm_metrics *metrics);

#endif
