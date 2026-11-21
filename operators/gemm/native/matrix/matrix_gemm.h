#ifndef MLU_MATRIX_GEMM_H
#define MLU_MATRIX_GEMM_H
#include <cnrt.h>
#include <stdint.h>

constexpr int matrix_m = 16, matrix_k = 128, matrix_n = 128;
constexpr int matrix_rhs_slot(int k, int n) {
    return (8*((n%64)/4) + 4*(n/64) + n%4)*matrix_k + k;
}
// Device A and C are row major; B is packed by N tile, then K tile.
// Pad the packed B tiles with zero for logical K and N tails.
extern "C" cnrtRet_t matrix_gemm_fp16(cnrtQueue_t, const uint16_t *, const uint16_t *,
                                       float *, int m, int n, int k);
extern "C" cnrtRet_t matrix_gemm_bf16(cnrtQueue_t, const uint16_t *, const uint16_t *,
                                       float *, int m, int n, int k);
#endif
