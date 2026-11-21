#include <bang.h>
#include "gemm_common.h"

__mlu_global__ void GEMM_KERNEL(const GEMM_INPUT *a, const GEMM_INPUT *b,
                                float *c, int m, int n, int k) {
    constexpr int tile = 128;
    __nram__ GEMM_INPUT a_low[tile], b_low[tile];
    __nram__ float a_wide[tile], b_wide[tile], accum[tile];
    const uint64_t strips = (static_cast<uint64_t>(n) + tile - 1) / tile;
    const uint64_t tiles = static_cast<uint64_t>(m) * strips;
    for (uint64_t id = taskId; id < tiles; id += taskDim) {
        const uint64_t row = id / strips;
        const uint64_t col = (id % strips) * tile;
        const int cols = n - col < tile ? static_cast<int>(n - col) : tile;
        for (int j = 0; j < cols; ++j)
            accum[j] = 0.0f;
        for (uint64_t start = 0; start < static_cast<uint64_t>(k); start += tile) {
            const int count = k - start < tile ? static_cast<int>(k - start) : tile;
            __memcpy(a_low, a + row * k + start, count * sizeof(GEMM_INPUT), GDRAM2NRAM);
            for (int p = count; p < tile; ++p)
                a_low[p] = GEMM_INPUT{};
            GEMM_CONVERT(a_wide, a_low, tile);
            for (int p = 0; p < count; ++p) {
                __memcpy(b_low, b + (start + p) * n + col, cols * sizeof(GEMM_INPUT), GDRAM2NRAM);
                for (int j = cols; j < tile; ++j)
                    b_low[j] = GEMM_INPUT{};
                GEMM_CONVERT(b_wide, b_low, tile);
                for (int j = 0; j < cols; ++j)
                    accum[j] += a_wide[p] * b_wide[j];
            }
        }
        __memcpy(c + row * n + col, accum, cols * sizeof(float), NRAM2GDRAM);
    }
}

extern "C" cnrtRet_t GEMM_LAUNCH(cnrtQueue_t queue, const uint16_t *a,
                                  const uint16_t *b, float *c, int m, int n, int k) {
    if (!queue || !a || !b || !c || m <= 0 || n <= 0 || k <= 0)
        return cnrtErrorArgsInvalid;
    const uint64_t tiles = static_cast<uint64_t>(m) * ((static_cast<uint64_t>(n) + 127) / 128);
    const cnrtDim3_t grid = {static_cast<unsigned>(tiles < 32 ? tiles : 32), 1, 1};
    GEMM_KERNEL<<<grid, cnrtFuncTypeBlock, queue>>>(
        reinterpret_cast<const GEMM_INPUT *>(a), reinterpret_cast<const GEMM_INPUT *>(b),
        c, m, n, k);
    return cnrtGetLastError();
}
